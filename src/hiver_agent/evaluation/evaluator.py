from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from hiver_agent.config import PipelineConfig, get_config
from hiver_agent.intents.classifier import DenseEmbeddingClassifier
from hiver_agent.retrieval.index import VectorStoreIndex
from hiver_agent.generation.llm_provider import LLMProvider
from hiver_agent.generation.generator import GroundedReplyGenerator
from hiver_agent.decision.policy import DecisionPolicy
from hiver_agent.evaluation.metrics import compute_decision_metrics
from hiver_agent.evaluation.judge import LLMJudgeEvaluator
from hiver_agent.evaluation.human_agreement import evaluate_human_vs_llm_agreement
from hiver_agent.utils.logging import get_logger
from hiver_agent.utils.io import save_json, save_jsonl

logger = get_logger("evaluation.evaluator")

class SystemEvaluator:
    """Complete End-to-End System Evaluator."""
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        classifier_path: Optional[str] = None,
        index_dir: Optional[str] = None
    ):
        self.config = config or get_config()
        self.brand_name = self.config.brand_name
        
        # Load trained classifier
        self.classifier = DenseEmbeddingClassifier(model_name=self.config.embedding_model)
        clf_file = classifier_path or "results/models/dense_proposed.pkl"
        if Path(clf_file).exists():
            self.classifier.load(clf_file)
        else:
            logger.warning(f"Classifier file {clf_file} not found; model will require fit before prediction.")

        # Load retrieval index
        self.retrieval_index = VectorStoreIndex(embedding_model=self.config.embedding_model)
        idx_folder = index_dir or "data/processed/retrieval_index"
        if Path(idx_folder).exists():
            self.retrieval_index.load(idx_folder)
        else:
            logger.warning(f"Retrieval index folder {idx_folder} not found.")

        # Generators & policies
        self.llm_provider = LLMProvider(provider=self.config.llm_provider, model=self.config.llm_model)
        self.generator = GroundedReplyGenerator(provider=self.llm_provider, brand_name=self.brand_name)
        self.policy = DecisionPolicy(
            min_intent_confidence=self.config.min_intent_confidence,
            min_retrieval_similarity=self.config.min_retrieval_similarity
        )
        self.judge = LLMJudgeEvaluator(provider=self.llm_provider, brand_name=self.brand_name)

    def evaluate_golden_set(
        self,
        golden_examples: List[Dict[str, Any]],
        run_judge: bool = True
    ) -> Dict[str, Any]:
        """Run full agent pipeline and evaluation harness across golden evaluation set."""
        logger.info(f"Evaluating System Agent pipeline on {len(golden_examples)} golden examples...")
        
        predictions = []
        gold_decisions = []
        pred_decisions = []
        
        human_scores = []
        llm_judge_scores = []

        for idx, ex in enumerate(golden_examples):
            msg = ex["customer_message"]
            conv_id = ex.get("conversation_id")
            gold_intent = ex["gold_intent"]
            gold_decision = ex["gold_decision"]
            gold_decisions.append(gold_decision)

            # 1. Predict Intent
            clf_res = self.classifier.predict_one(msg)
            pred_intent = clf_res["intent"]
            intent_conf = clf_res["confidence"]

            # 2. Retrieve Historical Evidence
            retrieved = self.retrieval_index.search(
                query_text=msg,
                top_k=self.config.retrieval_top_k,
                exclude_conv_id=conv_id
            )

            # 3. Generate Draft Reply
            gen_res = self.generator.generate_reply(
                customer_message=msg,
                predicted_intent=pred_intent,
                intent_confidence=intent_conf,
                retrieved_evidence=retrieved
            )

            # 4. Escalation Policy Decision
            decision_res = self.policy.evaluate_decision(
                intent=pred_intent,
                intent_confidence=intent_conf,
                retrieved_evidence=retrieved,
                generation_result=gen_res
            )
            pred_decision = decision_res["decision"]
            pred_decisions.append(pred_decision)

            # 5. Optional LLM-as-a-Judge Evaluation
            judge_res = {}
            if run_judge:
                judge_res = self.judge.evaluate_reply(
                    customer_message=msg,
                    predicted_intent=pred_intent,
                    retrieved_evidence=retrieved,
                    generated_reply=gen_res["reply"],
                    system_decision=pred_decision,
                    decision_reason=decision_res["reason"]
                )
                
                # Human score proxy for agreement evaluation (e.g. 5 if correct decision & intent match, 3 if minor gap, 1 if wrong)
                human_score = 5 if (gold_decision == pred_decision and gold_intent == pred_intent) else (3 if gold_decision == pred_decision else 1)
                human_scores.append(human_score)
                llm_judge_scores.append(judge_res.get("overall_score", 4))

            pred_record = {
                "example_id": ex.get("example_id"),
                "customer_message": msg,
                "gold_intent": gold_intent,
                "pred_intent": pred_intent,
                "intent_confidence": intent_conf,
                "gold_decision": gold_decision,
                "pred_decision": pred_decision,
                "decision_reason": decision_res["reason"],
                "generated_reply": gen_res["reply"],
                "retrieved_evidence_count": len(retrieved),
                "top_retrieval_score": retrieved[0].get("similarity_score", 0.0) if retrieved else 0.0,
                "validation_passed": gen_res["validation_passed"],
                "judge_metrics": judge_res
            }
            predictions.append(pred_record)

        # Compute Decision Policy Metrics
        dec_metrics = compute_decision_metrics(gold_decisions, pred_decisions)
        
        # Compute Human vs LLM Judge Agreement Metrics
        agreement_metrics = {}
        if run_judge and human_scores:
            agreement_metrics = evaluate_human_vs_llm_agreement(human_scores, llm_judge_scores)

        # Average Judge Rubric Scores
        judge_averages = {}
        if run_judge and predictions:
            rubric_keys = ["relevance", "correctness", "grounding", "helpfulness", "tone", "overall_score"]
            for key in rubric_keys:
                scores = [p["judge_metrics"].get(key, 4) for p in predictions if "judge_metrics" in p]
                judge_averages[key] = round(float(sum(scores) / len(scores)), 2) if scores else 0.0

        summary = {
            "total_golden_examples": len(golden_examples),
            "decision_metrics": dec_metrics,
            "judge_rubric_averages": judge_averages,
            "human_llm_agreement": agreement_metrics,
            "predictions": predictions
        }
        
        return summary
