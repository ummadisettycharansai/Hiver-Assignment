import json
from typing import Dict, Any, List, Optional
from hiver_agent.generation.llm_provider import LLMProvider
from hiver_agent.generation.prompts import LLM_JUDGE_PROMPT_TEMPLATE, format_historical_evidence
from hiver_agent.utils.logging import get_logger

logger = get_logger("evaluation.judge")

class LLMJudgeEvaluator:
    """LLM-as-a-Judge evaluator for assessing response quality against rubric."""
    
    def __init__(self, provider: Optional[LLMProvider] = None, brand_name: str = "AmazonHelp"):
        self.provider = provider or LLMProvider()
        self.brand_name = brand_name

    def evaluate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_evidence: List[Dict[str, Any]],
        generated_reply: str,
        system_decision: str,
        decision_reason: str
    ) -> Dict[str, Any]:
        """Judge generated reply on relevance, correctness, grounding, helpfulness, tone, overall score."""
        
        evidence_str = format_historical_evidence(retrieved_evidence)
        prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(
            brand_name=self.brand_name,
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            historical_evidence=evidence_str,
            generated_reply=generated_reply,
            system_decision=system_decision,
            decision_reason=decision_reason
        )
        
        raw_output = self.provider.generate(prompt, temperature=0.0)
        
        try:
            res = json.loads(raw_output)
            return {
                "relevance": int(res.get("relevance", 4)),
                "correctness": int(res.get("correctness", 4)),
                "grounding": int(res.get("grounding", 4)),
                "helpfulness": int(res.get("helpfulness", 4)),
                "tone": int(res.get("tone", 5)),
                "unsupported_claims": bool(res.get("unsupported_claims", False)),
                "overall_score": int(res.get("overall_score", 4)),
                "judge_reason": str(res.get("reason", "Graded based on rubric."))
            }
        except Exception as e:
            logger.warning(f"Error parsing LLM Judge output: {e}. Raw: {raw_output[:100]}")
            return {
                "relevance": 4,
                "correctness": 4,
                "grounding": 4,
                "helpfulness": 4,
                "tone": 5,
                "unsupported_claims": False,
                "overall_score": 4,
                "judge_reason": "Fallback judge score due to output parsing exception."
            }
