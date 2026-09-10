import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hiver_agent.config import get_config
from hiver_agent.intents.classifier import DenseEmbeddingClassifier, TFIDFBaselineClassifier
from hiver_agent.retrieval.index import VectorStoreIndex
from hiver_agent.generation.llm_provider import LLMProvider
from hiver_agent.generation.generator import GroundedReplyGenerator
from hiver_agent.decision.policy import DecisionPolicy
from hiver_agent.utils.io import load_jsonl, load_json

def run_deep_audit():
    config = get_config()
    proc_dir = Path("data/processed")
    
    print("="*80)
    print("         INDEPENDENT COMPREHENSIVE REPOSITORY & EVIDENCE AUDIT")
    print("="*80)
    
    # 1. Real LLM vs Mock Audit
    provider = LLMProvider(provider=config.llm_provider, model=config.llm_model)
    print(f"\n1. LLM IMPLEMENTATION AUDIT:")
    print(f"   Configured Provider : {provider.provider}")
    print(f"   Configured Model    : {provider.model}")
    print(f"   OpenAI API Key Set  : {bool(provider.openai_key)}")
    print(f"   Anthropic Key Set   : {bool(provider.anthropic_key)}")
    if provider.provider == "mock":
        print("   NOTE: Headline evaluation runs use deterministic Mock LLM responses with SHA-256 disk caching.")

    # 2. Selected Brand Audit
    print(f"\n2. SELECTED BRAND AUDIT:")
    brand_rep = load_json("results/brand_selection_report.json")
    print(f"   Selected Brand      : @{brand_rep['selected_brand']}")
    print(f"   Rationale           : {brand_rep['selection_rationale']}")

    # 3. Golden Set Audit
    golden_examples = load_jsonl("evaluation/golden_set.jsonl")
    df_gold = pd.DataFrame(golden_examples)
    print(f"\n3. GOLDEN EVALUATION SET AUDIT:")
    print(f"   Total Examples      : {len(golden_examples)}")
    print(f"   Intent Distribution :\n{df_gold['gold_intent'].value_counts().to_string()}")
    print(f"   Decision Distribution:\n{df_gold['gold_decision'].value_counts().to_string()}")
    print(f"   Difficulty Breakdown :\n{df_gold['difficulty'].value_counts().to_string()}")
    print(f"   Human Annotated Flag : {df_gold.get('human_annotated', pd.Series([False]*len(df_gold))).value_counts().to_dict()}")

    # 4. Retrieval Verification (10 Examples)
    print(f"\n4. HISTORICAL RETRIEVAL AUDIT (10 Examples):")
    index = VectorStoreIndex(embedding_model=config.embedding_model)
    index.load(proc_dir / "retrieval_index")
    
    sample_exs = golden_examples[:10]
    for idx, ex in enumerate(sample_exs, 1):
        msg = ex["customer_message"]
        conv_id = ex.get("conversation_id")
        results = index.search(msg, top_k=2, exclude_conv_id=conv_id)
        
        print(f"\n   Ex #{idx} [ID: {ex['example_id']}]")
        print(f"   Query Text    : \"{msg}\"")
        if results:
            print(f"   Retrieved #1  : [ID: {results[0]['conversation_id']} | Score: {results[0]['similarity_score']:.4f}]")
            print(f"                   Customer: \"{results[0]['customer_message'][:50]}...\"")
            print(f"                   Brand   : \"{results[0]['brand_reply'][:60]}...\"")
            # Self retrieval check
            assert results[0]['conversation_id'] != conv_id, "SELF RETRIEVAL LEAKAGE DETECTED!"
        else:
            print("   Retrieved     : None")

    # 5. Grounding & Safety Audit (10 Examples)
    print(f"\n5. GROUNDING & SAFETY AUDIT (10 Examples):")
    generator = GroundedReplyGenerator(provider=provider, brand_name=config.brand_name)
    clf = DenseEmbeddingClassifier(model_name=config.embedding_model)
    clf.load("results/models/dense_proposed.pkl")
    
    grounded_count = 0
    for idx, ex in enumerate(sample_exs, 1):
        msg = ex["customer_message"]
        clf_res = clf.predict_one(msg)
        retrieved = index.search(msg, top_k=2, exclude_conv_id=ex.get("conversation_id"))
        gen_res = generator.generate_reply(msg, clf_res["intent"], clf_res["confidence"], retrieved)
        
        passed = gen_res["validation_passed"]
        if passed:
            grounded_count += 1
        print(f"   Ex #{idx} | Passed Safety Validation: {passed} | Reply: \"{gen_res['reply'][:60]}...\"")

    print(f"   Safety/Grounding Pass Rate: {grounded_count / 10 * 100:.1f}%")

    # 6. Escalation Signals Audit (20 Examples)
    print(f"\n6. ESCALATION POLICY SIGNALS AUDIT (20 Examples):")
    policy = DecisionPolicy(min_intent_confidence=config.min_intent_confidence, min_retrieval_similarity=config.min_retrieval_similarity)
    
    exs_20 = golden_examples[:20]
    auto_cnt, esc_cnt = 0, 0
    for idx, ex in enumerate(exs_20, 1):
        msg = ex["customer_message"]
        clf_res = clf.predict_one(msg)
        retrieved = index.search(msg, top_k=2, exclude_conv_id=ex.get("conversation_id"))
        gen_res = generator.generate_reply(msg, clf_res["intent"], clf_res["confidence"], retrieved)
        dec_res = policy.evaluate_decision(clf_res["intent"], clf_res["confidence"], retrieved, gen_res)
        
        if dec_res["decision"] == "AUTO_HANDLE":
            auto_cnt += 1
        else:
            esc_cnt += 1
            
        print(f"   Ex #{idx:02d} | Intent: {clf_res['intent']:<26} | Conf: {clf_res['confidence']:.2f} | Score: {retrieved[0]['similarity_score']:.2f} | Dec: {dec_res['decision']:<11} | Reason: {dec_res['reason'][:50]}...")

    # 7. Out of Scope Class Breakdown
    print(f"\n7. OUT_OF_SCOPE_OTHER IMPACT AUDIT:")
    oos_count = sum(1 for g in golden_examples if g["gold_intent"] == "out_of_scope_other")
    print(f"   Support in Golden Set : {oos_count} / {len(golden_examples)} ({oos_count/len(golden_examples)*100:.1f}%)")
    print(f"   Explanation: High proportion caused by non-English tweets (Spanish/Portuguese/German/Japanese) in raw TWCS dataset and generic 'DM sent' mentions.")

    # 8. Headline Metric Analysis
    print(f"\n8. HEADLINE METRIC BREAKDOWN:")
    clf_res_data = load_json("results/classification_results.json")
    tfidf_gold = clf_res_data["tfidf_gold"]
    dense_gold = clf_res_data["dense_gold"]
    print(f"   TF-IDF Baseline Accuracy   : {tfidf_gold['accuracy'] * 100:.1f}% | Macro F1: {tfidf_gold['macro_f1']:.4f} | Weighted F1: {tfidf_gold['weighted_f1']:.4f}")
    print(f"   Dense Proposed Accuracy   : {dense_gold['accuracy'] * 100:.1f}% | Macro F1: {dense_gold['macro_f1']:.4f} | Weighted F1: {dense_gold['weighted_f1']:.4f}")

    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    run_deep_audit()
