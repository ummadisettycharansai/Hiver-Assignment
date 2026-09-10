import argparse
import sys
from typing import Optional
from pathlib import Path

# Add package root
sys.path.insert(0, str(Path(__file__).parent.parent))

from hiver_agent.config import get_config
from hiver_agent.intents.classifier import DenseEmbeddingClassifier
from hiver_agent.retrieval.index import VectorStoreIndex
from hiver_agent.generation.llm_provider import LLMProvider
from hiver_agent.generation.generator import GroundedReplyGenerator
from hiver_agent.decision.policy import DecisionPolicy

def run_interactive_demo(query_text: Optional[str] = None):
    config = get_config()
    
    print("\n" + "="*70)
    print(f"       HIVER AI SUPPORT AGENT - CLI DEMO (@{config.brand_name})")
    print("="*70)
    
    clf = DenseEmbeddingClassifier(model_name=config.embedding_model)
    clf.load("results/models/dense_proposed.pkl")
    
    index = VectorStoreIndex(embedding_model=config.embedding_model)
    index.load("data/processed/retrieval_index")
    
    provider = LLMProvider(provider=config.llm_provider, model=config.llm_model)
    generator = GroundedReplyGenerator(provider=provider, brand_name=config.brand_name)
    policy = DecisionPolicy(
        min_intent_confidence=config.min_intent_confidence,
        min_retrieval_similarity=config.min_retrieval_similarity
    )
    
    user_msg = query_text or "I have been waiting for my refund for two weeks and no money in my bank account."
    print(f"\n[CUSTOMER MESSAGE]\n\"{user_msg}\"\n")
    
    # 1. Intent prediction
    clf_res = clf.predict_one(user_msg)
    intent = clf_res["intent"]
    conf = clf_res["confidence"]
    print(f"[PREDICTED INTENT]\nIntent: {intent:<25} (Confidence: {conf:.2f})")
    if clf_res.get("alternatives"):
        print("Alternatives:", ", ".join([f"{a['intent']} ({a['confidence']:.2f})" for a in clf_res["alternatives"]]))
        
    # 2. Historical retrieval
    retrieved = index.search(user_msg, top_k=config.retrieval_top_k)
    print(f"\n[RETRIEVED HISTORICAL EVIDENCE ({len(retrieved)} cases)]")
    for idx, case in enumerate(retrieved, 1):
        print(f" {idx}. [Score {case['similarity_score']:.2f}] Customer: \"{case['customer_message'][:60]}...\"")
        print(f"    Brand Resolution: \"{case['brand_reply'][:80]}...\"")

    # 3. Grounded reply generation
    gen_res = generator.generate_reply(
        customer_message=user_msg,
        predicted_intent=intent,
        intent_confidence=conf,
        retrieved_evidence=retrieved
    )
    print(f"\n[DRAFTED REPLY]\n\"{gen_res['reply']}\"")
    
    # 4. Decision policy
    dec_res = policy.evaluate_decision(
        intent=intent,
        intent_confidence=conf,
        retrieved_evidence=retrieved,
        generation_result=gen_res
    )
    
    dec = dec_res["decision"]
    reason = dec_res["reason"]
    
    color = "\033[92m" if dec == "AUTO_HANDLE" else "\033[91m"
    reset = "\033[0m"
    print(f"\n[DECISION]: {color}{dec}{reset}")
    print(f"[REASON]  : {reason}")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLI support agent demo.")
    parser.add_argument("--query", type=str, default=None, help="Custom customer message query")
    args = parser.parse_args()
    run_interactive_demo(args.query)
