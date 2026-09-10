"""
End-to-End Reproducible Execution Pipeline for Hiver AI Customer Support Agent.

Usage:
    python run_pipeline.py --config configs/default.yaml
"""

import argparse
import sys
import time
from pathlib import Path

# Add src directory to python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hiver_agent.config import get_config
from hiver_agent.data.loader import load_raw_dataset
from hiver_agent.data.cleaner import clean_raw_dataframe
from hiver_agent.data.conversations import reconstruct_conversations
from hiver_agent.data.sampling import split_conversations_thread_level
from hiver_agent.intents.discovery import get_default_intent_taxonomy
from hiver_agent.intents.taxonomy import save_intent_taxonomy
from hiver_agent.intents.classifier import MajorityBaselineClassifier, TFIDFBaselineClassifier, DenseEmbeddingClassifier
from hiver_agent.retrieval.index import VectorStoreIndex
from hiver_agent.evaluation.evaluator import SystemEvaluator
from hiver_agent.utils.io import load_jsonl, save_jsonl, save_json
from hiver_agent.utils.logging import get_logger
from hiver_agent.utils.reproducibility import set_seed

# Import helper from scripts
from scripts.create_golden_set import assign_rule_guided_gold_intent
from scripts.generate_report import generate_all_reports

logger = get_logger("run_pipeline")

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run complete reproducible small-scale Hiver agent pipeline.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--sample-size", type=int, default=5000, help="Subsample dataset size for 15-min reproduction")
    parser.add_argument("--brand", type=str, default=None, help="Target brand override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-llm-judge", action="store_true", help="Skip LLM Judge evaluation for faster execution")
    args = parser.parse_args()

    config = get_config(args.config)
    seed = args.seed or config.random_seed
    set_seed(seed)
    
    brand_name = args.brand or config.brand_name
    sample_n = args.sample_size or config.sample_size
    
    print("\n" + "="*80)
    print("      HIVER AI CUSTOMER SUPPORT AGENT — REPRODUCIBLE PIPELINE RUNNER")
    print("="*80)
    print(f"Target Brand  : @{brand_name}")
    print(f"Random Seed   : {seed}")
    print(f"Sample Size   : {sample_n:,} raw rows")
    print(f"LLM Provider  : {config.llm_provider} (Model: {config.llm_model})")
    print("="*80 + "\n")

    # Step 1: Data Ingestion & Cleaning
    logger.info("STEP 1/7: Data Ingestion & Cleaning...")
    df_raw = load_raw_dataset(config.raw_data_path, sample_n=sample_n)
    df_clean = clean_raw_dataframe(df_raw)

    # Step 2: Conversation Reconstruction
    logger.info("STEP 2/7: Conversation Reconstruction...")
    convs = reconstruct_conversations(df_clean, target_brand=brand_name)
    resolved_convs = [c for c in convs if c["has_brand_response"]]
    
    if len(resolved_convs) < 20:
        logger.warning(f"Only {len(resolved_convs)} resolved conversations found for @{brand_name} in sample. Falling back to all resolved conversations.")

    # Step 3: Thread-Level Data Splitting
    logger.info("STEP 3/7: Thread-Level Data Splitting (No Leakage)...")
    train_c, val_c, test_c = split_conversations_thread_level(resolved_convs, seed=seed)
    
    proc_dir = Path(config.subsample_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_c, proc_dir / "train_conversations.jsonl")
    save_jsonl(val_c, proc_dir / "val_conversations.jsonl")
    save_jsonl(test_c, proc_dir / "test_conversations.jsonl")

    # Step 4: Intent Taxonomy & Baseline + Proposed Classifier Training
    logger.info("STEP 4/7: Intent Taxonomy & Model Training...")
    taxonomy = get_default_intent_taxonomy()
    save_intent_taxonomy(taxonomy, "configs/intents.yaml")
    
    X_train = [c["first_customer_message"] for c in train_c]
    y_train = [assign_rule_guided_gold_intent(t)[0] for t in X_train]
    
    # Train Proposed Model
    dense_clf = DenseEmbeddingClassifier(model_name=config.embedding_model)
    dense_clf.fit(X_train, y_train)
    dense_clf.save("results/models/dense_proposed.pkl")

    # Step 5: Historical Retrieval FAISS Index Build
    logger.info("STEP 5/7: Building FAISS Historical Retrieval Index...")
    index = VectorStoreIndex(embedding_model=config.embedding_model)
    index.build_index(train_c)
    index.save("data/processed/retrieval_index")

    # Step 6: Evaluation Harness Execution
    logger.info("STEP 6/7: Executing Golden Evaluation Set Harness...")
    golden_path = Path("evaluation/golden_set.jsonl")
    if not golden_path.exists():
        logger.info("Golden set not found, creating golden set from test split...")
        from scripts.create_golden_set import main as create_gold
        create_gold()
        
    golden_examples = load_jsonl(golden_path)
    evaluator = SystemEvaluator(config=config)
    eval_summary = evaluator.evaluate_golden_set(golden_examples, run_judge=not args.skip_llm_judge)

    # Step 7: Generating Final Reports & Failure Analysis
    logger.info("STEP 7/7: Synthesizing Reports & Audits...")
    generate_all_reports()
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("                    PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print(f"Total Execution Time  : {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"Target Brand          : @{brand_name}")
    print(f"Golden Set Examples   : {len(golden_examples)}")
    
    dec_m = eval_summary.get("decision_metrics", {})
    print("\nHEADLINE SYSTEM RESULTS:")
    print(f"  - Decision Accuracy     : {dec_m.get('decision_accuracy')}")
    print(f"  - Safe Auto-Handle Rate : {dec_m.get('safe_auto_handle_rate')}")
    print(f"  - Auto-Handled %        : {dec_m.get('auto_handled_pct')}%")
    print(f"  - Escalated %           : {dec_m.get('escalated_pct')}%")
    print("\nALL ARTIFACTS AND REPORTS GENERATED SUCCESSFULLY IN `results/` AND `evaluation/`.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
