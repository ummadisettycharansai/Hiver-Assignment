import argparse
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hiver_agent.config import get_config
from hiver_agent.intents.discovery import get_default_intent_taxonomy
from hiver_agent.utils.io import load_jsonl

def validate_golden_set(golden_path: str = "evaluation/golden_set.jsonl"):
    path = Path(golden_path)
    if not path.exists():
        print(f"ERROR: Golden set file not found at {path}")
        return False

    examples = load_jsonl(path)
    taxonomy = get_default_intent_taxonomy()
    valid_intents = set(taxonomy.get_intent_names())
    valid_decisions = {"AUTO_HANDLE", "ESCALATE"}

    total_count = len(examples)
    human_labelled_cnt = sum(1 for e in examples if e.get("annotated_by_human", False))

    errors = []
    warnings = []

    # Count check
    if not (150 <= total_count <= 250):
        warnings.append(f"Golden set count ({total_count}) is outside standard 150-250 range.")

    # Duplicate customer messages
    msg_counts = pd.Series([e.get("customer_message") for e in examples]).value_counts()
    dup_msgs = msg_counts[msg_counts > 1]
    if not dup_msgs.empty:
        warnings.append(f"Found {len(dup_msgs)} duplicate customer query messages.")

    # Validate each item
    for idx, ex in enumerate(examples, 1):
        ex_id = ex.get("example_id", f"idx_{idx}")
        intent = ex.get("gold_intent")
        decision = ex.get("gold_decision")
        
        if not intent or intent not in valid_intents:
            errors.append(f"[{ex_id}] Invalid gold intent: '{intent}'")
            
        if not decision or decision not in valid_decisions:
            errors.append(f"[{ex_id}] Invalid gold decision: '{decision}'")
            
        if decision == "ESCALATE" and not ex.get("escalation_reason"):
            warnings.append(f"[{ex_id}] Escalated example missing explicit escalation reason.")

    # Check leakage against training set
    config = get_config()
    train_path = Path(config.processed_data_dir) / "train_conversations.jsonl"
    if train_path.exists():
        train_convs = load_jsonl(train_path)
        train_ids = set(c["conversation_id"] for c in train_convs)
        gold_ids = set(e.get("conversation_id") for e in examples)
        leakage = train_ids & gold_ids
        if leakage:
            errors.append(f"DATA LEAKAGE DETECTED: {len(leakage)} golden conversation IDs exist in training set!")
        else:
            print("[✓] Data Leakage Check: 0 conversation ID overlap between Golden Set and Training Set.")

    print("\n" + "="*75)
    print("               GOLDEN EVALUATION SET VALIDATION REPORT")
    print("="*75)
    print(f"Golden File Path       : {path}")
    print(f"Total Examples         : {total_count}")
    print(f"Human-Labelled Count   : {human_labelled_cnt} / {total_count} ({human_labelled_cnt/total_count*100:.1f}%)")
    print(f"Validation Errors      : {len(errors)}")
    print(f"Validation Warnings    : {len(warnings)}")
    
    if errors:
        print("\nERRORS DETECTED:")
        for err in errors[:10]:
            print(f"  [X] {err}")
    else:
        print("\n[✓] ZERO ERRORS DETECTED! Golden set schema is structurally valid.")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings[:10]:
            print(f"  [!] {w}")

    print("="*75 + "\n")
    return len(errors) == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Golden Evaluation Set integrity and schema.")
    parser.add_argument("--golden-file", type=str, default="evaluation/golden_set.jsonl", help="Path to golden set jsonl")
    args = parser.parse_args()
    validate_golden_set(args.golden_file)
