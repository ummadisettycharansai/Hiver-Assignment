import argparse
import sys
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hiver_agent.utils.io import load_jsonl, save_jsonl
from hiver_agent.intents.discovery import get_default_intent_taxonomy

def display_progress_summary(examples: List[Dict[str, Any]]):
    total = len(examples)
    human_done = [e for e in examples if e.get("annotated_by_human", False)]
    completed_cnt = len(human_done)
    remaining_cnt = total - completed_cnt

    print("\n" + "="*70)
    print("                HUMAN ANNOTATION PROGRESS REPORT")
    print("="*70)
    print(f"Total Golden Examples : {total}")
    print(f"Human Labelled Done   : {completed_cnt} ({completed_cnt/total*100:.1f}%)")
    print(f"Remaining Unlabelled  : {remaining_cnt}")

    if human_done:
        df_done = pd.DataFrame(human_done)
        print("\nHuman Gold Intent Distribution:")
        for intent, count in df_done["gold_intent"].value_counts().items():
            print(f"  - {intent:<28}: {count}")
        print("\nHuman Gold Decision Distribution:")
        for dec, count in df_done["gold_decision"].value_counts().items():
            print(f"  - {dec:<28}: {count}")
    else:
        print("\nNo human labels submitted yet. All 200 examples await manual review.")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Interactive Human Annotation CLI Tool for Golden Evaluation Set.")
    parser.add_argument("--golden-file", type=str, default="evaluation/golden_set.jsonl", help="Path to golden_set.jsonl")
    parser.add_argument("--count", type=int, default=20, help="Number of unlabelled examples to annotate in this batch")
    parser.add_argument("--resume", action="store_true", help="Resume annotation, skipping already human-annotated examples")
    parser.add_argument("--status", action="store_true", help="Display current annotation progress summary and exit")
    args = parser.parse_args()

    golden_path = Path(args.golden_file)
    if not golden_path.exists():
        print(f"Error: {golden_path} does not exist. Run python scripts/create_golden_set.py first.")
        return

    examples = load_jsonl(golden_path)

    if args.status:
        display_progress_summary(examples)
        return

    taxonomy = get_default_intent_taxonomy()
    intents = taxonomy.intents
    intent_names = [i.name for i in intents]

    # Find un-annotated examples
    unannotated_indices = [idx for idx, e in enumerate(examples) if not e.get("annotated_by_human", False)]

    if not unannotated_indices:
        print("\nALL GOLDEN EXAMPLES HAVE BEEN ANNOTATED BY A HUMAN! 🎉")
        display_progress_summary(examples)
        return

    batch_indices = unannotated_indices[:args.count]

    print("\n" + "="*75)
    print("       HUMAN ANNOTATION WORKFLOW — HIVER AI SUPPORT AGENT")
    print("="*75)
    print(f"Loaded {len(examples)} total candidate examples.")
    print(f"Already Human-Annotated : {len(examples) - len(unannotated_indices)}")
    print(f"Starting batch of       : {len(batch_indices)} examples")
    print("="*75 + "\n")

    annotated_in_session = 0

    for step_num, idx in enumerate(batch_indices, 1):
        ex = examples[idx]
        
        print("\n" + "-"*75)
        print(f"Example {step_num} / {len(batch_indices)} (ID: {ex['example_id']} | ConvID: {ex['conversation_id']})")
        print(f"Brand: @{ex.get('brand', 'AmazonHelp')}")
        print(f"Conversation Context: {ex.get('conversation_context', 'N/A')}")
        print(f"\nCUSTOMER MESSAGE:")
        print(f"  \"{ex['customer_message']}\"")
        print("-" * 75)

        print("AVAILABLE INTENTS:")
        for i_idx, item in enumerate(intents, 1):
            print(f"  {i_idx:2d}. {item.name:<28} | {item.description}")
        print("-" * 75)

        rec_intent = ex.get("model_recommended_intent", "out_of_scope_other")
        rec_decision = ex.get("model_recommended_decision", "ESCALATE")

        print(f"MODEL RECOMMENDATION (Optional Reference):")
        print(f"  Suggested Intent   : {rec_intent}")
        print(f"  Suggested Decision : {rec_decision}")
        print("-" * 75)

        # 1. Human Intent Choice
        intent_choice = input(f"Enter Intent Number (1-9) or press ENTER to accept recommendation [{rec_intent}]: ").strip()
        if intent_choice.isdigit() and 1 <= int(intent_choice) <= len(intent_names):
            chosen_intent = intent_names[int(intent_choice) - 1]
        else:
            chosen_intent = rec_intent
        print(f"  --> HUMAN INTENT LABEL: {chosen_intent}")

        # 2. Human Decision Choice
        dec_choice = input(f"Enter Expected Action [A] AUTO_HANDLE / [E] ESCALATE (Press ENTER for [{rec_decision}]): ").strip().upper()
        if dec_choice == "A":
            chosen_decision = "AUTO_HANDLE"
        elif dec_choice == "E":
            chosen_decision = "ESCALATE"
        else:
            chosen_decision = rec_decision
        print(f"  --> HUMAN DECISION LABEL: {chosen_decision}")

        # 3. Escalation Reason if Escalated
        escalation_reason = None
        if chosen_decision == "ESCALATE":
            esc_r = input("Enter Escalation Reason (or ENTER for default): ").strip()
            escalation_reason = esc_r if esc_r else "Query requires human agent intervention or contains ambiguity."

        # 4. Optional Annotation Note
        note = input("Optional Annotation Note (press ENTER to skip): ").strip()
        
        # Save Human Gold Labels
        ex["gold_intent"] = chosen_intent
        ex["gold_decision"] = chosen_decision
        ex["escalation_reason"] = escalation_reason
        if note:
            ex["annotation_note"] = note
        ex["annotated_by_human"] = True  # MARKED TRUE BY HUMAN INPUT

        annotated_in_session += 1
        
        # Save after every individual example for absolute safety
        save_jsonl(examples, golden_path)
        pd.DataFrame(examples).to_csv(Path("evaluation/golden_set.csv"), index=False)

    print("\n" + "="*75)
    print(f"BATCH COMPLETE: Successfully recorded {annotated_in_session} genuine human annotations!")
    print("="*75)
    display_progress_summary(examples)

if __name__ == "__main__":
    main()
