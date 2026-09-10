import argparse
import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hiver_agent.evaluation.human_agreement import evaluate_human_vs_llm_agreement
from hiver_agent.utils.io import load_jsonl, save_jsonl

def display_agreement_report(annotations: List[Dict[str, Any]]):
    human_scores = [a["human_quality_score"] for a in annotations if "human_quality_score" in a]
    llm_scores = [a["llm_judge_score"] for a in annotations if "llm_judge_score" in a]

    if not human_scores:
        print("\nNo human reply quality annotations found yet.")
        return

    metrics = evaluate_human_vs_llm_agreement(human_scores, llm_scores)
    
    print("\n" + "="*70)
    print("      REAL HUMAN VS LLM JUDGE AGREEMENT METRICS")
    print("="*70)
    print(f"Human Sample Size        : {metrics.get('sample_size')}")
    print(f"Exact Score Agreement %  : {metrics.get('exact_agreement_pct')}%")
    print(f"Within-1 Score Agreement%: {metrics.get('within_one_agreement_pct')}%")
    print(f"Spearman Rank Correlation: {metrics.get('spearman_correlation')}")
    print(f"Quadratic Cohen's Kappa  : {metrics.get('cohens_kappa')}")
    print("="*70 + "\n")
    
    # Save to tables
    df_agr = pd.DataFrame([metrics])
    df_agr.to_csv(Path("results/tables/judge_agreement.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description="Interactive Human Reply-Quality Annotation Tool.")
    parser.add_argument("--predictions", type=str, default="results/predictions/agent_predictions.jsonl", help="Path to predictions")
    parser.add_argument("--count", type=int, default=10, help="Number of replies to grade in this session")
    parser.add_argument("--report-only", action="store_true", help="Display agreement report and exit")
    args = parser.parse_args()

    ann_path = Path("evaluation/human_reply_annotations.jsonl")
    
    existing_anns = []
    if ann_path.exists():
        existing_anns = load_jsonl(ann_path)

    if args.report_only:
        display_agreement_report(existing_anns)
        return

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"Error: Predictions file not found at {pred_path}. Run python scripts/evaluate.py first.")
        return

    predictions = load_jsonl(pred_path)
    annotated_ids = set(a["example_id"] for a in existing_anns)
    
    unannotated = [p for p in predictions if p.get("example_id") not in annotated_ids]

    if not unannotated:
        print("\nAll predictions have been annotated by human!")
        display_agreement_report(existing_anns)
        return

    batch = unannotated[:args.count]

    print("\n" + "="*75)
    print("      HUMAN REPLY-QUALITY ANNOTATION & LLM JUDGE CALIBRATION TOOL")
    print("="*75)
    print(f"Total Predictions Available : {len(predictions)}")
    print(f"Already Human Annotated     : {len(existing_anns)}")
    print(f"Annotating Batch of         : {len(batch)}")
    print("="*75 + "\n")

    for step_num, pred in enumerate(batch, 1):
        print("\n" + "-"*75)
        print(f"Reply #{step_num} / {len(batch)} (Example ID: {pred.get('example_id')})")
        print(f"Customer Message: \"{pred.get('customer_message')}\"")
        print(f"Predicted Intent: {pred.get('pred_intent')} (Conf: {pred.get('intent_confidence'):.2f})")
        print(f"\nGENERATED REPLY:\n  \"{pred.get('generated_reply')}\"")
        print(f"\nSYSTEM DECISION: {pred.get('pred_decision')} | Reason: {pred.get('decision_reason')}")
        print("-" * 75)

        # Prompt 1-5 Quality Score
        while True:
            score_input = input("Enter Quality Score (1=Unacceptable, 2=Poor, 3=Acceptable, 4=Good, 5=Excellent): ").strip()
            if score_input in ["1", "2", "3", "4", "5"]:
                human_score = int(score_input)
                break
            print("Invalid input! Enter a number between 1 and 5.")

        # Defect flags
        print("Select Flags (comma separated, or press ENTER if none):")
        print("  1 = incorrect, 2 = unsupported claim / hallucination, 3 = irrelevant, 4 = inappropriate escalation")
        flags_input = input("Flags: ").strip()
        
        flags = []
        if "1" in flags_input: flags.append("incorrect")
        if "2" in flags_input: flags.append("unsupported_claim")
        if "3" in flags_input: flags.append("irrelevant")
        if "4" in flags_input: flags.append("inappropriate_escalation")

        llm_score = pred.get("judge_metrics", {}).get("overall_score", 4)
        
        record = {
            "example_id": pred.get("example_id"),
            "customer_message": pred.get("customer_message"),
            "generated_reply": pred.get("generated_reply"),
            "human_quality_score": human_score,
            "llm_judge_score": llm_score,
            "defect_flags": flags,
            "annotated_by_human": True
        }
        existing_anns.append(record)
        
        save_jsonl(existing_anns, ann_path)
        pd.DataFrame(existing_anns).to_csv(Path("evaluation/human_reply_annotations.csv"), index=False)
        print(f"Saved Annotation (Human: {human_score} vs LLM Judge: {llm_score})")

    print("\n" + "="*75)
    print("SESSION COMPLETE!")
    display_agreement_report(existing_anns)

if __name__ == "__main__":
    main()
