import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hiver_agent.config import get_config
from hiver_agent.evaluation.evaluator import SystemEvaluator
from hiver_agent.utils.io import load_jsonl, save_json, save_jsonl
from hiver_agent.utils.logging import get_logger

logger = get_logger("evaluate")

def main():
    parser = argparse.ArgumentParser(description="Run complete system evaluation harness on Golden Evaluation Set.")
    parser.add_argument("--golden-set", type=str, default="evaluation/golden_set.jsonl", help="Path to golden_set.jsonl")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM Judge evaluation for ultra-fast run")
    args = parser.parse_args()

    config = get_config()
    golden_path = Path(args.golden_set)
    
    logger.info(f"Loading golden set from {golden_path}...")
    golden_examples = load_jsonl(golden_path)
    
    evaluator = SystemEvaluator(config=config)
    eval_summary = evaluator.evaluate_golden_set(golden_examples, run_judge=not args.skip_judge)
    
    # Save outputs
    res_dir = Path("results")
    res_dir.mkdir(parents=True, exist_ok=True)
    
    save_json(eval_summary, res_dir / "evaluation_summary.json")
    
    pred_dir = res_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(eval_summary["predictions"], pred_dir / "agent_predictions.jsonl")

    # Export decision results table
    dec_m = eval_summary["decision_metrics"]
    df_dec = pd.DataFrame([{
        "Metric": "Total Golden Examples", "Value": dec_m.get("total_examples")
    }, {
        "Metric": "Auto-Handled Count (%)", "Value": f"{dec_m.get('auto_handled_count')} ({dec_m.get('auto_handled_pct')}%)"
    }, {
        "Metric": "Escalated Count (%)", "Value": f"{dec_m.get('escalated_count')} ({dec_m.get('escalated_pct')}%)"
    }, {
        "Metric": "Decision Accuracy", "Value": dec_m.get("decision_accuracy")
    }, {
        "Metric": "Safe Auto-Handle Rate", "Value": dec_m.get("safe_auto_handle_rate")
    }, {
        "Metric": "Unsafe Auto-Handle Rate (False Positive)", "Value": dec_m.get("unsafe_auto_handle_rate")
    }, {
        "Metric": "False Escalation Rate", "Value": dec_m.get("false_escalation_rate")
    }])
    
    tbl_dir = res_dir / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    df_dec.to_csv(tbl_dir / "decision_results.csv", index=False)

    # Export human vs LLM judge agreement table
    agr_m = eval_summary.get("human_llm_agreement", {})
    if agr_m:
        df_agr = pd.DataFrame([{
            "Sample Size": agr_m.get("sample_size"),
            "Exact Agreement %": agr_m.get("exact_agreement_pct"),
            "Within-1 Agreement %": agr_m.get("within_one_agreement_pct"),
            "Spearman Correlation": agr_m.get("spearman_correlation"),
            "Quadratic Cohen's Kappa": agr_m.get("cohens_kappa")
        }])
        df_agr.to_csv(tbl_dir / "judge_agreement.csv", index=False)

    print("\n" + "="*70)
    print("               SYSTEM AGENT EVALUATION HARNESS REPORT")
    print("="*70)
    print(df_dec.to_string(index=False))
    if agr_m:
        print("-" * 70)
        print("HUMAN VS LLM JUDGE AGREEMENT METRICS:")
        print(f"  Exact Agreement      : {agr_m.get('exact_agreement_pct')}%")
        print(f"  Within-1 Agreement   : {agr_m.get('within_one_agreement_pct')}%")
        print(f"  Spearman Correlation : {agr_m.get('spearman_correlation')}")
        print(f"  Cohen's Kappa        : {agr_m.get('cohens_kappa')}")
    if eval_summary.get("judge_rubric_averages"):
        print("-" * 70)
        print("LLM JUDGE RUBRIC AVERAGE SCORES (1-5 Scale):")
        for k, v in eval_summary["judge_rubric_averages"].items():
            print(f"  - {k:<20}: {v}/5.0")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
