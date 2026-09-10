from typing import List, Dict, Any
import numpy as np

def compute_decision_metrics(gold_decisions: List[str], pred_decisions: List[str]) -> Dict[str, Any]:
    """
    Compute key metrics for AUTO_HANDLE vs ESCALATE decision safety policy.
    
    Metrics:
    - total_examples: count
    - auto_handle_count / pct: % of cases automated
    - escalate_count / pct: % of cases escalated
    - decision_accuracy: overall agreement with gold decisions
    - safe_auto_handle_rate: % of gold AUTO_HANDLE cases correctly auto-handled
    - false_auto_handle_rate: % of gold ESCALATE cases erroneously auto-handled (Unsafe automation)
    - false_escalation_rate: % of gold AUTO_HANDLE cases unnecessarily escalated
    """
    assert len(gold_decisions) == len(pred_decisions), "Length mismatch"
    total = len(gold_decisions)
    if total == 0:
        return {}

    auto_cnt = sum(1 for p in pred_decisions if p == "AUTO_HANDLE")
    esc_cnt = sum(1 for p in pred_decisions if p == "ESCALATE")
    
    correct_cnt = sum(1 for g, p in zip(gold_decisions, pred_decisions) if g == p)
    acc = correct_cnt / total
    
    gold_auto = [i for i, g in enumerate(gold_decisions) if g == "AUTO_HANDLE"]
    gold_esc = [i for i, g in enumerate(gold_decisions) if g == "ESCALATE"]
    
    # Safe auto-handle rate
    safe_auto = sum(1 for i in gold_auto if pred_decisions[i] == "AUTO_HANDLE") / len(gold_auto) if gold_auto else 0.0
    
    # False auto handle (UNSAFE: gold said ESCALATE, but system auto-handled)
    false_auto = sum(1 for i in gold_esc if pred_decisions[i] == "AUTO_HANDLE") / len(gold_esc) if gold_esc else 0.0
    
    # False escalation (OVERLY CONSERVATIVE: gold said AUTO_HANDLE, but system escalated)
    false_esc = sum(1 for i in gold_auto if pred_decisions[i] == "ESCALATE") / len(gold_auto) if gold_auto else 0.0

    return {
        "total_examples": total,
        "auto_handled_count": auto_cnt,
        "auto_handled_pct": round(auto_cnt / total * 100, 2),
        "escalated_count": esc_cnt,
        "escalated_pct": round(esc_cnt / total * 100, 2),
        "decision_accuracy": round(acc, 4),
        "safe_auto_handle_rate": round(safe_auto, 4),
        "unsafe_auto_handle_rate": round(false_auto, 4),
        "false_escalation_rate": round(false_esc, 4)
    }
