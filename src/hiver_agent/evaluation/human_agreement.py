import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, accuracy_score
from hiver_agent.utils.logging import get_logger

logger = get_logger("evaluation.human_agreement")

def evaluate_human_vs_llm_agreement(
    human_scores: List[int],
    llm_scores: List[int]
) -> Dict[str, Any]:
    """
    Compute inter-rater agreement statistics between Human judgements and LLM Judge scores.
    """
    assert len(human_scores) == len(llm_scores), "Length mismatch between human and LLM scores"
    n = len(human_scores)
    if n == 0:
        return {}

    # Exact agreement
    exact_match = sum(1 for h, l in zip(human_scores, llm_scores) if h == l)
    exact_pct = round((exact_match / n) * 100, 2)
    
    # Within-1 agreement (e.g. human=4, llm=5 -> within 1)
    within_one = sum(1 for h, l in zip(human_scores, llm_scores) if abs(h - l) <= 1)
    within_one_pct = round((within_one / n) * 100, 2)
    
    # Spearman rank correlation
    rho, p_val = spearmanr(human_scores, llm_scores)
    rho_val = round(float(rho), 4) if not np.isnan(rho) else 0.0
    
    # Quadratic weighted Cohen's Kappa
    try:
        kappa = cohen_kappa_score(human_scores, llm_scores, weights="quadratic")
        kappa_val = round(float(kappa), 4)
    except Exception:
        kappa_val = 0.0

    logger.info(
        f"Human vs LLM Judge Agreement (N={n}): "
        f"Exact={exact_pct}% | Within-1={within_one_pct}% | Spearman Rho={rho_val} | Quadratic Kappa={kappa_val}"
    )

    return {
        "sample_size": n,
        "exact_agreement_pct": exact_pct,
        "within_one_agreement_pct": within_one_pct,
        "spearman_correlation": rho_val,
        "spearman_p_value": round(float(p_val), 5) if not np.isnan(p_val) else 1.0,
        "cohens_kappa": kappa_val
    }
