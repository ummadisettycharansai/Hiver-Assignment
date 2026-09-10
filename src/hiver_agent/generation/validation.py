import re
from typing import Dict, Any, List
from hiver_agent.utils.logging import get_logger

logger = get_logger("generation.validation")

UNSUPPORTED_PROMISE_KEYWORDS = [
    "full refund", "100% refund", "free gift card", "free replacement",
    "compensation of", "$50 credit", "$100 voucher", "guarantee"
]

SENSITIVE_INFO_PATTERNS = [
    r"\b\d{16}\b",  # Credit card
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
    r"password", "pin number", "cvv"
]

def validate_generated_reply(
    reply_text: str,
    retrieved_evidence: List[Dict[str, Any]],
    customer_message: str
) -> Dict[str, Any]:
    """
    Validate LLM-generated draft reply against safety, grounding, and policy rules.
    Returns dictionary with validation result (passed: bool, reasons: List[str]).
    """
    reasons = []
    passed = True
    
    if not reply_text or len(reply_text.strip()) < 10:
        return {"passed": False, "reasons": ["Reply is empty or excessively short."]}

    reply_lower = reply_text.lower()
    
    # 1. Check unsupported financial promises / hallucinations
    has_evidence_refund = any("refund" in item.get("brand_reply", "").lower() for item in retrieved_evidence)
    for kw in UNSUPPORTED_PROMISE_KEYWORDS:
        if kw in reply_lower and not has_evidence_refund:
            passed = False
            reasons.append(f"Contains unsupported promise keyword '{kw}' not backed by evidence.")

    # 2. Check sensitive info requests or leakages
    for pattern in SENSITIVE_INFO_PATTERNS:
        if re.search(pattern, reply_lower):
            passed = False
            reasons.append(f"Flagged sensitive data pattern match ({pattern}).")

    # 3. Check for abrupt length or refusal
    if "i cannot help" in reply_lower or "i don't know" in reply_lower:
        passed = False
        reasons.append("Reply expresses explicit inability to assist.")

    return {
        "passed": passed,
        "reasons": reasons
    }
