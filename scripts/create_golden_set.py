import argparse
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hiver_agent.config import get_config
from hiver_agent.utils.io import load_jsonl, save_jsonl
from hiver_agent.utils.logging import get_logger

logger = get_logger("create_golden_set")

def assign_candidate_recommendations(text: str) -> Tuple[str, str, str]:
    """
    Model/rule recommendation helper for candidate test messages.
    Returns (recommended_intent, recommended_decision, difficulty).
    """
    t = text.lower()
    
    # Customer service complaint
    if any(k in t for k in ["rude", "hung up", "worst service", "supervisor", "manager", "disappointed", "terrible support", "useless"]):
        return "customer_service_complaint", "ESCALATE", "hard"
        
    # Refund / Return
    if any(k in t for k in ["refund", "return label", "money back", "reimbursement", "sent back"]):
        if "late" in t or "delay" in t:
            return "refund_return_status", "AUTO_HANDLE", "medium"
        return "refund_return_status", "AUTO_HANDLE", "easy"

    # Damaged / Defective
    if any(k in t for k in ["damaged", "broken", "shattered", "expired", "wrong item", "defective", "torn"]):
        return "damaged_defective_item", "AUTO_HANDLE", "medium"

    # Account / Prime
    if any(k in t for k in ["prime", "subscription", "membership", "account locked", "otp", "login"]):
        return "account_login_prime", "AUTO_HANDLE", "easy"

    # Payment / Billing
    if any(k in t for k in ["charged", "billing", "card debited", "gift card", "double charge", "invoice"]):
        return "payment_billing_issue", "AUTO_HANDLE", "medium"

    # Digital services
    if any(k in t for k in ["prime video", "kindle", "alexa", "firestick", "error code", "streaming"]):
        return "digital_services", "AUTO_HANDLE", "easy"

    # General inquiry
    if any(k in t for k in ["stock", "price match", "when will", "available", "pre order", "warranty"]):
        return "general_inquiry_availability", "AUTO_HANDLE", "easy"

    # Out of scope / Short / Greeting
    if len(t.split()) <= 3 or any(k in t for k in ["hello", "dm sent", "hi @", "???", "thanks"]):
        return "out_of_scope_other", "ESCALATE", "easy"

    # Order delivery delay (default shipping query)
    if any(k in t for k in ["delivery", "delivered", "package", "tracking", "order", "shipped", "carrier", "arriving"]):
        return "order_delivery_delay", "AUTO_HANDLE", "easy"

    return "out_of_scope_other", "ESCALATE", "hard"

def main():
    parser = argparse.ArgumentParser(description="Create candidate golden evaluation set from test split.")
    parser.add_argument("--count", type=int, default=200, help="Number of candidate examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = get_config()
    test_path = Path(config.processed_data_dir) / "test_conversations.jsonl"
    if not test_path.exists():
        test_path = Path(config.subsample_dir) / "test_conversations.jsonl"

    logger.info(f"Loading test set from: {test_path}")
    test_convs = load_jsonl(test_path)
    
    golden_examples = []
    
    for idx, conv in enumerate(test_convs[:args.count]):
        cust_msg = conv.get("first_customer_message", "").strip()
        if not cust_msg:
            continue
            
        rec_intent, rec_decision, difficulty = assign_candidate_recommendations(cust_msg)
        
        ex = {
            "example_id": f"gold_{idx+1:03d}",
            "conversation_id": conv["conversation_id"],
            "brand": config.brand_name,
            "customer_message": cust_msg,
            "conversation_context": f"Customer ID: {conv.get('customer_id')} | Turn Count: {conv.get('turn_count')}",
            "model_recommended_intent": rec_intent,
            "model_recommended_decision": rec_decision,
            "gold_intent": rec_intent,
            "gold_decision": rec_decision,
            "escalation_reason": "Low intent confidence or out-of-scope query." if rec_decision == "ESCALATE" else None,
            "difficulty": difficulty,
            "annotation_note": "Initial candidate example.",
            "annotated_by_human": False  # MUST BE FALSE UNTIL HUMAN ANNOTATES
        }
        golden_examples.append(ex)

    out_dir = Path("evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_jsonl(golden_examples, out_dir / "golden_set.jsonl")
    
    df_gold = pd.DataFrame(golden_examples)
    df_gold.to_csv(out_dir / "golden_set.csv", index=False)
    
    print("\n" + "="*70)
    print("             GOLDEN EVALUATION SET CANDIDATES CREATED")
    print("="*70)
    print(f"Total Examples Created : {len(golden_examples)}")
    print(f"Annotated by Human     : 0 / {len(golden_examples)} (False)")
    print(f"Saved Files            : evaluation/golden_set.jsonl & golden_set.csv")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
