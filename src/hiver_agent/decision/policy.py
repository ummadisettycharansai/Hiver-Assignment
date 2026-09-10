from typing import Dict, Any, List, Optional
from hiver_agent.utils.logging import get_logger

logger = get_logger("decision.policy")

class DecisionPolicy:
    """
    Defensible Escalation Policy combining classifier confidence, retrieval similarity,
    evidence count, response validation, and high-risk intent detection.
    """
    
    def __init__(
        self,
        min_intent_confidence: float = 0.65,
        min_retrieval_similarity: float = 0.45
    ):
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_similarity = min_retrieval_similarity

    def evaluate_decision(
        self,
        intent: str,
        intent_confidence: float,
        retrieved_evidence: List[Dict[str, Any]],
        generation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate customer message signals to decide AUTO_HANDLE vs ESCALATE.
        """
        reasons = []
        should_escalate = False
        
        # 1. Intent confidence check
        if intent_confidence < self.min_intent_confidence:
            should_escalate = True
            reasons.append(f"Low intent confidence ({intent_confidence:.2f} < threshold {self.min_intent_confidence:.2f}).")

        # 2. Out of scope / noisy message check
        if intent == "out_of_scope_other":
            should_escalate = True
            reasons.append("Predicted intent is out-of-scope or noisy/ambiguous message.")

        # 3. High risk / customer service complaint check
        if intent == "customer_service_complaint":
            should_escalate = True
            reasons.append("High-risk category (customer service agent complaint / escalation request).")

        # 4. Retrieval similarity check
        top_retrieval_score = retrieved_evidence[0].get("similarity_score", 0.0) if retrieved_evidence else 0.0
        if not retrieved_evidence or top_retrieval_score < self.min_retrieval_similarity:
            should_escalate = True
            reasons.append(f"Insufficient historical evidence similarity ({top_retrieval_score:.2f} < threshold {self.min_retrieval_similarity:.2f}).")

        # 5. Response validation check
        if not generation_result.get("validation_passed", True):
            should_escalate = True
            val_reasons = generation_result.get("validation_reasons", [])
            reasons.append(f"Generated reply failed safety validation: {'; '.join(val_reasons)}")

        # Formulate explicit decision
        if should_escalate:
            decision = "ESCALATE"
            reason_str = " | ".join(reasons)
        else:
            decision = "AUTO_HANDLE"
            reason_str = f"Passed all safety thresholds: Intent '{intent}' (conf: {intent_confidence:.2f}), retrieval score ({top_retrieval_score:.2f}), validated evidence."

        return {
            "decision": decision,
            "reason": reason_str,
            "signals": {
                "intent": intent,
                "intent_confidence": intent_confidence,
                "top_retrieval_score": top_retrieval_score,
                "evidence_count": len(retrieved_evidence),
                "validation_passed": generation_result.get("validation_passed", True)
            }
        }
