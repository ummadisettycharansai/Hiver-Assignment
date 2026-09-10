import json
from typing import Dict, Any, List, Optional
from hiver_agent.generation.llm_provider import LLMProvider
from hiver_agent.generation.prompts import REPLY_GENERATION_PROMPT_TEMPLATE, format_historical_evidence
from hiver_agent.generation.validation import validate_generated_reply
from hiver_agent.utils.logging import get_logger

logger = get_logger("generation.generator")

class GroundedReplyGenerator:
    """Generates grounded customer support replies using historical retrieval evidence."""
    
    def __init__(self, provider: Optional[LLMProvider] = None, brand_name: str = "AmazonHelp"):
        self.provider = provider or LLMProvider()
        self.brand_name = brand_name

    def generate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieved_evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate response grounded in evidence."""
        
        evidence_str = format_historical_evidence(retrieved_evidence)
        prompt = REPLY_GENERATION_PROMPT_TEMPLATE.format(
            brand_name=self.brand_name,
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            historical_evidence_formatted=evidence_str
        )
        
        raw_output = self.provider.generate(prompt, temperature=0.0)
        
        try:
            parsed = json.loads(raw_output)
            reply_text = parsed.get("reply", "")
            evidence_ids = parsed.get("evidence_ids", [])
            confidence = parsed.get("confidence", 0.8)
        except Exception as e:
            logger.warning(f"Failed to parse LLM JSON output: {e}. Raw: {raw_output[:100]}")
            reply_text = raw_output
            evidence_ids = [item.get("conversation_id") for item in retrieved_evidence if "conversation_id" in item]
            confidence = 0.7

        # Validate response
        val_result = validate_generated_reply(reply_text, retrieved_evidence, customer_message)
        
        return {
            "reply": reply_text,
            "evidence_ids": evidence_ids,
            "confidence": confidence,
            "validation_passed": val_result["passed"],
            "validation_reasons": val_result["reasons"]
        }
