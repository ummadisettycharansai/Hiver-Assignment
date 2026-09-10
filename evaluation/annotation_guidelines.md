# Golden Evaluation Set Annotation Guidelines

## 1. Objective
The Golden Evaluation Set provides a benchmark of 200 real-world customer support messages sampled from `@AmazonHelp` threads in the TWCS test split (`data/processed/test_conversations.jsonl`). It serves as the primary evaluation set for intent classification accuracy, historical resolution retrieval quality, and auto-handling vs escalation safety policy decisions.

## 2. Label Source & Annotation Methodology
> [!IMPORTANT]
> **Annotation Transparency**: Candidate gold intent and decision labels in `golden_set.csv` / `golden_set.jsonl` were generated via deterministic rule-guided heuristic extraction on real held-out test queries from `@AmazonHelp`. Complete manual human-in-the-loop re-annotation is identified as a remaining production onboarding step.

## 3. Sampling Strategy
To ensure the golden set covers realistic customer inquiries and edge cases:
- **Coverage**: 200 real customer queries were sampled from the held-out test split (`data/processed/test_conversations.jsonl`).
- **Distribution**:
  - **Standard Inquiries**: ~45% delivery, account, refund, and payment inquiries.
  - **Noisy / Foreign / Out-of-Scope**: ~54% foreign language tweets (Spanish, Portuguese, German, Japanese), incomplete mentions ("DM sent"), or generic rants.
  - **Escalation Triggers**: Queries requesting supervisory escalation or expressing extreme agent frustration.
- **Independence**: The golden evaluation set was strictly reserved for evaluation and NEVER used during baseline training or FAISS index construction.

## 4. Taxonomy & Label Definitions

| Gold Intent Name | Description | Example Query |
|---|---|---|
| `order_delivery_delay` | Tracking, missing package, shipment delays | "Where is my package? Says delivered but not here." |
| `refund_return_status` | Returns, refund status, reimbursement | "When will my refund be credited for order 123?" |
| `damaged_defective_item` | Broken, damaged, wrong product received | "The glass jar arrived shattered inside the box." |
| `account_login_prime` | Prime membership, account locking, OTP | "Charged $14.99 for Prime after cancellation." |
| `payment_billing_issue` | Double charge, gift card issue, payment error | "Payment failed but money was debited from bank." |
| `digital_services` | Prime Video, Kindle, Fire TV, Alexa streaming | "Prime Video app showing error code 5004." |
| `customer_service_complaint` | Agent critique, rude rep, escalation request | "Your support rep hung up on me!" |
| `general_inquiry_availability` | Stock availability, pre-order, price match | "When will PS5 be back in stock?" |
| `out_of_scope_other` | Foreign language, greetings, spam, emojis | "Hello @AmazonHelp", "DM sent", "Parabéns..." |

## 5. Decision Label Rules
Each example is labeled with `gold_decision` (`AUTO_HANDLE` or `ESCALATE`):
- **`AUTO_HANDLE`**: Customer issue is clear, written in English, belongs to a standard intent, and can be resolved using standard historical resolution evidence without requesting sensitive credentials.
- **`ESCALATE`**: Customer query is ambiguous, written in a non-English language, expresses high frustration with agent support, requests out-of-policy refund exceptions, or lacks sufficient context.

## 6. Handling Ambiguous Examples
- If a message contains multiple intents (e.g. "My broken item arrived late, give me a refund"), assign the **primary driver** of the customer's request.
- Non-English tweets (Spanish, Portuguese, German, Japanese) present in the TWCS dataset are assigned `out_of_scope_other` and flagged for `ESCALATE` to prevent un-grounded English reply generation.
