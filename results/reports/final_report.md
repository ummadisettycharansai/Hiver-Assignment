# Technical Report — Hiver AI Customer Support Agent

**Author**: Candidate (SDE Intern Take-Home Submission)  
**Target Brand**: `@AmazonHelp`  
**Dataset**: Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`)  
**Date**: September 2026  

---

## 1. Executive Summary
We built an end-to-end, reproducible AI customer support agent for `@AmazonHelp` trained on reconstructed multi-turn Twitter support interactions. The system automates routine customer inquiries (`AUTO_HANDLE`) using grounded historical resolutions retrieved via FAISS vector search, while enforcing a defensible multi-signal escalation policy (`ESCALATE`) for ambiguous, un-grounded, or high-risk queries.

Key empirical findings across a 200 hand-verified Golden Evaluation Set:
- **Baseline #1 (Majority Class)**: 54.5% Accuracy | Macro F1: 0.0784
- **Baseline #2 (TF-IDF + Logistic Regression)**: 89.0% Accuracy | Macro F1: 0.8251 | Weighted F1: 0.8876
- **Proposed System (SentenceEmbedding + LogReg Classifier)**: 70.5% Accuracy | Macro F1: 0.5187 | Weighted F1: 0.7290
- **Decision Safety Policy**: **76.0% Overall Decision Accuracy**, with a **51.8% Safe Auto-Handle Rate** and only **6.09% Unsafe Automation Rate** (False Positives).

---

## 2. Problem Framing & Brand Selection
Customer support operations on social media face heavy volume, variable query quality, and strict resolution policies. 

### What "Good" Means for `@AmazonHelp`
1. **Safety Over Automation**: Never promise unauthorized refunds or free replacement items that contradict brand policy.
2. **Grounded Consistency**: Responses must mirror historical resolution patterns (e.g. directing customers to order history or verifying tracking numbers).
3. **Transparent Escalation**: Unclear queries or agent complaints must be escalated immediately to human agents with an explicit, understandable reason.

### Brand Selection Rationale
Through quantitative analysis of 100,000 raw tweets, `@AmazonHelp` was selected as the #1 target brand over 101 candidates:
- **Total Multi-Turn Conversations**: 2,974
- **Resolution Rate**: 100.0%
- **Average Reply Length**: 130.9 characters

---

## 3. Discovered Intent Taxonomy
We derived 9 distinct intent categories directly from `@AmazonHelp` customer queries:

| Intent | Description | Example Query |
|---|---|---|
| `order_delivery_delay` | Package tracking, shipment delay | "Where is my order? Delivery was expected yesterday." |
| `refund_return_status` | Return labels, refund status | "When will my refund be processed for order #123?" |
| `damaged_defective_item` | Broken, defective, or incorrect items | "Item arrived shattered inside the shipping box." |
| `account_login_prime` | Prime membership, OTP, account access | "Charged $14.99 for Prime after cancelling." |
| `payment_billing_issue` | Double charges, gift card redemption | "My card was charged twice for order #987." |
| `digital_services` | Prime Video, Kindle, Alexa glitches | "Prime Video giving error code 5004." |
| `customer_service_complaint` | Agent critique, rude rep, escalation | "Your support chat agent disconnected me!" |
| `general_inquiry_availability` | Stock availability, pre-orders | "When will PS5 be back in stock?" |
| `out_of_scope_other` | Greetings, spam, uninterpretable | "Hello @AmazonHelp", "DM sent" |

---

## 4. System Architecture

```
[ Incoming Customer Tweet ]
          │
          ▼
[ Intent Classifier (Dense Embedding / TF-IDF) ] ──► (Predicted Intent & Confidence)
          │
          ▼
[ FAISS Vector Store Retrieval Index ] ───────────► (Top-K Historical Resolutions)
          │
          ▼
[ Grounded LLM Reply Generator & Validator ] ──────► (Draft Reply + Validation Check)
          │
          ▼
[ Defensible Escalation Policy ]
          │
     ┌────┴──────────────────────────┐
     ▼                               ▼
[ AUTO_HANDLE ]               [ ESCALATE ]
(Safe, Grounded Reply)      (Human Agent + Explicit Reason)
```

---

## 5. Experimental Evaluation & Results

### Intent Classification Performance (Golden Set N=200)

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
|---|---|---|---|---|---|
| **1. Majority Baseline** | 54.5% | 0.0784 | 0.3845 | 0.0606 | 0.1111 |
| **2. TF-IDF + Logistic Regression** | **89.0%** | **0.8251** | **0.8876** | **0.9140** | **0.7770** |
| **3. SentenceEmbedding + LogReg** | 70.5% | 0.5187 | 0.7290 | 0.4856 | 0.5921 |

### Escalation Safety Policy Performance

| Decision Metric | Result | Interpretation |
|---|---|---|
| Total Golden Examples | 200 | Hand-verified evaluation set |
| Auto-Handled Count (%) | 51 (25.5%) | Safely automated queries |
| Escalated Count (%) | 149 (74.5%) | Queries routed to human agents |
| **Decision Accuracy** | **76.0%** | Overall alignment with gold decisions |
| **Safe Auto-Handle Rate** | **51.76%** | Correctly automated valid queries |
| **Unsafe Auto-Handle Rate** | **6.09%** | **Low false-automation risk** |
| False Escalation Rate | 48.24% | Conservative safety threshold bias |

---

## 6. LLM-as-a-Judge & Human Agreement

### LLM Judge Rubric Scores (1-5 Scale)
- **Relevance**: 5.0 / 5.0
- **Correctness**: 4.0 / 5.0
- **Grounding**: 5.0 / 5.0
- **Helpfulness**: 4.0 / 5.0
- **Tone**: 5.0 / 5.0
- **Overall Score**: **4.0 / 5.0**

### Human vs LLM Judge Inter-Rater Agreement
- **Within-1 Score Agreement**: **76.0%**

---

## 7. What is Misleading About My Headline Number?

> [!WARNING]
> **Mandatory Critique Section**: Our headline numbers (e.g. 89.0% TF-IDF Classification Accuracy and 76.0% Escalation Policy Decision Accuracy) present an overly optimistic picture of real-world readiness if interpreted without nuance.

1. **TF-IDF Keyword Overfitting vs Semantic Generalization**: TF-IDF achieved 89.0% accuracy primarily because Twitter customer messages heavily rely on explicit keywords ("refund", "damaged", "tracking"). However, TF-IDF fails completely on paraphrased or novel queries lacking exact vocabulary terms.
2. **Dominance of `out_of_scope_other` Class**: In the raw test set, generic greetings ("DM sent", "hello @AmazonHelp") make up >50% of incoming messages. Predicting `out_of_scope_other` inflates overall accuracy while hiding lower recall on minority classes like `damaged_defective_item`.
3. **Conservative Escalation Bias**: The 76.0% decision accuracy score is driven by a high escalation rate (74.5%). While this keeps the **Unsafe Auto-Handle Rate low (6.09%)**, it results in a high **False Escalation Rate (48.24%)**, meaning half of automatable queries are unnecessarily sent to human agents.

---

## 8. Top 5 Failure Modes
1. **Multi-Intent Customer Messages** (11.5%): Customer complains about both delivery delay and broken item in one tweet.
2. **Keyword Noise & Abbreviations** (8.0%): Short tweets like "DM sent check fast PLZ!" trigger misclassification.
3. **Out-of-Distribution Policy Exceptions** (6.5%): Queries about gift exchanges without sender notification.
4. **False Escalation on Easy Queries** (5.0%): Informational queries about app features failing retrieval thresholds.
5. **Generic Historical Reply Templates** (4.0%): Retrieval index returning generic "Please DM us" responses instead of direct instructions.

---

## 9. One-Week Development Roadmap
1. **Multi-Label Intent Classifier** (Impact: HIGH): Transition from single-label to multi-label intent detection.
2. **Direct Instruction FAQ Indexing** (Impact: HIGH): Index official Amazon support documentation alongside Twitter threads to eliminate "DM us" retrieval artifacts.
3. **Probability Calibration (Platt Scaling)** (Impact: MEDIUM): Calibrate classifier confidence scores to improve escalation precision.
4. **Active Learning Data Pipeline** (Impact: MEDIUM): Automatically flag low-confidence predictions for human re-annotation.
5. **Real-time Latency & Cost Monitoring Dashboard** (Impact: MEDIUM): Track LLM cache hit rates and API latency.
