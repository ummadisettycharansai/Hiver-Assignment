# Top 5 Failure Modes & Empirical Analysis

Based on systematic evaluation of the Hiver AI Support Agent across 200 hand-verified golden examples and held-out test splits, we identified the top 5 empirical failure modes.

---

### Failure Mode 1: Multi-Intent Customer Messages (11.5% Frequency)
- **Real Example**: *"My package arrived late yesterday and when I opened it the jar was broken, I need a refund now!"*
- **Expected Behavior**: Intent identified as `damaged_defective_item` (primary physical damage) or escalated for multi-claim handling.
- **Actual Behavior**: Classifier predicted `order_delivery_delay` with 0.58 confidence based on keyword matching "late yesterday".
- **Root Cause Hypothesis**: Single-label classification assumption fails on compound customer tweets containing multiple distinct grievances (delivery delay + physical damage + refund demand).
- **Proposed Fix**: Implement multi-label intent classification with sigmoid probability heads or extract sub-intent spans.

---

### Failure Mode 2: Extreme Keyword Noise & Conversational Abbreviations (8.0% Frequency)
- **Real Example**: *"DM sent @AmazonHelp check fast PLZ!!!"*
- **Expected Behavior**: Intent `out_of_scope_other` and `ESCALATE` (no actionable problem stated in tweet body).
- **Actual Behavior**: Predicted `customer_service_complaint` with 0.49 confidence.
- **Root Cause Hypothesis**: TF-IDF and dense embeddings associate exclamation marks and urgency words ("PLZ", "fast") with customer agent complaints.
- **Proposed Fix**: Pre-filter ultra-short tweets lacking noun/verb problem clauses directly into `out_of_scope_other`.

---

### Failure Mode 3: Out-of-Distribution/Novel Policy Exceptions (6.5% Frequency)
- **Real Example**: *"Can I exchange a gift I received without the original sender knowing?"*
- **Expected Behavior**: `ESCALATE` (Requires specialized privacy & gift return policy verification).
- **Actual Behavior**: Retrieved generic return steps ("Return item via Your Orders tab") and attempted `AUTO_HANDLE`.
- **Root Cause Hypothesis**: Vector index retrieval found standard return policy examples with high cosine similarity (0.61) due to terms like "exchange" and "gift", missing the subtle privacy constraint ("without sender knowing").
- **Proposed Fix**: Add semantic policy constraint checkers to response validation layer to flag sensitive policy edge cases.

---

### Failure Mode 4: False Escalation on Easy Informational Queries (5.0% Frequency)
- **Real Example**: *"Are Prime Video downloads available for offline viewing on Mac OS?"*
- **Expected Behavior**: `AUTO_HANDLE` with intent `digital_services` (Standard clear technical query).
- **Actual Behavior**: `ESCALATE` with reason "Retrieval similarity (0.41) below min threshold (0.45)".
- **Root Cause Hypothesis**: Training set lacked exact phrasing for "Mac OS offline viewing", resulting in slightly lower top-1 retrieval similarity despite high classifier confidence (0.84).
- **Proposed Fix**: Expand historical retrieval index with official FAQ documentation in addition to raw Twitter support threads.

---

### Failure Mode 5: Misaligned Historical Reply Phrasing (4.0% Frequency)
- **Real Example**: *"Where do I find my invoice PDF for order #112-9843729?"*
- **Expected Behavior**: Provide step-by-step instructions ("Go to Your Orders -> View Invoice").
- **Actual Behavior**: Retrieved historical reply saying *"Please send us a DM so we can look into this order for you."*
- **Root Cause Hypothesis**: Many historical Twitter brand responses default to generic "Send us a DM" templates, which the generator adopted as evidence instead of direct instructional guidance.
- **Proposed Fix**: Filter out generic "DM us" templated replies from the retrieval index, indexing only direct informative resolution replies.
