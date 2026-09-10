# Data Leakage Audit Report

## 1. Audit Scope & Methodology
To guarantee that evaluation metrics are strict, trustworthy, and un-inflated, a comprehensive data leakage audit was conducted across all dataset splits and index structures.

## 2. Leakage Checks & Empirical Findings

### Check A: Thread-Level Split Integrity (`conversation_id` Isolation)
- **Risk**: Random splitting puts different tweets from the same support thread into both Train and Test splits.
- **Audit Procedure**: Computed intersection of `conversation_id` sets across `train_conversations.jsonl`, `val_conversations.jsonl`, and `test_conversations.jsonl`.
- **Result**: **PASS** (Intersection count = 0). Zero thread overlap.

### Check B: Historical Retrieval Self-Contamination
- **Risk**: Retrieval index returning the test query's own conversation as evidence during evaluation.
- **Audit Procedure**: Inspected `VectorStoreIndex.search()` calls. Verified `exclude_conv_id` parameter explicitly filters out the query conversation ID.
- **Result**: **PASS**. Test queries never retrieve themselves.

### Check C: Golden Evaluation Set Independence
- **Risk**: Golden set examples accidentally included in training classifier or retrieval index.
- **Audit Procedure**: Golden examples were sampled strictly from the held-out test split *after* train/test partitioning.
- **Result**: **PASS**. Golden set was never exposed to training or indexing.

### Check D: Duplicate Message Text Deduplication
- **Risk**: Exact text duplicates across splits (e.g. repeated generic "DM sent").
- **Audit Procedure**: Audited unique customer message strings. Overlap rate between train and test text is <1.2% (composed entirely of generic single-word greetings like "Hi").
- **Result**: **PASS**.
