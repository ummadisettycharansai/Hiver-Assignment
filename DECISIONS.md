# System Architecture & Engineering Decision Log

This document records 15 non-obvious technical, architectural, and evaluation decisions made during the design and implementation of the Hiver AI Customer Support Agent.

---

### Decision 1: Brand Selection via Quantitative Multi-Metric Ranking
- **Decision**: Select `@AmazonHelp` as the target support brand.
- **Reason**: Quantitative analysis across 100k raw dataset rows proved `@AmazonHelp` had the highest volume of complete multi-turn threads (2,974 conversations), 100.0% resolution rate, and sufficient average brand reply length (130.9 characters).
- **Alternative Considered**: `@AppleSupport` or `@Uber_Support`.
- **Why Rejected**: `@AppleSupport` had fewer total threads (2,050) and narrower intent variety in Twitter interactions.
- **Evidence**: Empirical brand ranking saved in `results/tables/brand_ranking.csv` and `results/brand_selection_report.json`.

---

### Decision 2: Data Splitting at Thread/Conversation Level
- **Decision**: Enforce strict thread-level data splitting (70% Train / 15% Val / 15% Test by `conversation_id`).
- **Reason**: Prevents severe data leakage. Random tweet-level splitting causes different turns of the exact same customer support interaction to appear in both training and test sets, inflating accuracy artificially.
- **Alternative Considered**: Random row-level train/test split.
- **Why Rejected**: Causes data leakage and violates Rule 8 of assignment core principles.
- **Evidence**: `results/leakage_audit.md` verified 0 conversation ID overlaps between splits.

---

### Decision 3: Intent Taxonomy Discovered from Brand Data rather than Off-the-Shelf Benchmarks
- **Decision**: Define a custom 9-intent taxonomy specifically for `@AmazonHelp` customer interactions rather than adopting `Banking77`.
- **Reason**: Banking77 covers financial banking queries (cards, SWIFT, accounts), whereas e-commerce customer support requires specific intents like `order_delivery_delay`, `damaged_defective_item`, and `refund_return_status`.
- **Alternative Considered**: Directly mapping queries to Banking77.
- **Why Rejected**: Violated assignment directive ("Intents must be defined from the actual data").
- **Evidence**: Schema saved in `configs/intents.yaml`.

---

### Decision 4: Dense Vector Embeddings + Logistic Regression for Intent Classification
- **Decision**: Use `sentence-transformers/all-MiniLM-L6-v2` dense embeddings with a balanced Logistic Regression classifier as the proposed intent model.
- **Reason**: Lightweight (384 dimensions, fast CPU inference <10ms), captures semantic intent far better than TF-IDF without needing heavy fine-tuning or expensive LLM API calls.
- **Alternative Considered**: Pure zero-shot LLM classification for every incoming query.
- **Why Rejected**: Unacceptably high API latency, high cost, and difficulty ensuring reproducible confidence scores without logprobs.
- **Evidence**: Achieved superior macro F1 and weighted F1 over TF-IDF baseline on golden set.

---

### Decision 5: FAISS Vector Index with Exact Cosine Similarity for Historical Retrieval
- **Decision**: Use FAISS `IndexFlatIP` on L2-normalized embeddings for historical resolution retrieval.
- **Reason**: Exact Inner Product on normalized vectors provides exact Cosine similarity, fast search speed on thousands of historical cases, and local file persistence.
- **Alternative Considered**: External vector database (Pinecone, Qdrant, Milvus).
- **Why Rejected**: Unnecessary external infrastructure dependency violating self-contained local runnable pipeline principles.
- **Evidence**: `data/processed/retrieval_index/faiss_index.bin` indexes 3,406 historical AmazonHelp resolutions in <100MB.

---

### Decision 6: Exclusion of Query Conversation ID During Historical Retrieval
- **Decision**: Explicitly filter out the query's own `conversation_id` from top-K retrieved historical candidates.
- **Reason**: When evaluating on training or held-out validation sets, retrieving the exact conversation itself causes 100% trivial similarity matches and masks retrieval evaluation quality.
- **Alternative Considered**: Unfiltered top-K retrieval.
- **Why Rejected**: Introduces self-retrieval bias and trivial evidence grounding.
- **Evidence**: Implemented in `VectorStoreIndex.search(exclude_conv_id=...)`.

---

### Decision 7: Multi-Signal Defensible Escalation Policy
- **Decision**: Implement explicit multi-signal policy evaluating classifier confidence, retrieval similarity, evidence count, validation results, and high-risk intent triggers.
- **Reason**: Real support automation must prioritize safety over raw automation volume. A simple `if confidence > 0.5` rule fails to catch ungrounded or high-risk complaint cases.
- **Alternative Considered**: Single confidence threshold classifier cutoff.
- **Why Rejected**: Fails to account for evidence relevance or response safety validation failures.
- **Evidence**: Evaluated in `src/hiver_agent/decision/policy.py`.

---

### Decision 8: Grounded Generation Prompting with Explicit Citation Requirements
- **Decision**: Prompt template requires LLM to cite `evidence_ids` from retrieved cases and explicitly forbids inventing unbacked refund/compensation promises.
- **Reason**: Prevents LLM hallucinations (e.g. promising $100 refunds or free replacements) that create legal and brand liabilities.
- **Alternative Considered**: Free-form unconstrained response generation.
- **Why Rejected**: High hallucination risk.
- **Evidence**: `src/hiver_agent/generation/prompts.py`.

---

### Decision 9: Offline LLM Provider Abstraction with Mock & Disk Cache Fallbacks
- **Decision**: Abstract `LLMProvider` supporting OpenAI, Anthropic, Ollama, and a deterministic Mock fallback backed by SHA-256 disk caching.
- **Reason**: Allows reviewers to run the complete pipeline without requiring private API keys or spending money, while guaranteeing cached execution under 15 minutes.
- **Alternative Considered**: Hardcoding OpenAI API calls.
- **Why Rejected**: Violates requirements to support API key-less running and configurable providers.
- **Evidence**: `src/hiver_agent/generation/llm_provider.py`.

---

### Decision 10: 200 Hand-Verified Golden Evaluation Examples Sampled from Held-Out Test Set
- **Decision**: Construct 200 hand-verified golden set examples with explicit gold intent, gold decision, and difficulty metadata.
- **Reason**: Provides a trustworthy, un-contaminated evaluation ground truth that was never exposed to model training.
- **Alternative Considered**: Using model-generated pseudo-labels as ground truth.
- **Why Rejected**: Pseudo-labels perpetuate model biases and render evaluation meaningless.
- **Evidence**: Saved in `evaluation/golden_set.csv` and `golden_set.jsonl`.

---

### Decision 11: Quantitative LLM-as-a-Judge Rubric with Human Agreement Calibration
- **Decision**: Grade generated responses across 6 distinct rubric dimensions (Relevance, Correctness, Grounding, Helpfulness, Tone, Overall) and compute Human vs LLM Judge agreement.
- **Reason**: Single BLEU/ROUGE metrics perform poorly for conversational quality; LLM judge gives nuanced qualitative scores calibrated against human agreement (Spearman rho & Cohen's kappa).
- **Alternative Considered**: Sole reliance on ROUGE/BLEU metrics.
- **Why Rejected**: ROUGE rewards exact n-gram matching rather than semantic helpfulness and grounding.
- **Evidence**: `results/tables/judge_agreement.csv`.

---

### Decision 12: Automated Safety & Hallucination Response Validation Layer
- **Decision**: Pass generated replies through a rule-based safety validator prior to returning `AUTO_HANDLE`.
- **Reason**: Catches regex patterns for sensitive credentials (credit cards, SSNs) and ungrounded financial keywords ("100% refund", "free gift card").
- **Alternative Considered**: Relying solely on LLM self-policing.
- **Why Rejected**: LLMs occasionally ignore system prompts under complex edge cases.
- **Evidence**: `src/hiver_agent/generation/validation.py`.

---

### Decision 13: Sampled Sub-Dataset for 15-Minute Reproduction Guarantee
- **Decision**: Package a documented 5,000-row sample dataset in `data/subsample/` for fast execution.
- **Reason**: Fulfills Hiver requirement that headline results can be reproduced in under 15 minutes on a standard developer laptop without GPU.
- **Alternative Considered**: Forcing reviewers to download and index all 3 million raw tweets.
- **Why Rejected**: Processing 3M tweets takes >2 hours and requires substantial RAM.
- **Evidence**: `python run_pipeline.py --config configs/default.yaml` completes in <10 minutes.

---

### Decision 14: Structured Error Handling & Zero Silent Failures
- **Decision**: Implement explicit exception handling across dataset loading, JSON parsing, vector indexing, and API timeouts.
- **Reason**: Ensures pipeline gracefully reports errors (e.g. malformed CSV rows or API failures) rather than crashing or swallowing exceptions silently.
- **Alternative Considered**: Bare try-except blocks with silent pass.
- **Why Rejected**: Violates guidelines ("Inspect logs & stack traces, no silent symptom patches").
- **Evidence**: Explicit logging across all `src/hiver_agent` modules.

---

### Decision 15: CLI Interface and One-Command Automation over Complex Web UI
- **Decision**: Build a clean CLI demo (`python -m hiver_agent.demo`) and one-line pipeline runner (`run_pipeline.py`) rather than spending time on a Web UI.
- **Reason**: SDE/ML Intern take-home assignment explicitly states: "Prefer strong evaluation over unnecessary UI complexity."
- **Alternative Considered**: Building a React / Next.js web chat dashboard.
- **Why Rejected**: Web UI consumes engineering time without adding proof of ML rigor.
- **Evidence**: `run_pipeline.py` and `src/hiver_agent/demo.py`.
