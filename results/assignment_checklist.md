# Final Assignment Deliverables Checklist

| Requirement | Status | Verification & Evidence |
|---|---|---|
| 1. Runnable pipeline | PASS | `python run_pipeline.py --config configs/default.yaml` completes end-to-end |
| 2. Under-15-minute reproduction | PASS | Subsample pipeline completes in **~58 seconds** (<1 minute) |
| 3. One selected brand | PASS | `@AmazonHelp` selected via quantitative ranking in `results/brand_selection_report.json` |
| 4. Data-derived intents | PASS | 9 custom intents in `configs/intents.yaml` derived from `@AmazonHelp` customer queries |
| 5. Historical retrieval | PASS | SentenceTransformer + FAISS index in `data/processed/retrieval_index` |
| 6. Grounded replies | PASS | Evidence-grounded generation in `src/hiver_agent/generation/generator.py` |
| 7. Auto-handle vs Escalation | PASS | Defensible policy in `src/hiver_agent/decision/policy.py` |
| 8. Escalation reasons | PASS | Every escalation returns human-explainable reason string |
| 9. 150–250 golden examples | PASS | 200 examples in `evaluation/golden_set.csv` & `golden_set.jsonl` |
| 10. Sampling methodology | PASS | Documented in `evaluation/annotation_guidelines.md` |
| 11. Human labelling methodology | PARTIAL | Candidate labels rule-assisted from test split; manual annotation CLI script [`scripts/annotate_golden_set.py`](file:///c:/Users/Ummadisetty%20Charan/OneDrive/Desktop/Hiver%20assignment/scripts/annotate_golden_set.py) provided for human review pass |
| 12. Trivial baseline | PASS | Majority-class classifier evaluated (54.5% Accuracy) |
| 13. Simple baseline | PASS | TF-IDF + Logistic Regression evaluated (**89.0% Accuracy, 0.8251 F1**) |
| 14. Proposed system | PASS | Dense Embedding Classifier + FAISS Retriever + LLM Generator |
| 15. Automated metrics | PASS | Accuracy, Macro F1, Weighted F1, Decision Metrics in `results/tables/` |
| 16. LLM judge | PASS | Rubric evaluation in `src/hiver_agent/evaluation/judge.py` with mock and real provider support |
| 17. Human-vs-LLM agreement | PARTIAL | Within-1 score agreement framework implemented; awaiting candidate human annotation pass via CLI tool |
| 18. Five failure modes | PASS | Top 5 empirical failure modes with real examples in `results/failure_analysis.md` |
| 19. Real examples for failures | PASS | Real Twitter customer queries included for all 5 failure categories |
| 20. Misleading headline number | PASS | Mandatory critique section in `README.md` and `final_report.md` |
| 21. One-week roadmap | PASS | 5 prioritized improvements in `final_report.md` & `README.md` |
| 22. 10–15 decisions | PASS | 15 non-obvious engineering decisions in `DECISIONS.md` |
| 23. Data leakage audit | PASS | Verified 0 thread overlap in `results/leakage_audit.md` |
| 24. Tests | PASS | 10 / 10 unit tests passing via `pytest tests/` |
| 25. Citations | PASS | Dataset, libraries, models cited in `README.md` |
| 26. No secrets | PASS | `.gitignore` and `.env.example` verified; zero committed secrets |
| 27. Reproducibility | PASS | Fixed seed 42 set globally; reproducible across machines |
| 28. No full dataset in git | PASS | `data/raw/*.csv` gitignored |

**OVERALL ASSIGNMENT AUDIT: 26 PASS / 2 PARTIAL (Awaiting candidate manual annotation pass).**
