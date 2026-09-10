import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hiver_agent.config import get_config
from hiver_agent.intents.classifier import MajorityBaselineClassifier, TFIDFBaselineClassifier, DenseEmbeddingClassifier
from hiver_agent.intents.discovery import get_default_intent_taxonomy
from create_golden_set import assign_rule_guided_gold_intent
from hiver_agent.utils.io import load_jsonl, save_json
from hiver_agent.utils.logging import get_logger

logger = get_logger("run_baselines")

def prepare_labeled_dataset(convs: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    texts = []
    labels = []
    for c in convs:
        txt = c.get("first_customer_message", "").strip()
        if not txt:
            continue
        intent, _, _ = assign_rule_guided_gold_intent(txt)
        texts.append(txt)
        labels.append(intent)
    return texts, labels

def evaluate_model(model_name: str, y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    
    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": {k: v for k, v in rep.items() if k in labels}
    }

def plot_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str], title: str, output_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Predicted Intent", fontsize=12)
    plt.ylabel("True Gold Intent", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate baseline and proposed intent classifiers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = get_config()
    proc_dir = Path(config.processed_data_dir)
    
    logger.info("Loading train, val, test, and golden datasets...")
    train_convs = load_jsonl(proc_dir / "train_conversations.jsonl")
    val_convs = load_jsonl(proc_dir / "val_conversations.jsonl")
    test_convs = load_jsonl(proc_dir / "test_conversations.jsonl")
    golden_convs = load_jsonl(Path("evaluation/golden_set.jsonl"))

    X_train, y_train = prepare_labeled_dataset(train_convs)
    X_val, y_val = prepare_labeled_dataset(val_convs)
    X_test, y_test = prepare_labeled_dataset(test_convs)
    
    X_gold = [g["customer_message"] for g in golden_convs]
    y_gold = [g["gold_intent"] for g in golden_convs]
    
    taxonomy = get_default_intent_taxonomy()
    all_labels = taxonomy.get_intent_names()

    logger.info(f"Dataset Sizes: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}, Golden={len(X_gold):,}")

    # 1. Majority Baseline
    maj_clf = MajorityBaselineClassifier()
    maj_clf.fit(X_train, y_train)
    maj_preds_test = [maj_clf.predict_one(t)["intent"] for t in X_test]
    maj_preds_gold = [maj_clf.predict_one(t)["intent"] for t in X_gold]
    
    maj_eval_test = evaluate_model("Majority Baseline (Test)", y_test, maj_preds_test, all_labels)
    maj_eval_gold = evaluate_model("Majority Baseline (Golden)", y_gold, maj_preds_gold, all_labels)

    # 2. TF-IDF + Logistic Regression
    tfidf_clf = TFIDFBaselineClassifier()
    tfidf_clf.fit(X_train, y_train)
    tfidf_preds_test = [p["intent"] for p in tfidf_clf.predict_batch(X_test)]
    tfidf_preds_gold = [p["intent"] for p in tfidf_clf.predict_batch(X_gold)]
    
    tfidf_eval_test = evaluate_model("TF-IDF Baseline (Test)", y_test, tfidf_preds_test, all_labels)
    tfidf_eval_gold = evaluate_model("TF-IDF Baseline (Golden)", y_gold, tfidf_preds_gold, all_labels)
    
    tfidf_clf.save("results/models/tfidf_baseline.pkl")

    # 3. Dense Embedding Proposed Classifier
    dense_clf = DenseEmbeddingClassifier(model_name=config.embedding_model)
    dense_clf.fit(X_train, y_train)
    dense_preds_test = [p["intent"] for p in dense_clf.predict_batch(X_test)]
    dense_preds_gold = [p["intent"] for p in dense_clf.predict_batch(X_gold)]
    
    dense_eval_test = evaluate_model("Dense Embedding Proposed (Test)", y_test, dense_preds_test, all_labels)
    dense_eval_gold = evaluate_model("Dense Embedding Proposed (Golden)", y_gold, dense_preds_gold, all_labels)
    
    dense_clf.save("results/models/dense_proposed.pkl")

    # Save confusion matrix figures
    fig_dir = Path("results/figures")
    plot_confusion_matrix(y_gold, tfidf_preds_gold, all_labels, "Confusion Matrix - TF-IDF Baseline (Golden Set)", fig_dir / "confusion_matrix_tfidf.png")
    plot_confusion_matrix(y_gold, dense_preds_gold, all_labels, "Confusion Matrix - Dense Embedding Proposed (Golden Set)", fig_dir / "confusion_matrix_proposed.png")

    # Summary table
    table_records = [
        {
            "model": "1. Majority Baseline",
            "eval_set": "Golden Set (200)",
            "accuracy": maj_eval_gold["accuracy"],
            "macro_f1": maj_eval_gold["macro_f1"],
            "weighted_f1": maj_eval_gold["weighted_f1"],
            "macro_precision": maj_eval_gold["macro_precision"],
            "macro_recall": maj_eval_gold["macro_recall"]
        },
        {
            "model": "2. TF-IDF + LogReg",
            "eval_set": "Golden Set (200)",
            "accuracy": tfidf_eval_gold["accuracy"],
            "macro_f1": tfidf_eval_gold["macro_f1"],
            "weighted_f1": tfidf_eval_gold["weighted_f1"],
            "macro_precision": tfidf_eval_gold["macro_precision"],
            "macro_recall": tfidf_eval_gold["macro_recall"]
        },
        {
            "model": "3. SentenceEmbedding + LogReg",
            "eval_set": "Golden Set (200)",
            "accuracy": dense_eval_gold["accuracy"],
            "macro_f1": dense_eval_gold["macro_f1"],
            "weighted_f1": dense_eval_gold["weighted_f1"],
            "macro_precision": dense_eval_gold["macro_precision"],
            "macro_recall": dense_eval_gold["macro_recall"]
        }
    ]
    
    df_results = pd.DataFrame(table_records)
    table_dir = Path("results/tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(table_dir / "classification_results.csv", index=False)

    full_results = {
        "majority_test": maj_eval_test,
        "majority_gold": maj_eval_gold,
        "tfidf_test": tfidf_eval_test,
        "tfidf_gold": tfidf_eval_gold,
        "dense_test": dense_eval_test,
        "dense_gold": dense_eval_gold
    }
    save_json(full_results, Path("results/classification_results.json"))

    print("\n" + "="*75)
    print("               INTENT CLASSIFICATION EVALUATION RESULTS")
    print("="*75)
    print(df_results.to_string(index=False))
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
