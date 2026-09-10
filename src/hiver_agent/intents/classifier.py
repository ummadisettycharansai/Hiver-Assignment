import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

from hiver_agent.intents.taxonomy import IntentTaxonomy, load_intent_taxonomy
from hiver_agent.utils.logging import get_logger

logger = get_logger("intents.classifier")

class MajorityBaselineClassifier:
    """Baseline 1: Always predicts the most frequent class from training data."""
    def __init__(self):
        self.majority_class: str = "order_delivery_delay"

    def fit(self, texts: List[str], labels: List[str]):
        if labels:
            label_counts = pd.Series(labels).value_counts()
            self.majority_class = label_counts.index[0]
            logger.info(f"Majority class identified: '{self.majority_class}' ({label_counts.iloc[0]} occurrences)")

    def predict_one(self, text: str) -> Dict[str, Any]:
        return {
            "intent": self.majority_class,
            "confidence": 1.0,
            "alternatives": []
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict_one(t) for t in texts]


class TFIDFBaselineClassifier:
    """Baseline 2: TF-IDF + Logistic Regression with probability confidence."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        self.clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
        self.classes_: List[str] = []

    def fit(self, texts: List[str], labels: List[str]):
        logger.info(f"Training TF-IDF + Logistic Regression baseline on {len(texts):,} examples...")
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self.classes_ = list(self.clf.classes_)
        logger.info(f"TF-IDF model trained on {len(self.classes_)} distinct intent classes.")

    def predict_one(self, text: str) -> Dict[str, Any]:
        X = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X)[0]
        top_indices = np.argsort(probs)[::-1]
        
        top_intent = self.classes_[top_indices[0]]
        top_conf = float(probs[top_indices[0]])
        
        alternatives = [
            {"intent": self.classes_[idx], "confidence": round(float(probs[idx]), 4)}
            for idx in top_indices[1:3]
        ]
        return {
            "intent": top_intent,
            "confidence": round(top_conf, 4),
            "alternatives": alternatives
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict_one(t) for t in texts]

    def save(self, filepath: str | Path):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "clf": self.clf, "classes": self.classes_}, f)

    def load(self, filepath: str | Path):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.clf = data["clf"]
            self.classes_ = data["classes"]


class DenseEmbeddingClassifier:
    """Proposed System Intent Classifier: SentenceTransformer Embeddings + Softmax Logistic Classifier."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self.clf = LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced", random_state=42)
        self.classes_: List[str] = []

    def _get_encoder(self):
        if self.encoder is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
        return self.encoder

    def fit(self, texts: List[str], labels: List[str]):
        logger.info(f"Encoding {len(texts):,} texts for Dense Embedding Classifier training...")
        encoder = self._get_encoder()
        embeddings = encoder.encode(texts, show_progress_bar=False, batch_size=64)
        
        self.clf.fit(embeddings, labels)
        self.classes_ = list(self.clf.classes_)
        logger.info(f"Dense Embedding Classifier trained successfully.")

    def predict_one(self, text: str) -> Dict[str, Any]:
        encoder = self._get_encoder()
        emb = encoder.encode([text], show_progress_bar=False)
        probs = self.clf.predict_proba(emb)[0]
        top_indices = np.argsort(probs)[::-1]
        
        top_intent = self.classes_[top_indices[0]]
        top_conf = float(probs[top_indices[0]])
        
        alternatives = [
            {"intent": self.classes_[idx], "confidence": round(float(probs[idx]), 4)}
            for idx in top_indices[1:3]
        ]
        return {
            "intent": top_intent,
            "confidence": round(top_conf, 4),
            "alternatives": alternatives
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        encoder = self._get_encoder()
        embeddings = encoder.encode(texts, show_progress_bar=False, batch_size=64)
        probs_matrix = self.clf.predict_proba(embeddings)
        
        results = []
        for probs in probs_matrix:
            top_indices = np.argsort(probs)[::-1]
            top_intent = self.classes_[top_indices[0]]
            top_conf = float(probs[top_indices[0]])
            alternatives = [
                {"intent": self.classes_[idx], "confidence": round(float(probs[idx]), 4)}
                for idx in top_indices[1:3]
            ]
            results.append({
                "intent": top_intent,
                "confidence": round(top_conf, 4),
                "alternatives": alternatives
            })
        return results

    def save(self, filepath: str | Path):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model_name": self.model_name, "clf": self.clf, "classes": self.classes_}, f)

    def load(self, filepath: str | Path):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model_name = data["model_name"]
            self.clf = data["clf"]
            self.classes_ = data["classes"]
