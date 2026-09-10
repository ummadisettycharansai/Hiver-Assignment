import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from hiver_agent.utils.logging import get_logger

logger = get_logger("retrieval.index")

class VectorStoreIndex:
    """FAISS vector store for fast similarity search over historical support conversations."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.encoder = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension: int = 384

    def _get_encoder(self):
        if self.encoder is None:
            logger.info(f"Loading retriever embedding model: {self.embedding_model}")
            self.encoder = SentenceTransformer(self.embedding_model)
        return self.encoder

    def build_index(self, conversations: List[Dict[str, Any]]):
        """Index historical customer support conversations."""
        logger.info(f"Building retrieval index for {len(conversations):,} historical conversations...")
        
        texts = []
        self.metadata = []
        
        for conv in conversations:
            if not conv.get("has_brand_response"):
                continue
            cust_text = conv.get("first_customer_message", "").strip()
            brand_reply = conv.get("brand_responses", [""])[0] if conv.get("brand_responses") else ""
            
            if not cust_text or not brand_reply:
                continue
                
            texts.append(cust_text)
            self.metadata.append({
                "conversation_id": conv["conversation_id"],
                "root_tweet_id": conv.get("root_tweet_id"),
                "customer_id": conv.get("customer_id"),
                "customer_message": cust_text,
                "brand_reply": brand_reply,
                "created_at": conv.get("created_at")
            })

        if not texts:
            raise ValueError("No valid customer conversations available for indexing.")

        encoder = self._get_encoder()
        logger.info(f"Encoding {len(texts):,} customer query vectors...")
        embeddings = encoder.encode(texts, show_progress_bar=False, batch_size=64, normalize_embeddings=True)
        
        embeddings_np = np.array(embeddings, dtype=np.float32)
        self.dimension = embeddings_np.shape[1]
        
        # Inner Product on normalized vectors = Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_np)
        
        logger.info(f"FAISS index built successfully with {self.index.ntotal:,} vectors (Dim={self.dimension}).")

    def search(
        self,
        query_text: str,
        top_k: int = 3,
        exclude_conv_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search top-k most similar historical cases."""
        if self.index is None or not self.metadata:
            raise RuntimeError("Retrieval index is empty. Build or load index first.")
            
        encoder = self._get_encoder()
        query_vec = encoder.encode([query_text], show_progress_bar=False, normalize_embeddings=True)
        query_np = np.array(query_vec, dtype=np.float32)
        
        # Retrieve extra if excluding
        fetch_k = top_k + 5 if exclude_conv_id else top_k
        distances, indices = self.index.search(query_np, fetch_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = dict(self.metadata[idx])
            if exclude_conv_id and meta["conversation_id"] == exclude_conv_id:
                continue
            meta["similarity_score"] = round(float(dist), 4)
            results.append(meta)
            if len(results) >= top_k:
                break
                
        return results

    def save(self, directory: str | Path):
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(dir_path / "faiss_index.bin"))
        with open(dir_path / "index_metadata.pkl", "wb") as f:
            pickle.dump({
                "embedding_model": self.embedding_model,
                "dimension": self.dimension,
                "metadata": self.metadata
            }, f)
        logger.info(f"Index and metadata saved to {dir_path}")

    def load(self, directory: str | Path):
        dir_path = Path(directory)
        self.index = faiss.read_index(str(dir_path / "faiss_index.bin"))
        with open(dir_path / "index_metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.embedding_model = data["embedding_model"]
            self.dimension = data["dimension"]
            self.metadata = data["metadata"]
        logger.info(f"Loaded FAISS index with {self.index.ntotal:,} vectors from {dir_path}")
