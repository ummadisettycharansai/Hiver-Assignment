import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

class SimpleCache:
    """Disk-backed JSON cache using SHA-256 key hashing."""
    
    def __init__(self, cache_dir: str = ".cache/llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        hashed = self._hash_key(key)
        filepath = self.cache_dir / f"{hashed}.json"
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def set(self, key: str, value: Any) -> None:
        hashed = self._hash_key(key)
        filepath = self.cache_dir / f"{hashed}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)
