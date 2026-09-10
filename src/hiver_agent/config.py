import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class PipelineConfig(BaseModel):
    brand_name: str = Field(default="AppleSupport", description="Selected brand handle for customer support")
    raw_data_path: str = Field(default="data/raw/twcs/twcs.csv")
    processed_data_dir: str = Field(default="data/processed")
    subsample_dir: str = Field(default="data/subsample")
    random_seed: int = Field(default=42)
    
    # Subsample settings for 15-min reproduction requirement
    sample_size: int = Field(default=5000, description="Subsample row count for reproducible fast run")
    
    # LLM Settings
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock"))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    
    # Embeddings & Retrieval
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    retrieval_top_k: int = Field(default=3)
    
    # Escalation Policy Thresholds
    min_intent_confidence: float = Field(default=0.65)
    min_retrieval_similarity: float = Field(default=0.45)
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PipelineConfig":
        path = Path(yaml_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()

def get_config(yaml_path: Optional[str] = None) -> PipelineConfig:
    if yaml_path and Path(yaml_path).exists():
        return PipelineConfig.from_yaml(yaml_path)
    default_config_path = Path("configs/default.yaml")
    if default_config_path.exists():
        return PipelineConfig.from_yaml(default_config_path)
    return PipelineConfig()
