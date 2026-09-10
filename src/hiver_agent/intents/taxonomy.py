from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field

class IntentDefinition(BaseModel):
    name: str
    description: str
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    ambiguous_cases: List[str] = Field(default_factory=list)

class IntentTaxonomy(BaseModel):
    brand: str
    intents: List[IntentDefinition]

    def get_intent_names(self) -> List[str]:
        return [i.name for i in self.intents]

    def get_intent(self, name: str) -> Optional[IntentDefinition]:
        for i in self.intents:
            if i.name == name:
                return i
        return None

def load_intent_taxonomy(filepath: str | Path = "configs/intents.yaml") -> IntentTaxonomy:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Intent taxonomy config not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return IntentTaxonomy(**data)

def save_intent_taxonomy(taxonomy: IntentTaxonomy, filepath: str | Path = "configs/intents.yaml") -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(taxonomy.model_dump(), f, sort_keys=False, allow_unicode=True)
