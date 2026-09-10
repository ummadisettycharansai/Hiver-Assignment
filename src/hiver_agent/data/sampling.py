import random
from typing import List, Dict, Any, Tuple
from hiver_agent.utils.logging import get_logger

logger = get_logger("data.sampling")

def split_conversations_thread_level(
    conversations: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deterministically split conversations into Train/Val/Test by conversation_id.
    This guarantees zero leakage of tweets between splits from the same thread.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    # Shuffle conversation IDs deterministically
    convs_copy = list(conversations)
    rng = random.Random(seed)
    rng.shuffle(convs_copy)
    
    n_total = len(convs_copy)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_convs = convs_copy[:n_train]
    val_convs = convs_copy[n_train:n_train + n_val]
    test_convs = convs_copy[n_train + n_val:]
    
    logger.info(
        f"Thread-level data split (Seed={seed}): "
        f"Total={n_total:,} | Train={len(train_convs):,} | Val={len(val_convs):,} | Test={len(test_convs):,}"
    )
    return train_convs, val_convs, test_convs

def create_reproducible_subsample(
    conversations: List[Dict[str, Any]],
    max_count: int = 2000,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Sample a compact, reproducible subset of conversations for fast pipeline execution."""
    rng = random.Random(seed)
    convs_copy = list(conversations)
    rng.shuffle(convs_copy)
    return convs_copy[:max_count]
