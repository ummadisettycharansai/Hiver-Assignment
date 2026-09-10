from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from hiver_agent.utils.logging import get_logger

logger = get_logger("data.loader")

REQUIRED_COLUMNS = [
    "tweet_id", "author_id", "inbound", "created_at",
    "text", "response_tweet_id", "in_response_to_tweet_id"
]

def load_raw_dataset(filepath: str | Path, sample_n: Optional[int] = None) -> pd.DataFrame:
    """
    Load raw TWCS dataset CSV file with basic validation.
    
    Args:
        filepath: Path to twcs.csv
        sample_n: Optional number of rows to load for fast inspection/subsampling
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {path}")

    logger.info(f"Loading raw dataset from {path} (sample_n={sample_n})...")
    
    df = pd.read_csv(
        path,
        nrows=sample_n,
        dtype={
            "tweet_id": "str",
            "author_id": "str",
            "inbound": "bool",
            "created_at": "str",
            "text": "str",
            "response_tweet_id": "str",
            "in_response_to_tweet_id": "str"
        },
        low_memory=False
    )
    
    # Column check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    logger.info(f"Successfully loaded {len(df):,} rows.")
    return df

def inspect_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive dataset quality and structural metrics."""
    total_rows = len(df)
    missing_text = df["text"].isna().sum()
    duplicate_tweets = df["tweet_id"].duplicated().sum()
    
    inbound_count = int(df["inbound"].sum())
    outbound_count = total_rows - inbound_count
    
    # Unique authors
    unique_authors = df["author_id"].nunique()
    
    # Brands are non-numeric author_ids (handles)
    # Customers are numeric author_ids
    brand_authors = df[~df["author_id"].str.replace('.', '', regex=False).str.isdigit()]["author_id"].unique().tolist()
    customer_count = df[df["author_id"].str.replace('.', '', regex=False).str.isdigit()]["author_id"].nunique()
    
    summary = {
        "total_rows": total_rows,
        "missing_text_rows": int(missing_text),
        "duplicate_tweet_ids": int(duplicate_tweets),
        "inbound_customer_tweets": inbound_count,
        "outbound_brand_tweets": outbound_count,
        "unique_authors_total": unique_authors,
        "unique_brands_count": len(brand_authors),
        "unique_customers_count": customer_count,
        "top_brands": df[~df["inbound"]]["author_id"].value_counts().head(10).to_dict()
    }
    return summary
