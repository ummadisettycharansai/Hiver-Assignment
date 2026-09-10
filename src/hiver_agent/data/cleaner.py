import re
import pandas as pd
from hiver_agent.utils.logging import get_logger

logger = get_logger("data.cleaner")

def clean_tweet_text(text: str) -> str:
    """
    Clean raw tweet text:
    - Remove extra whitespaces
    - Standardize mentions (keep @handles intact as they identify brands/users)
    - Preserve punctuation relevant for sentiment/intent (? and !)
    """
    if not isinstance(text, str):
        return ""
    
    # Replace multiple spaces/newlines with a single space
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned

def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data cleaning pipeline on raw dataframe:
    - Drop missing text or missing tweet_id
    - Drop duplicates
    - Strip text
    """
    initial_len = len(df)
    
    # Drop rows without text or tweet_id
    df_clean = df.dropna(subset=["tweet_id", "text"]).copy()
    
    # Drop exact duplicate tweet_ids
    df_clean = df_clean.drop_duplicates(subset=["tweet_id"])
    
    # Clean text column
    df_clean["text"] = df_clean["text"].apply(clean_tweet_text)
    
    # Filter out empty text
    df_clean = df_clean[df_clean["text"].str.len() > 0]
    
    final_len = len(df_clean)
    logger.info(f"Cleaned dataframe: {initial_len:,} -> {final_len:,} rows ({initial_len - final_len:,} dropped).")
    return df_clean
