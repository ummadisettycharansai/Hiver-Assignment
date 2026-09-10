from typing import List, Dict, Any
import pandas as pd
from hiver_agent.utils.logging import get_logger

logger = get_logger("data.analysis")

def analyze_brand_statistics(conversations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze conversation metrics across all brands to rank and select the best brand.
    
    Metrics evaluated:
    - total_conversations: Number of conversations involving the brand
    - resolved_conversations: Conversations with at least one brand reply
    - resolution_rate: % of customer queries replied to by brand
    - avg_turns: Average messages per conversation
    - avg_cust_msg_length: Avg character length of first customer message
    - avg_brand_reply_length: Avg character length of brand reply
    """
    brand_stats = {}
    
    for conv in conversations:
        brand = conv["brand"]
        if brand == "Unknown":
            continue
            
        if brand not in brand_stats:
            brand_stats[brand] = {
                "total_conversations": 0,
                "resolved_conversations": 0,
                "total_turns": 0,
                "cust_text_len_sum": 0,
                "brand_text_len_sum": 0,
                "brand_reply_count": 0
            }
            
        stats = brand_stats[brand]
        stats["total_conversations"] += 1
        if conv["has_brand_response"]:
            stats["resolved_conversations"] += 1
            
        stats["total_turns"] += conv["turn_count"]
        stats["cust_text_len_sum"] += len(conv["first_customer_message"])
        
        for reply in conv["brand_responses"]:
            stats["brand_text_len_sum"] += len(reply)
            stats["brand_reply_count"] += 1

    records = []
    for brand, s in brand_stats.items():
        tot = s["total_conversations"]
        res = s["resolved_conversations"]
        res_rate = (res / tot * 100.0) if tot > 0 else 0.0
        avg_turns = (s["total_turns"] / tot) if tot > 0 else 0.0
        avg_cust_len = (s["cust_text_len_sum"] / tot) if tot > 0 else 0.0
        avg_reply_len = (s["brand_text_len_sum"] / s["brand_reply_count"]) if s["brand_reply_count"] > 0 else 0.0
        
        # Composite score favoring high volume, high resolution rate, and sufficient response length
        score = (res * 0.4) + (res_rate * 2.0) + (avg_reply_len * 0.2)
        
        records.append({
            "brand": brand,
            "total_conversations": tot,
            "resolved_conversations": res,
            "resolution_rate_pct": round(res_rate, 2),
            "avg_turns": round(avg_turns, 2),
            "avg_customer_msg_len": round(avg_cust_len, 1),
            "avg_brand_reply_len": round(avg_reply_len, 1),
            "brand_score": round(score, 2)
        })

    df_stats = pd.DataFrame(records)
    if not df_stats.empty:
        df_stats = df_stats.sort_values(by="resolved_conversations", ascending=False).reset_index(drop=True)
    return df_stats
