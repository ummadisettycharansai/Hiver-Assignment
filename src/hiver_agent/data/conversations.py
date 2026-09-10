from typing import List, Dict, Any, Optional
import pandas as pd
from hiver_agent.utils.logging import get_logger

logger = get_logger("data.conversations")

def reconstruct_conversations(
    df: pd.DataFrame,
    target_brand: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Reconstruct multi-turn support conversations from tweet reply linkages.
    
    Args:
        df: Cleaned TWCS dataframe.
        target_brand: If specified, filter conversations where brand matches target_brand.
        
    Returns:
        List of structured conversation objects.
    """
    logger.info("Building tweet lookup tables for conversation reconstruction...")
    
    # Build dictionary by tweet_id
    tweet_dict = {}
    for idx, row in df.iterrows():
        t_id = str(row["tweet_id"]).strip()
        tweet_dict[t_id] = row
        
    # Map child responses: parent_tweet_id -> list of child_tweet_ids
    children_map: Dict[str, List[str]] = {}
    
    for t_id, row in tweet_dict.items():
        # Check in_response_to_tweet_id
        parent_id = str(row["in_response_to_tweet_id"]).strip() if pd.notna(row["in_response_to_tweet_id"]) else None
        if parent_id and parent_id != "nan":
            children_map.setdefault(parent_id, []).append(t_id)
            
        # Check response_tweet_id (may be comma separated)
        resp_id_str = str(row["response_tweet_id"]).strip() if pd.notna(row["response_tweet_id"]) else None
        if resp_id_str and resp_id_str != "nan":
            for resp_id in resp_id_str.split(","):
                resp_id = resp_id.strip()
                if resp_id:
                    if resp_id in tweet_dict:
                        if resp_id not in children_map.setdefault(t_id, []):
                            children_map.setdefault(t_id, []).append(resp_id)

    # Identify candidate root tweets: inbound customer tweets with no parent in tweet_dict
    root_tweet_ids = []
    for t_id, row in tweet_dict.items():
        if row["inbound"]:  # Customer message
            parent_id = str(row["in_response_to_tweet_id"]).strip() if pd.notna(row["in_response_to_tweet_id"]) else None
            if not parent_id or parent_id == "nan" or parent_id not in tweet_dict:
                root_tweet_ids.append(t_id)
                
    logger.info(f"Identified {len(root_tweet_ids):,} root customer messages out of {len(df):,} total tweets.")
    
    conversations = []
    visited_tweets = set()
    
    for root_id in root_tweet_ids:
        if root_id in visited_tweets:
            continue
            
        root_row = tweet_dict[root_id]
        
        # Traverse thread using BFS/DFS
        thread_tweet_ids = []
        queue = [root_id]
        
        while queue:
            curr_id = queue.pop(0)
            if curr_id in visited_tweets or curr_id not in tweet_dict:
                continue
            visited_tweets.add(curr_id)
            thread_tweet_ids.append(curr_id)
            
            # Add children
            children = children_map.get(curr_id, [])
            for child in children:
                if child not in visited_tweets:
                    queue.append(child)

        # Build thread messages ordered by timestamp / sequence
        messages = []
        brand_participating = set()
        
        for t_id in thread_tweet_ids:
            t_row = tweet_dict[t_id]
            is_inbound = bool(t_row["inbound"])
            speaker = "customer" if is_inbound else "brand"
            author = str(t_row["author_id"])
            
            if not is_inbound:
                brand_participating.add(author)
                
            messages.append({
                "message_id": t_id,
                "speaker": speaker,
                "author_id": author,
                "text": str(t_row["text"]),
                "created_at": str(t_row["created_at"])
            })

        if not messages:
            continue
            
        primary_brand = list(brand_participating)[0] if brand_participating else "Unknown"
        
        # If target brand specified, check filter
        if target_brand and primary_brand.lower() != target_brand.lower():
            continue
            
        customer_msgs = [m for m in messages if m["speaker"] == "customer"]
        brand_msgs = [m for m in messages if m["speaker"] == "brand"]
        
        conv_obj = {
            "conversation_id": f"conv_{root_id}",
            "root_tweet_id": root_id,
            "brand": primary_brand,
            "customer_id": messages[0]["author_id"],
            "created_at": messages[0]["created_at"],
            "turn_count": len(messages),
            "customer_message_count": len(customer_msgs),
            "brand_message_count": len(brand_msgs),
            "has_brand_response": len(brand_msgs) > 0,
            "first_customer_message": customer_msgs[0]["text"] if customer_msgs else "",
            "brand_responses": [m["text"] for m in brand_msgs],
            "messages": messages
        }
        
        conversations.append(conv_obj)
        
    logger.info(f"Reconstructed {len(conversations):,} valid conversations.")
    return conversations
