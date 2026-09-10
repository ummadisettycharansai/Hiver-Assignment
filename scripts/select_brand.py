import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hiver_agent.config import get_config
from hiver_agent.data.loader import load_raw_dataset
from hiver_agent.data.cleaner import clean_raw_dataframe
from hiver_agent.data.conversations import reconstruct_conversations
from hiver_agent.data.analysis import analyze_brand_statistics
from hiver_agent.utils.logging import get_logger
from hiver_agent.utils.io import save_json

logger = get_logger("select_brand")

def main():
    parser = argparse.ArgumentParser(description="Analyze brands and select the optimal target brand.")
    parser.add_argument("--sample-n", type=int, default=100000, help="Number of rows to sample for brand analysis")
    args = parser.parse_args()

    config = get_config()
    logger.info(f"Loading data sample of {args.sample_n:,} rows...")
    df_raw = load_raw_dataset(config.raw_data_path, sample_n=args.sample_n)
    df_clean = clean_raw_dataframe(df_raw)
    
    logger.info("Reconstructing conversations across all brands...")
    conversations = reconstruct_conversations(df_clean)
    
    logger.info("Computing brand ranking statistics...")
    df_stats = analyze_brand_statistics(conversations)
    
    table_dir = Path("results/tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(table_dir / "brand_ranking.csv", index=False)
    
    top_brand_row = df_stats.iloc[0]
    selected_brand = top_brand_row["brand"]
    
    report = {
        "selected_brand": selected_brand,
        "selection_rationale": (
            f"@{selected_brand} ranked #1 with {top_brand_row['total_conversations']:,} total conversations, "
            f"{top_brand_row['resolved_conversations']:,} resolved multi-turn threads "
            f"({top_brand_row['resolution_rate_pct']}% resolution rate), and average reply length of "
            f"{top_brand_row['avg_brand_reply_len']} characters. It provides rich, repeatable customer support "
            f"patterns suitable for intent discovery and historical resolution retrieval."
        ),
        "top_5_brands": df_stats.head(5).to_dict(orient="records")
    }
    
    save_json(report, Path("results/brand_selection_report.json"))
    
    print("\n" + "="*70)
    print("               EMPIRICAL BRAND SELECTION REPORT")
    print("="*70)
    print(f"Selected Target Brand : @{selected_brand}")
    print(f"Rationale             : {report['selection_rationale']}")
    print("\nTop 5 Candidate Brands:")
    for idx, r in df_stats.head(5).iterrows():
        print(f"  {idx+1}. @{r['brand']:<15} | Convs: {r['total_conversations']:<5} | Resolved: {r['resolved_conversations']:<5} | Res%: {r['resolution_rate_pct']:<5}% | ReplyLen: {r['avg_brand_reply_len']}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
