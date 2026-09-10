import argparse
import json
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hiver_agent.config import get_config
from hiver_agent.data.loader import load_raw_dataset, inspect_dataset_summary
from hiver_agent.utils.logging import get_logger
from hiver_agent.utils.io import save_json

logger = get_logger("inspect_dataset")

def main():
    parser = argparse.ArgumentParser(description="Inspect TWCS raw dataset schema, counts, quality.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to twcs.csv")
    parser.add_argument("--sample-n", type=int, default=None, help="Sample N rows to inspect quickly")
    args = parser.parse_args()

    config = get_config()
    data_path = args.data_path or config.raw_data_path
    
    logger.info(f"Inspecting raw dataset from: {data_path}")
    df = load_raw_dataset(data_path, sample_n=args.sample_n)
    
    summary = inspect_dataset_summary(df)
    
    print("\n" + "="*60)
    print("           TWCS DATASET INSPECTION REPORT")
    print("="*60)
    print(f"Total Rows Loaded           : {summary['total_rows']:,}")
    print(f"Missing Text Rows           : {summary['missing_text_rows']:,}")
    print(f"Duplicate Tweet IDs         : {summary['duplicate_tweet_ids']:,}")
    print(f"Inbound (Customer) Tweets   : {summary['inbound_customer_tweets']:,}")
    print(f"Outbound (Brand) Tweets     : {summary['outbound_brand_tweets']:,}")
    print(f"Unique Total Authors        : {summary['unique_authors_total']:,}")
    print(f"Unique Brands Count         : {summary['unique_brands_count']:,}")
    print(f"Unique Customers Count      : {summary['unique_customers_count']:,}")
    print("\nTop 10 Brands by Tweet Count:")
    for brand, count in summary['top_brands'].items():
        print(f"  - @{brand:<20}: {count:,} tweets")
    print("="*60 + "\n")
    
    output_path = Path("results/dataset_inspection_summary.json")
    save_json(summary, output_path)
    logger.info(f"Summary report saved to {output_path}")

if __name__ == "__main__":
    main()
