import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hiver_agent.config import get_config
from hiver_agent.data.loader import load_raw_dataset
from hiver_agent.data.cleaner import clean_raw_dataframe
from hiver_agent.data.conversations import reconstruct_conversations
from hiver_agent.data.sampling import split_conversations_thread_level, create_reproducible_subsample
from hiver_agent.utils.logging import get_logger
from hiver_agent.utils.io import save_jsonl, save_json

logger = get_logger("build_dataset")

def main():
    parser = argparse.ArgumentParser(description="Process raw dataset into reconstructed AmazonHelp conversations and splits.")
    parser.add_argument("--brand", type=str, default=None, help="Target brand handle (e.g. AmazonHelp)")
    parser.add_argument("--sample-n", type=int, default=None, help="Rows from twcs.csv to process (None for full dataset)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    config = get_config()
    target_brand = args.brand or config.brand_name
    seed = args.seed or config.random_seed
    
    logger.info(f"Building dataset for brand: @{target_brand} (seed={seed}, sample_n={args.sample_n})...")
    df_raw = load_raw_dataset(config.raw_data_path, sample_n=args.sample_n)
    df_clean = clean_raw_dataframe(df_raw)
    
    # Reconstruct conversations for selected brand
    brand_convs = reconstruct_conversations(df_clean, target_brand=target_brand)
    
    # Require at least resolved conversations for retrieval & intent classifier training
    resolved_convs = [c for c in brand_convs if c["has_brand_response"]]
    logger.info(f"Reconstructed {len(brand_convs):,} total conversations, {len(resolved_convs):,} resolved conversations.")
    
    # Perform thread-level data split
    train_c, val_c, test_c = split_conversations_thread_level(resolved_convs, seed=seed)
    
    # Save processed jsonl files
    proc_dir = Path(config.processed_data_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    save_jsonl(brand_convs, proc_dir / "all_conversations.jsonl")
    save_jsonl(train_c, proc_dir / "train_conversations.jsonl")
    save_jsonl(val_c, proc_dir / "val_conversations.jsonl")
    save_jsonl(test_c, proc_dir / "test_conversations.jsonl")
    
    # Save subsample for fast 15-minute reproduction requirement
    subsample_dir = Path(config.subsample_dir)
    subsample_dir.mkdir(parents=True, exist_ok=True)
    
    subsample_convs = create_reproducible_subsample(resolved_convs, max_count=config.sample_size, seed=seed)
    sub_train, sub_val, sub_test = split_conversations_thread_level(subsample_convs, seed=seed)
    
    save_jsonl(subsample_convs, subsample_dir / "subsample_conversations.jsonl")
    save_jsonl(sub_train, subsample_dir / "train_conversations.jsonl")
    save_jsonl(sub_val, subsample_dir / "val_conversations.jsonl")
    save_jsonl(sub_test, subsample_dir / "test_conversations.jsonl")
    
    meta = {
        "brand": target_brand,
        "seed": seed,
        "total_conversations": len(brand_convs),
        "resolved_conversations": len(resolved_convs),
        "full_split": {"train": len(train_c), "val": len(val_c), "test": len(test_c)},
        "subsample_split": {"train": len(sub_train), "val": len(sub_val), "test": len(sub_test)}
    }
    save_json(meta, proc_dir / "dataset_metadata.json")
    
    print("\n" + "="*70)
    print("             DATASET RECONSTRUCTION & SPLITTING SUMMARY")
    print("="*70)
    print(f"Target Brand             : @{target_brand}")
    print(f"Total Reconstructed      : {len(brand_convs):,} threads")
    print(f"Resolved Threads         : {len(resolved_convs):,} threads")
    print(f"Full Train / Val / Test  : {len(train_c):,} / {len(val_c):,} / {len(test_c):,}")
    print(f"Subsample (15-min demo)  : {len(sub_train):,} train / {len(sub_val):,} val / {len(sub_test):,} test")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
