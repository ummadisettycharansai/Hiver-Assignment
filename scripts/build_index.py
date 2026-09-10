import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hiver_agent.config import get_config
from hiver_agent.retrieval.index import VectorStoreIndex
from hiver_agent.utils.io import load_jsonl
from hiver_agent.utils.logging import get_logger

logger = get_logger("build_index")

def main():
    parser = argparse.ArgumentParser(description="Build FAISS retrieval index from training conversations.")
    parser.add_argument("--input-file", type=str, default=None, help="Path to train_conversations.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/processed/retrieval_index", help="Output directory for index")
    args = parser.parse_args()

    config = get_config()
    input_path = args.input_file or (Path(config.processed_data_dir) / "train_conversations.jsonl")
    
    logger.info(f"Loading training conversations from: {input_path}")
    train_convs = load_jsonl(input_path)
    
    index = VectorStoreIndex(embedding_model=config.embedding_model)
    index.build_index(train_convs)
    
    out_dir = Path(args.output_dir)
    index.save(out_dir)
    
    print("\n" + "="*70)
    print("               RETRIEVAL INDEX BUILD COMPLETE")
    print("="*70)
    print(f"Indexed Conversations : {len(index.metadata):,}")
    print(f"Embedding Model       : {config.embedding_model}")
    print(f"Saved Directory       : {out_dir}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
