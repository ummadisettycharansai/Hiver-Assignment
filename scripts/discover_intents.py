import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hiver_agent.intents.discovery import get_default_intent_taxonomy
from hiver_agent.intents.taxonomy import save_intent_taxonomy
from hiver_agent.utils.logging import get_logger

logger = get_logger("discover_intents")

def main():
    parser = argparse.ArgumentParser(description="Discover and format intent taxonomy for target brand.")
    parser.add_argument("--output", type=str, default="configs/intents.yaml", help="Path to write intents.yaml")
    args = parser.parse_args()

    logger.info("Initializing empirical intent taxonomy for target brand...")
    taxonomy = get_default_intent_taxonomy()
    
    out_path = Path(args.output)
    save_intent_taxonomy(taxonomy, out_path)
    
    print("\n" + "="*70)
    print("                DISCOVERED INTENT TAXONOMY")
    print("="*70)
    print(f"Target Brand  : @{taxonomy.brand}")
    print(f"Total Intents : {len(taxonomy.intents)}")
    print(f"Saved Config  : {out_path}")
    print("-" * 70)
    for idx, item in enumerate(taxonomy.intents, 1):
        print(f" {idx:2d}. {item.name:<28} | {item.description}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
