"""
Entry point — run the DataCollectionAgent with config.yaml and print a summary.

Usage:
    python run_agent.py
    python run_agent.py --config config.yaml
"""

import argparse
import sys
from pathlib import Path

# Allow importing agent from the agents/ package
sys.path.insert(0, str(Path(__file__).parent))

from agents.data_collection_agent import DataCollectionAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DataCollectionAgent")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml"
    )
    args = parser.parse_args()

    agent = DataCollectionAgent(config=args.config)
    df = agent.run()

    if df.empty:
        print("No data collected. Check your config and network connection.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  COLLECTION COMPLETE")
    print("=" * 60)
    print(f"  Total rows   : {len(df):,}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"\nRows per source:")
    print(df["source"].value_counts().to_string())
    print(f"\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print("\nSample rows:")
    print(df.head(5).to_string(index=False, max_colwidth=60))
    print("=" * 60)
    print(f"\nOutput saved to: data/raw/unified_dataset.csv")


if __name__ == "__main__":
    main()
