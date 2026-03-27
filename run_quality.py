"""
CLI entry point for the DataQualityAgent.

Usage:
    python run_quality.py                    # uses data/raw/unified_dataset.csv
    python run_quality.py --input path.csv  # custom input
    python run_quality.py --llm             # enable Claude LLM explanation (needs .env)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.data_quality_agent import DataQualityAgent
import numpy as np
import pandas as pd


def inject_issues(df: pd.DataFrame, seed: int = 99) -> pd.DataFrame:
    """Inject synthetic quality issues into a copy of the DataFrame."""
    rng = np.random.default_rng(seed)
    dirty = df.copy()

    # 5% missing text
    n_missing = int(len(dirty) * 0.05)
    dirty.loc[rng.choice(len(dirty), n_missing, replace=False), 'text'] = np.nan

    # 4% duplicate rows
    n_dups = int(len(dirty) * 0.04)
    dup_idx = rng.choice(len(dirty), n_dups, replace=False)
    dirty = pd.concat([dirty, dirty.iloc[dup_idx]], ignore_index=True)

    # Skew class balance 2×
    pos_idx = dirty[dirty['label'] == 'positive'].index
    drop_pos = rng.choice(pos_idx, int(len(pos_idx) * 0.60), replace=False)
    dirty = dirty.drop(index=drop_pos).reset_index(drop=True)

    return dirty


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DataQualityAgent")
    parser.add_argument("--input", default="data/raw/unified_dataset.csv")
    parser.add_argument("--llm", action="store_true", help="Call Claude API for explanation")
    args = parser.parse_args()

    df_raw = pd.read_csv(args.input)
    print(f"Loaded {len(df_raw):,} rows from {args.input}")

    print("\nInjecting synthetic issues for demonstration...")
    df_dirty = inject_issues(df_raw)
    print(f"Dirty dataset: {len(df_dirty):,} rows")

    agent = DataQualityAgent()
    report = agent.detect_issues(df_dirty)

    if args.llm:
        print("\n--- LLM Explanation (Claude) ---")
        explanation = agent.explain_with_llm(report)
        print(explanation)

    # Apply two strategies and compare
    print("\n--- Strategy A: Aggressive ---")
    df_a = agent.fix(df_dirty, {
        'missing': 'drop', 'duplicates': 'drop',
        'outliers': 'drop_iqr', 'imbalance': 'undersample'
    })

    print("\n--- Strategy B: Conservative ---")
    df_b = agent.fix(df_dirty, {
        'missing': 'fill', 'duplicates': 'drop',
        'outliers': 'clip_iqr', 'imbalance': 'oversample'
    })

    print("\n--- Comparison ---")
    cmp_a = agent.compare(df_dirty, df_a, "A: Aggressive")
    cmp_b = agent.compare(df_dirty, df_b, "B: Conservative")
    comparison = pd.concat([cmp_a, cmp_b], ignore_index=True)
    print(comparison.to_string(index=False))

    # Save cleaned datasets
    Path("data/clean").mkdir(parents=True, exist_ok=True)
    df_a.to_csv("data/clean/cleaned_strategy_a.csv", index=False)
    df_b.to_csv("data/clean/cleaned_strategy_b.csv", index=False)
    print("\nSaved cleaned datasets to data/clean/")


if __name__ == "__main__":
    main()
