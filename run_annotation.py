"""
CLI entry point for the AnnotationAgent.

Usage:
    python run_annotation.py
    python run_annotation.py --input data/raw/unified_dataset.csv --sample 300
    python run_annotation.py --zero-shot   # use zero-shot classification
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.annotation_agent import AnnotationAgent
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AnnotationAgent")
    parser.add_argument("--input", default="data/raw/unified_dataset.csv")
    parser.add_argument("--sample", type=int, default=300,
                        help="Number of rows to process (0 = all)")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="HITL confidence threshold")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Use zero-shot classification instead of SST-2")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.sample and args.sample < len(df):
        df = df.groupby("source").apply(
            lambda g: g.sample(min(args.sample // 3, len(g)), random_state=42)
        ).reset_index(drop=True)

    print(f"Input: {len(df)} rows")

    agent = AnnotationAgent(
        modality="text",
        confidence_threshold=args.threshold,
        zero_shot=args.zero_shot,
    )

    # 1. Auto-label
    print("\n[1/4] Auto-labeling...")
    df_labeled = agent.auto_label(df)
    print(f"  → {dict(df_labeled['predicted_label'].value_counts())}")
    print(f"  → Mean confidence: {df_labeled['confidence'].mean():.3f}")

    # 2. Generate spec
    print("\n[2/4] Generating annotation spec...")
    spec_path = agent.generate_spec(df, task="sentiment_classification")
    print(f"  → {spec_path}")

    # 3. Quality check
    print("\n[3/4] Checking quality...")
    metrics = agent.check_quality(df_labeled)
    print(f"  → Cohen's κ     : {metrics.get('kappa')}")
    print(f"  → Agreement     : {metrics.get('percent_agreement')}%")
    print(f"  → Confidence μ  : {metrics.get('confidence_mean')}")
    print(f"  → Low-conf rows : {metrics.get('low_confidence_count')}")

    # 4. Export to LabelStudio
    print("\n[4/4] Exporting to LabelStudio...")
    ls = agent.export_to_labelstudio(df_labeled)
    print(f"  → {ls['n_tasks']} tasks  |  {ls['n_with_predictions']} with predictions")
    print(f"  → {ls['json_path']}")

    # BONUS: HITL
    print(f"\n[BONUS] HITL flagging (threshold={args.threshold})...")
    low_df = agent.flag_low_confidence(df_labeled)
    print(f"  → {len(low_df)} rows flagged for manual review")

    print("\nDone. All outputs in data/annotations/")


if __name__ == "__main__":
    main()
