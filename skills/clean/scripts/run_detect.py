"""Detect quality issues and print JSON report."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd
from agents.data_quality_agent import DataQualityAgent

df = pd.read_csv(ROOT / "data" / "raw" / "unified_dataset.csv")
agent = DataQualityAgent(text_col="text", label_col="label")
report = agent.detect_issues(df)

print(json.dumps({
    "total_rows": report.total_rows,
    "missing": {
        "count": report.missing.total_affected_rows,
        "pct": round(report.missing.total_affected_rows / report.total_rows * 100, 1),
    },
    "duplicates": {
        "count": report.duplicates.count,
        "pct": round(report.duplicates.percentage, 1),
    },
    "outliers": {
        "count": report.outliers.count,
        "pct": round(report.outliers.percentage, 1),
        "bounds": [report.outliers.lower_bound, report.outliers.upper_bound],
    },
    "imbalance": {
        "ratio": round(report.imbalance.imbalance_ratio, 2),
        "class_counts": report.imbalance.class_counts,
    },
}, default=str))
