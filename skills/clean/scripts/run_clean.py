"""Apply cleaning strategy and save clean dataset + quality report."""
import argparse, json, sys, datetime
from pathlib import Path

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd
from agents.data_quality_agent import DataQualityAgent

parser = argparse.ArgumentParser()
parser.add_argument("--missing",    default="drop")
parser.add_argument("--duplicates", default="drop")
parser.add_argument("--outliers",   default="clip_iqr")
parser.add_argument("--imbalance",  default="undersample")
args = parser.parse_args()

strategy = {
    "missing":    args.missing,
    "duplicates": args.duplicates,
    "outliers":   args.outliers,
    "imbalance":  args.imbalance,
}

df = pd.read_csv(ROOT / "data" / "raw" / "unified_dataset.csv")
agent = DataQualityAgent(text_col="text", label_col="label")

report = agent.detect_issues(df)
df_clean = agent.fix(df, strategy=strategy)

# Save clean dataset
(ROOT / "data" / "clean").mkdir(parents=True, exist_ok=True)
df_clean.to_csv(ROOT / "data" / "clean" / "pipeline_clean.csv", index=False)

# LLM analysis
llm_advice = agent.explain_with_llm(report)

# Save report
(ROOT / "reports").mkdir(parents=True, exist_ok=True)
report_md = f"""# Data Quality Report

Generated: {datetime.datetime.now().isoformat(timespec='seconds')}

## Raw Dataset
- Rows: {report.total_rows}
- Missing: {report.missing.total_affected_rows} rows ({report.missing.total_affected_rows/report.total_rows*100:.1f}%)
- Duplicates: {report.duplicates.count} ({report.duplicates.percentage:.1f}%)
- Outliers: {report.outliers.count} ({report.outliers.percentage:.1f}%)
- Imbalance ratio: {report.imbalance.imbalance_ratio:.2f}x

## Strategy Applied
| Issue | Strategy |
|-------|---------|
| Missing | {strategy["missing"]} |
| Duplicates | {strategy["duplicates"]} |
| Outliers | {strategy["outliers"]} |
| Imbalance | {strategy["imbalance"]} |

## After Cleaning
- Rows: {len(df_clean)}
- Removed: {report.total_rows - len(df_clean)}
- Distribution: {dict(df_clean["label"].value_counts())}

## LLM Analysis (Claude)
{llm_advice}
"""
(ROOT / "reports" / "quality_report.md").write_text(report_md, encoding="utf-8")

print(json.dumps({
    "before": report.total_rows,
    "after": len(df_clean),
    "removed": report.total_rows - len(df_clean),
    "strategy": strategy,
    "output": "data/clean/pipeline_clean.csv",
    "report": "reports/quality_report.md",
}, default=str))
