"""Merge high-confidence auto-accepted + human-corrected labels into final dataset."""
import json, sys, datetime
from pathlib import Path

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd

thr = 0.70
df_annotated = pd.read_csv(ROOT / "data" / "annotations" / "pipeline_annotated.csv")
df_high = df_annotated[df_annotated["confidence"] >= thr].copy()
df_low  = df_annotated[df_annotated["confidence"] <  thr].copy()

corrected_path = ROOT / "review_queue_corrected.csv"
if corrected_path.exists():
    df_low_corrected = pd.read_csv(corrected_path)
    n_changed = (df_low_corrected["predicted_label"].values
                 != df_low["predicted_label"].values[:len(df_low_corrected)]).sum()
else:
    df_low_corrected = df_low.copy()
    n_changed = 0

df_high["final_label"] = df_high["predicted_label"]
df_low_corrected["final_label"] = df_low_corrected["predicted_label"]

df_final = pd.concat([df_high, df_low_corrected], ignore_index=True)

(ROOT / "data" / "labeled").mkdir(parents=True, exist_ok=True)
df_final.to_csv(ROOT / "data" / "labeled" / "final_dataset.csv", index=False)

# Data card
src_dist = df_final.get("source", pd.Series(dtype=str)).value_counts().to_dict()
card = f"""# Data Card — Sentiment Classification Dataset

| Field | Value |
|-------|-------|
| Task | Binary sentiment classification |
| Classes | positive, negative |
| Total examples | {len(df_final)} |
| Language | English |
| Created | {datetime.datetime.now().strftime("%Y-%m-%d")} |

## Class Distribution
| Label | Count | % |
|-------|-------|---|
| positive | {(df_final["final_label"]=="positive").sum()} | {(df_final["final_label"]=="positive").mean()*100:.1f}% |
| negative | {(df_final["final_label"]=="negative").sum()} | {(df_final["final_label"]=="negative").mean()*100:.1f}% |

## Sources
{chr(10).join(f"| {s} | {c} |" for s, c in src_dist.items())}

## HITL
- Auto-accepted (conf ≥ {thr}): {len(df_high)}
- Human-reviewed (conf < {thr}): {len(df_low)}
- Labels changed by human: {n_changed}
"""
(ROOT / "data" / "labeled" / "data_card.md").write_text(card, encoding="utf-8")

# Annotation report
(ROOT / "reports").mkdir(parents=True, exist_ok=True)
report_md = f"""# Annotation Report

Generated: {datetime.datetime.now().isoformat(timespec='seconds')}

## Model
- distilbert-base-uncased-finetuned-sst-2-english
- Confidence threshold: {thr}

## Results
- Rows labeled: {len(df_annotated)}
- Mean confidence: {df_annotated['confidence'].mean():.3f}
- Auto-accepted: {len(df_high)} ({len(df_high)/len(df_annotated)*100:.1f}%)
- Human-reviewed: {len(df_low)} ({len(df_low)/len(df_annotated)*100:.1f}%)
- Labels changed: {n_changed}
- Final dataset: {len(df_final)} rows
"""
(ROOT / "reports" / "annotation_report.md").write_text(report_md, encoding="utf-8")

print(json.dumps({
    "final_rows": len(df_final),
    "high_conf": len(df_high),
    "human_reviewed": len(df_low),
    "labels_changed": n_changed,
    "output": "data/labeled/final_dataset.csv",
}, default=str))
