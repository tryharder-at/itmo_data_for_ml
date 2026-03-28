"""Run AnnotationAgent on pipeline_clean.csv and write review_queue.csv."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd
from agents.annotation_agent import AnnotationAgent

df = pd.read_csv(ROOT / "data" / "clean" / "pipeline_clean.csv")

# Stratified sample: up to 100 rows per source
parts = [
    grp.sample(min(100, len(grp)), random_state=42)
    for _, grp in df.groupby("source")
]
sample = pd.concat(parts, ignore_index=True)

agent = AnnotationAgent(
    modality="text",
    labels=["positive", "negative"],
    text_model="distilbert-base-uncased-finetuned-sst-2-english",
    zero_shot=False,
    batch_size=64,
    confidence_threshold=0.70,
    output_dir=str(ROOT / "data" / "annotations"),
)

df_labeled = agent.auto_label(sample)
metrics = agent.check_quality(df_labeled, reference_col="label", pred_col="predicted_label")

# Write review queue (low-confidence)
thr = 0.70
df_low  = df_labeled[df_labeled["confidence"] <  thr]
df_high = df_labeled[df_labeled["confidence"] >= thr]

df_low.to_csv(ROOT / "review_queue.csv", index=False)
df_labeled.to_csv(ROOT / "data" / "annotations" / "pipeline_annotated.csv", index=False)

# Export to LabelStudio format (tasks 3 & 5 requirement)
agent.export_to_labelstudio(df_labeled)

print(json.dumps({
    "total": len(df_labeled),
    "high_conf": len(df_high),
    "low_conf": len(df_low),
    "mean_conf": round(float(df_labeled["confidence"].mean()), 3),
    "kappa": round(float(metrics.get("kappa", 0)), 3),
    "agreement_pct": metrics.get("percent_agreement"),
    "by_predicted_label": df_labeled["predicted_label"].value_counts().to_dict(),
    "review_queue": "review_queue.csv",
}, default=str))
