"""Run DataCollectionAgent and print a JSON summary."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

from agents.data_collection_agent import DataCollectionAgent

agent = DataCollectionAgent(config=str(ROOT / "config.yaml"))
df = agent.run()

summary = {
    "total_rows": len(df),
    "by_source": df["source"].value_counts().to_dict(),
    "by_label": df["label"].value_counts().to_dict(),
    "output": "data/raw/unified_dataset.csv",
}
print(json.dumps(summary, indent=2, default=str))
