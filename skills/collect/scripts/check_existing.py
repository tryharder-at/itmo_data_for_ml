"""Check if unified_dataset.csv exists and print stats."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
p = ROOT / "data" / "raw" / "unified_dataset.csv"

if not p.exists():
    print(json.dumps({"exists": False}))
    sys.exit(0)

import pandas as pd
df = pd.read_csv(p)
print(json.dumps({
    "exists": True,
    "rows": len(df),
    "by_source": df["source"].value_counts().to_dict(),
    "by_label": df["label"].value_counts().to_dict(),
}, default=str))
