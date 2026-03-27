# Skill: Data Collection

Collect sentiment-labelled text from multiple sources using `DataCollectionAgent`.

---

## Context
- Agent: `agents/data_collection_agent.py`
- Config: `config.yaml` (sources: IMDB via HuggingFace, books.toscrape.com, OpenLibrary API)
- Output: `data/raw/unified_dataset.csv`

---

## Steps

### 1. Check existing data

Run:
```bash
python skills/collect/scripts/check_existing.py
```

This prints whether `data/raw/unified_dataset.csv` exists and its row count.

### 2. Ask user whether to collect or reuse

Show the user:
- Whether the file exists
- Row count and label distribution (if it exists)

**Ask:** "Re-collect fresh data, or use the existing dataset?"
- `reuse` → skip to step 5
- `collect` (default if no file) → continue

> Do NOT ask about config details, source names, or label mapping — those are already in `config.yaml`.

### 3. Run collection

```bash
python skills/collect/scripts/run_collect.py
```

Show live progress. If a source fails with a network error, report it and continue with remaining sources — do not abort the whole run.

### 4. HITL — Confirm collected data

Show the user a summary table:

| Source | Rows | Positive | Negative |
|--------|------|----------|----------|

**Ask:** "Does the collected data look good? (yes / recollect)"

Only ask if:
- Any source returned 0 rows
- Label distribution is severely imbalanced (>80% one class)
- Total rows < 200

If everything looks fine, proceed automatically without asking.

### 5. Report

Print final stats:
```
✓ Data collection complete
  Total rows   : N
  Positive     : N  (X%)
  Negative     : N  (X%)
  Output       : data/raw/unified_dataset.csv
```

---

## Constraints
- Do NOT ask about colors, plot styles, or minor formatting
- Do NOT ask the user to pick between sources — config.yaml defines them
- Only surface blockers: 0-row sources, severe imbalance, file write errors
