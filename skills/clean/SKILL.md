# Skill: Data Quality

Detect and fix data quality issues using `DataQualityAgent`, then generate a quality report.

---

## Context
- Agent: `agents/data_quality_agent.py`
- Input: `data/raw/unified_dataset.csv`
- Output: `data/clean/pipeline_clean.csv`, `reports/quality_report.md`

---

## Steps

### 1. Run quality detection

```bash
python skills/clean/scripts/run_detect.py
```

This prints a JSON quality report with counts for:
- Missing values
- Duplicates
- Outliers (word-count IQR)
- Class imbalance ratio

### 2. HITL — Confirm cleaning strategy

Show the user the detected issues as a table:

| Issue | Count | % | Default fix |
|-------|-------|---|-------------|
| Missing values | N | X% | drop |
| Duplicates | N | X% | drop |
| Outliers | N | X% | clip_iqr |
| Imbalance | ratio | — | undersample |

**Ask** only if any of these are true:
- Outliers > 15% of dataset → ask: clip or drop?
- Imbalance ratio > 2.0 → ask: undersample or oversample?
- Missing values > 20% → ask: drop rows or fill with empty string?

If all issues are within normal bounds, use defaults and proceed WITHOUT asking.

Defaults:
```json
{"missing": "drop", "duplicates": "drop", "outliers": "clip_iqr", "imbalance": "undersample"}
```

### 3. Run cleaning

```bash
python skills/clean/scripts/run_clean.py --missing drop --duplicates drop --outliers clip_iqr --imbalance undersample
```

(Replace values with whatever was confirmed in step 2.)

### 4. LLM analysis (if ANTHROPIC_API_KEY is set)

The cleaning script automatically calls `explain_with_llm()`. If it returns a non-error string, include it in the report.

### 5. Report

Save `reports/quality_report.md` (the script does this).

Print:
```
✓ Data cleaning complete
  Before : N rows
  After  : N rows  (removed N)
  Report : reports/quality_report.md
  Output : data/clean/pipeline_clean.csv
```

---

## Constraints
- Do NOT ask about visualization choices, plot colors, or report formatting
- Do NOT ask the user to choose between IQR and z-score unless both give very different results (>5% difference in rows removed)
- Only one HITL checkpoint (step 2) — do not ask again after cleaning starts
