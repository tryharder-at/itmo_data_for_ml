# Skill: Active Learning + Model Training

Run an Active Learning cycle using entropy uncertainty sampling, train a final
sentiment model, and save it with full metrics.

---

## Context
- Agent: `agents/al_agent.py`
- Input: `data/labeled/final_dataset.csv`
- Outputs: `models/sentiment_model.joblib`, `data/al/pipeline_learning_curve.png`, `reports/al_report.md`

---

## Steps

### 1. Run the AL cycle

```bash
python skills/al/scripts/run_al.py
```

This runs:
- Entropy strategy (5 iterations × 20 examples)
- Random baseline for comparison
- Saves the **best** model by F1-macro (not the last iteration) to `models/sentiment_model.joblib`
- Saves learning curve to `data/al/pipeline_learning_curve.png`
- Saves `reports/al_report.md` with savings analysis

### 2. Show results

After the script finishes, display:
- A table of iteration metrics (n_labeled, accuracy, f1_macro)
- The best iteration (highest f1_macro) highlighted
- Savings vs random baseline (from JSON field `savings_vs_random`)

Tell the user where to find the image:
```
Learning curve: data/al/pipeline_learning_curve.png
```

### 3. HITL — Confirm model is ready for production

Show the **best iteration** metrics (from JSON field `best_metrics`):

| Metric | Value |
|--------|-------|
| N labeled (best) | N |
| Accuracy | X.XXXX |
| F1-macro | X.XXXX |
| F1-positive | X.XXXX |

**Ask:** "The model has been trained. Would you like to (a) accept and save, or (b) run more AL iterations?"
- `accept` (default) → proceed
- `more` → ask how many additional iterations, then re-run:
  ```bash
  python skills/al/scripts/run_al.py --n-iter N
  ```
  Note: this restarts the AL cycle from scratch with N total iterations.
  The best model across all iterations is saved automatically.

> Only ask this once. If the user says `accept`, finalize.
> Do NOT ask about hyperparameters, model architecture, or TF-IDF settings.

### 4. Confirm LLM analysis

The script automatically calls Claude API to explain the learning curves
(requires `ANTHROPIC_API_KEY` in `.env`). The explanation is included in `reports/al_report.md`.

### 5. Report

Print:
```
✓ Active Learning complete
  Best accuracy  : X.XXXX  (iter N, n=XXX labeled)
  Best F1-macro  : X.XXXX
  Savings vs random: N examples
  Model saved    : models/sentiment_model.joblib
  Learning curve : data/al/pipeline_learning_curve.png
  Report         : reports/al_report.md
```

Use `best_metrics` from the JSON output, not `final_metrics`.

---

## Constraints
- Do NOT ask about model choice (LogisticRegression is fixed)
- Do NOT ask about entropy vs margin — entropy is the default strategy
- Do NOT ask about TF-IDF parameters
- Only one HITL checkpoint (step 3) — accept/more iterations only
- When reporting metrics, always use the best iteration, not the last
