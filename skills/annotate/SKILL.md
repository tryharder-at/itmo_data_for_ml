# Skill: Auto-Annotation + HITL Review

Auto-label text data with DistilBERT SST-2, then run a Human-in-the-Loop
review of uncertain predictions via a Streamlit UI.

---

## Context
- Agent: `agents/annotation_agent.py`
- Input: `data/clean/pipeline_clean.csv`
- HITL UI: `skills/annotate/scripts/hitl_app.py`
- Outputs: `data/labeled/final_dataset.csv`, `data/labeled/data_card.md`, `reports/annotation_report.md`

---

## Steps

### 1. Run auto-annotation

```bash
python skills/annotate/scripts/run_annotate.py
```

This labels a stratified sample (up to 100 rows per source) with DistilBERT SST-2,
writes `review_queue.csv` for HITL, and exports `data/annotations/labelstudio_import.json`
for LabelStudio import.

### 2. Show annotation statistics

After the script finishes, read and display the printed JSON summary:
- Total rows labeled
- Mean confidence
- Predicted label distribution
- Number of low-confidence rows (< 0.70)

### 3. HITL — Launch Streamlit review UI

This is the **mandatory human checkpoint**. Always show it, even if low-conf count is 0.

Tell the user:

```
❗ HITL CHECKPOINT — Annotation Review

N examples have confidence < 0.70 and need your review.
A Streamlit app will open in your browser for visual label editing.

Run in a separate terminal:
  streamlit run skills/annotate/scripts/hitl_app.py

The app will open at: http://localhost:8501
```

**Wait** for the user to confirm they are done reviewing.
Ask: "Have you finished reviewing labels in the Streamlit app? (yes/skip)"
- `yes` → read `review_queue_corrected.csv` and merge
- `skip` → accept all predictions as-is (auto-accept mode)

> Do NOT ask: "Should I open the browser for you?" — just print the command.
> Do NOT ask about confidence threshold changes — 0.70 is fixed.

### 4. Merge and save final labeled dataset

```bash
python skills/annotate/scripts/merge_labels.py
```

This merges high-confidence auto-accepted rows with the human-corrected low-confidence rows,
and saves `data/labeled/final_dataset.csv` + `data/labeled/data_card.md`.

Also tell the user:
```
LabelStudio export : data/annotations/labelstudio_import.json
```

### 5. Report

Print:
```
✓ Annotation complete
  Labeled      : N rows
  Auto-accepted: N rows  (conf ≥ 0.70)
  Human-reviewed: N rows
  Mean conf    : 0.XXX
  Cohen's κ    : 0.XXX
  Output       : data/labeled/final_dataset.csv
  Report       : reports/annotation_report.md
```

---

## Constraints
- Do NOT ask about model choice (DistilBERT SST-2 is fixed)
- Do NOT ask about confidence threshold
- Do NOT ask about batch size or device
- Only one HITL checkpoint (step 3) — after the UI confirms done, proceed automatically
- If `review_queue.csv` has 0 rows, skip the HITL UI and proceed
