# Sentiment Analysis ML Pipeline

End-to-end data pipeline for binary sentiment classification —
from raw web data to a trained model, with Human-in-the-Loop label review.

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set your Anthropic API key for LLM bonus steps
cp .env.example .env
# edit .env and add your key: ANTHROPIC_API_KEY=sk-ant-...

# 4. Run the full pipeline (interactive HITL)
python run_pipeline.py

# 4a. Run non-interactively (CI / demo mode)
python run_pipeline.py --auto-hitl

# 4b. Data collection is automatically skipped if unified_dataset.csv already exists
# Use --force-collect / --force-clean / --force-annotate to re-run individual steps
python run_pipeline.py --force-collect --auto-hitl

# 5. (Bonus) Launch the Streamlit HITL app instead of editing CSV manually
streamlit run hitl_app.py
```

> **Note:** Data collection (web scraping + HuggingFace download) takes 2–5 min
> on first run. Use `--skip-collect` on subsequent runs to reuse cached data.

---

## Repository Structure

```
├── agents/
│   ├── data_collection_agent.py   # Task 1 — DataCollectionAgent
│   ├── data_quality_agent.py      # Task 2 — DataQualityAgent
│   ├── annotation_agent.py        # Task 3 — AnnotationAgent
│   └── al_agent.py                # Task 4 — ActiveLearningAgent
│
├── notebooks/
│   ├── eda.ipynb                  # Task 1 — Exploratory Data Analysis
│   ├── quality_report.ipynb       # Task 2 — Quality Detective & Surgeon
│   ├── annotation.ipynb           # Task 3 — Annotation experiment
│   └── al_experiment.ipynb        # Task 4 — AL strategies comparison
│
├── data/
│   ├── raw/                       # Step 1 output: unified_dataset.csv
│   ├── clean/                     # Step 2 output: pipeline_clean.csv
│   ├── annotations/               # Step 3 output: pipeline_annotated.csv
│   ├── labeled/                   # HITL output: final_dataset.csv + data_card.md
│   └── al/                        # Step 5 output: learning curves
│
├── models/
│   └── sentiment_model.joblib     # Trained sklearn Pipeline (TF-IDF + LogReg)
│
├── reports/
│   ├── quality_report.md          # Data quality metrics + LLM analysis
│   ├── annotation_report.md       # Auto-labeling metrics
│   ├── al_report.md               # AL learning curves + LLM analysis
│   └── pipeline_summary.md        # Full 5-section summary
│
├── review_queue.csv               # HITL: low-confidence examples for review
├── review_queue_corrected.csv     # HITL: after human corrections (auto-created)
├── run_pipeline.py                # ← Main pipeline entry point
├── hitl_app.py                    # Streamlit HITL app (bonus)
├── run_agent.py                   # CLI for Task 1 standalone
├── run_quality.py                 # CLI for Task 2 standalone
├── run_annotation.py              # CLI for Task 3 standalone
├── run_al.py                      # CLI for Task 4 standalone
├── config.yaml                    # Data collection configuration
├── requirements.txt
└── .env.example
```

---

## 1. Task and Dataset

**ML task:** Binary sentiment classification — predict whether a text expresses
**positive** or **negative** sentiment.

**Modality:** Text (English)

**Data sources:**

| Source | Type | Label method |
|--------|------|-------------|
| HuggingFace `imdb` | Movie reviews | Human annotation |
| books.toscrape.com | Book reviews | Star rating → binary |
| OpenLibrary REST API | Book metadata | Avg. rating threshold 3.8 |

**Raw dataset:** ~1 400 rows · ~50/50 class split (after shuffle fix)
**Final labeled dataset:** ~300 rows · balanced · see `data/labeled/data_card.md`

---

## 2. What Each Agent Did

### DataCollectionAgent (Task 1)
- Loads a shuffled sample from IMDB (`datasets` library) to avoid the
  all-negative bias present in the first N records.
- Scrapes books.toscrape.com (20 pages); maps 1–2 stars → negative,
  4–5 stars → positive; skips 3-star neutral.
- Queries OpenLibrary REST API with exponential-backoff retry logic;
  labels by average rating ≥ 3.8.
- Outputs a unified schema: `text, label, source, collected_at`.

### DataQualityAgent (Task 2)
- Detects 4 issue types: missing values, duplicates, word-count outliers
  (IQR), and class imbalance.
- Strategy chosen for the pipeline: `drop + drop + clip_iqr + undersample` —
  preserves all text while achieving a perfectly balanced set.
- **LLM bonus:** Claude explains why this combination is preferable
  (saved to `reports/quality_report.md`).

### AnnotationAgent (Task 3)
- Auto-labels with `distilbert-base-uncased-finetuned-sst-2-english` (66 M params).
- Falls back to VADER → keyword matching if the model is unavailable.
- Runs on Apple Silicon MPS / CUDA / CPU depending on hardware.
- Mean confidence 0.955; Cohen's κ = 0.27 (low due to mixed label sources).
- **HITL bonus:** flags all predictions with confidence < 0.70 for human review.

### ActiveLearningAgent (Task 4)
- Implements entropy sampling, margin sampling, and random baseline.
- Entropy strategy reaches the random baseline's final quality at N=50
  (vs N=150 for random) — saves **67% of labeling effort**.
- **LLM bonus:** Claude analyses the learning curves and recommends a
  stopping point (saved to `reports/al_report.md`).

---

## 3. Human-in-the-Loop

**HITL checkpoint** is between Step 3 (auto-annotation) and Step 5 (AL training).

### What happens

1. `AnnotationAgent` labels all examples; predictions with confidence < 0.70
   are written to **`review_queue.csv`**.
2. The pipeline **pauses** and prints instructions:
   - **Option A:** Open `review_queue.csv` in a spreadsheet, edit the
     `predicted_label` column for wrong rows, save as `review_queue_corrected.csv`.
   - **Option B:** Run `streamlit run hitl_app.py` for a visual review UI
     with a drop-down label editor and diff summary.
3. After the human presses **Enter**, the corrected file is merged with the
   high-confidence auto-accepted subset to form the final training set.

### Statistics (typical run)

| Metric | Value |
|--------|-------|
| Total annotated | ~300 |
| Auto-accepted (conf ≥ 0.70) | ~289 (~96%) |
| Sent for review (conf < 0.70) | ~11 (~4%) |
| HITL tool | `hitl_app.py` (Streamlit) |

Even a 4% correction rate on uncertain predictions reduces label noise
and improves model F1 at train time.

---

## 4. Metrics at Each Stage

| Stage | Metric | Typical value |
|-------|--------|--------------|
| Raw collection | Rows | ~1 417 |
| After cleaning | Rows | ~538 |
| Auto-annotation | Mean confidence | 0.955 |
| Auto-annotation | Cohen's κ | 0.27 (Fair) |
| AL — iter 0 (N=50) | F1-macro | ~0.47 |
| AL — iter 3 (N=110) | F1-macro | ~0.65 |
| **Final model** | **Accuracy** | **see `reports/al_report.md`** |
| **Final model** | **F1-macro** | **see `reports/al_report.md`** |

Exact numbers depend on HITL corrections made. Reports generated automatically.

**Load and use the saved model:**
```python
import joblib
model = joblib.load("models/sentiment_model.joblib")
model.predict(["This film was absolutely fantastic!"])
# → ['positive']
```

---

## 5. Retrospective

### What worked
- **TF-IDF + LogReg** reaches ~65% F1 with only 110 labeled examples on a
  noisy mixed-source dataset — fast and interpretable.
- **Entropy uncertainty sampling** saves ~67% of the labeling budget vs random
  for the same quality target.
- **DistilBERT SST-2** produces highly confident predictions (mean 0.955) —
  only ~4% of examples need human review.
- **Pipeline-as-code** design (pure Python, no Prefect/Airflow) is easy to
  reproduce on any machine: `pip install -r requirements.txt && python run_pipeline.py`.

### What didn't work as expected
- **Cohen's κ = 0.27** despite high model confidence. Root cause: heterogeneous
  label sources — IMDB uses human annotations while books.toscrape uses star
  ratings as a proxy. Comparing only against IMDB raises κ to ~0.7.
- **Small dataset** (~300 rows after cleaning) limits generalisation. F1
  plateaus early; more AL iterations would not help without more pool data.
- **OpenLibrary API** occasionally returns 503 errors — mitigated by
  exponential-backoff retry, but some runs may get fewer API examples.

### What I would do differently
1. Use a **single coherent labeling scheme** (all human-annotated or all
   model-annotated) to avoid κ anomalies.
2. Collect a **larger pool** (1 000+ examples) to better demonstrate AL savings.
3. Fine-tune **DistilBERT** end-to-end instead of TF-IDF to push F1 above 0.80.
4. Add a **validation split** tracked across runs for longitudinal comparison.
5. Use **Prefect** `@flow` / `@task` decorators for production observability.

---

## Pipeline CLI Reference

```
python run_pipeline.py [OPTIONS]

  --force-collect      Re-collect raw data  (default: reuse if exists)
  --force-clean        Re-run data cleaning (default: reuse if exists)
  --force-annotate     Re-run annotation    (default: reuse if exists)
  --auto-hitl          Skip interactive HITL checkpoint (CI mode)
  --n-initial INT      Initial AL labeled set size  (default: 50)
  --n-test    INT      Hold-out test set size       (default: 100)
  --n-iter    INT      AL iterations                (default: 5)
  --batch-size INT     Examples per AL iteration    (default: 20)
```

---

*ITMO "Data for ML" course — Assignments 1–5.*
