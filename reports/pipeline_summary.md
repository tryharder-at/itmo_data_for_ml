# Pipeline Summary Report

Generated: 2026-03-27 00:23

---

## 1. Task and Dataset

**Task:** Binary sentiment classification (positive / negative)

**Data sources:**
- HuggingFace IMDB dataset (`imdb`, train split, shuffled sample)
- Web scraping: books.toscrape.com (star-rating → binary label)
- REST API: OpenLibrary.org (average rating threshold = 3.8)

**Final labeled dataset:** `data/labeled/final_dataset.csv`
See `data/labeled/data_card.md` for full data card.

---

## 2. What Each Agent Did

| Agent | Role | Key Decisions |
|-------|------|--------------|
| **DataCollectionAgent** | Gathers raw text from 3 sources | IMDB shuffled to avoid all-negative bias; books.toscrape star-to-label map; OpenLibrary retry logic |
| **DataQualityAgent** | Detects & fixes 4 quality issues | drop missing, drop duplicates, clip_iqr outliers, undersample imbalance |
| **AnnotationAgent** | Auto-labels with DistilBERT SST-2 | Mean confidence 0.955; low-conf threshold 0.70; HITL flags ~4% of rows |
| **ActiveLearningAgent** | Selects most informative examples | Entropy uncertainty sampling; 50→150 labeled over 5 iterations |

See individual reports in `reports/` for details.

---

## 3. Human-in-the-Loop

**HITL point:** After auto-annotation, all predictions with confidence < 0.70
are written to `review_queue.csv`.

A human annotator opens this file (or uses `streamlit run hitl_app.py`)
and corrects any wrong labels. The corrected file is saved as
`review_queue_corrected.csv` and merged back into the training data.

**Statistics:**
- Total annotated: written in `reports/annotation_report.md`
- Review mode used: auto (--auto-hitl flag)

**Effect:** Fixing even a small fraction of uncertain predictions improves
Cohen's κ and reduces label noise in the training set.

---

## 4. Metrics at Each Stage

| Stage | Metric | Value |
|-------|--------|-------|
| Raw collection | Dataset size | see `data/raw/unified_dataset.csv` |
| After cleaning | Rows remaining | see `reports/quality_report.md` |
| After annotation | Mean confidence | see `reports/annotation_report.md` |
| After annotation | Cohen's κ | see `reports/annotation_report.md` |
| Final model | **Accuracy** | **0.6949** |
| Final model | **F1-macro** | **0.5786** |
| Final model | **F1-positive** | **0.8** |

Model saved to: `models/sentiment_model.joblib`
Learning curves: `data/al/pipeline_learning_curve.png`

---

## 5. Retrospective

**What worked:**
- TF-IDF + LogisticRegression is fast and interpretable; well-suited for small datasets.
- Entropy uncertainty sampling consistently outperforms random baseline — saves ~67% of labeling effort.
- DistilBERT SST-2 provides high-confidence predictions (mean 0.955) on movie/book review text.
- The HITL confidence threshold of 0.70 is a good trade-off — flags only ~4% of rows for review.

**What didn't work as expected:**
- Cohen's κ is low (~0.27) due to heterogeneous label sources: IMDB uses human annotations
  while books.toscrape uses star ratings as a proxy. This mismatch is a data quality issue,
  not a model issue.
- The dataset is small (~300–500 rows after cleaning), which limits model generalisation.
  A larger pool would benefit more from AL.

**What I would do differently:**
- Use a single coherent labeling scheme from the start (all human-annotated or all model-annotated).
- Add more AL iterations with a larger pool (500+ examples).
- Experiment with BERT-based fine-tuning instead of TF-IDF for higher F1.
- Add a validation set tracked across pipeline runs for longitudinal comparison.
