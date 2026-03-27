# Skill: Master Pipeline

Orchestrate all 4 sub-skills in sequence to run the complete ML pipeline
from raw data collection to a trained sentiment model.

Sub-skills executed in order:
1. `/collect` — DataCollectionAgent
2. `/clean` — DataQualityAgent
3. `/annotate` — AnnotationAgent + HITL review
4. `/al` — ActiveLearningAgent + model training

---

## Steps

### 0. Pipeline introduction

Tell the user:
```
Starting the unified ML pipeline.

Steps:
  1/4  Data Collection   → data/raw/unified_dataset.csv
  2/4  Data Cleaning     → data/clean/pipeline_clean.csv
  3/4  Annotation + HITL → data/labeled/final_dataset.csv  ← human checkpoint
  4/4  Active Learning   → models/sentiment_model.joblib

There will be one human-in-the-loop checkpoint in Step 3 (annotation review).
All other steps run automatically.
```

### 1. Execute /collect skill

Use the Skill tool to invoke the `collect` skill.
Wait for it to complete before proceeding.

### 2. Execute /clean skill

Use the Skill tool to invoke the `clean` skill.
Wait for it to complete before proceeding.

### 3. Execute /annotate skill

Use the Skill tool to invoke the `annotate` skill.

This step contains the **HITL checkpoint** — the annotate skill will pause
and ask the user to review labels in the Streamlit app. Wait for the user
to confirm they are done before the annotate skill proceeds.

### 4. Execute /al skill

Use the Skill tool to invoke the `al` skill.
Wait for it to complete before proceeding.

### 5. Final summary

After all 4 skills complete, print the pipeline summary:

```
╔══════════════════════════════════════════════════════╗
║           PIPELINE COMPLETE                          ║
╠══════════════════════════════════════════════════════╣
║  Step 1  Data collected   → data/raw/               ║
║  Step 2  Data cleaned     → data/clean/             ║
║  Step 3  Data annotated   → data/labeled/           ║
║  Step 4  Model trained    → models/                 ║
╠══════════════════════════════════════════════════════╣
║  Final accuracy  : X.XXXX                           ║
║  Final F1-macro  : X.XXXX                           ║
╠══════════════════════════════════════════════════════╣
║  Reports: reports/quality_report.md                 ║
║           reports/annotation_report.md              ║
║           reports/al_report.md                      ║
╚══════════════════════════════════════════════════════╝
```

Fill in the actual final metrics from the /al skill output.

---

## Error handling

If any sub-skill fails:
- Report which step failed and the error message
- Ask the user: "Retry this step, skip it, or abort the pipeline?"
  - `retry` → re-invoke the same skill
  - `skip` → continue to next step (warn that downstream steps may fail)
  - `abort` → stop, print what was completed so far

## Constraints
- Do NOT ask for confirmation between steps (only the annotate HITL checkpoint is a pause)
- Do NOT ask about model choice, confidence thresholds, or AL parameters
- Keep the human interaction minimal — one real checkpoint in step 3
