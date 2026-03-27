"""Run AL cycle, save model + report + learning curve."""
import argparse, json, sys, datetime
from pathlib import Path

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from agents.al_agent import ActiveLearningAgent

parser = argparse.ArgumentParser()
parser.add_argument("--n-initial",  type=int, default=50)
parser.add_argument("--n-test",     type=int, default=100)
parser.add_argument("--n-iter",     type=int, default=5)
parser.add_argument("--batch-size", type=int, default=20)
args = parser.parse_args()

df = pd.read_csv(ROOT / "data" / "labeled" / "final_dataset.csv")
if "final_label" in df.columns:
    df = df.drop(columns=["label"], errors="ignore")
    df = df.rename(columns={"final_label": "label"})

n_test = min(args.n_test, max(40, len(df) // 5))
df_trainpool, df_test = train_test_split(
    df[["text", "label"]], test_size=n_test,
    stratify=df["label"] if "label" in df.columns else None, random_state=42,
)
n_init = min(args.n_initial, len(df_trainpool) - 10)
df_init, df_pool = train_test_split(
    df_trainpool, train_size=n_init,
    stratify=df_trainpool["label"], random_state=42,
)

agent = ActiveLearningAgent(
    model="logreg", text_col="text", label_col="label",
    random_state=42, output_dir=str(ROOT / "data" / "al"),
)

hist_entropy = agent.run_cycle(
    labeled_df=df_init, pool_df=df_pool, test_df=df_test,
    strategy="entropy", n_iterations=args.n_iter, batch_size=args.batch_size,
)
hist_random = agent.run_cycle(
    labeled_df=df_init, pool_df=df_pool, test_df=df_test,
    strategy="random", n_iterations=args.n_iter, batch_size=args.batch_size,
)

agent.report(
    history_list=[hist_entropy, hist_random],
    metric="f1_macro",
    output_filename="pipeline_learning_curve.png",
)

# Save model
(ROOT / "models").mkdir(parents=True, exist_ok=True)
joblib.dump(agent._pipeline, ROOT / "models" / "sentiment_model.joblib")

final = hist_entropy[-1]
llm_text = agent.explain_with_llm(
    history_list=[hist_entropy, hist_random],
    task_description="Binary sentiment classification on mixed-source text data.",
)

# AL report
hist_df = pd.DataFrame(hist_entropy)[["iteration", "n_labeled", "accuracy", "f1_macro"]]
try:
    table_md = hist_df.to_markdown(index=False)
except Exception:
    table_md = hist_df.to_string(index=False)

(ROOT / "reports").mkdir(parents=True, exist_ok=True)
report_md = f"""# Active Learning Report

Generated: {datetime.datetime.now().isoformat(timespec='seconds')}

## Protocol
- Initial: {n_init} | Pool: {len(df_pool)} | Test: {len(df_test)}
- Iterations: {args.n_iter} × {args.batch_size} examples

## Entropy Strategy Results
{table_md}

## Final Metrics
| Metric | Value |
|--------|-------|
| Accuracy | {final["accuracy"]} |
| F1-macro | {final["f1_macro"]} |
| F1-positive | {final["f1_positive"]} |

## LLM Analysis (Claude)
{llm_text}
"""
(ROOT / "reports" / "al_report.md").write_text(report_md, encoding="utf-8")

print(json.dumps({
    "iterations": hist_entropy,
    "final_metrics": {
        "accuracy": final["accuracy"],
        "f1_macro": final["f1_macro"],
        "f1_positive": final["f1_positive"],
    },
    "model": "models/sentiment_model.joblib",
    "curve": "data/al/pipeline_learning_curve.png",
    "report": "reports/al_report.md",
}, default=str))
