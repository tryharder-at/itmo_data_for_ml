"""
CLI entry point for the ActiveLearningAgent.

Usage:
    python run_al.py
    python run_al.py --model logreg --n-initial 50 --n-iter 5 --batch 20
    python run_al.py --strategy entropy  # single strategy only
    python run_al.py --all-strategies    # compare entropy + margin + random
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from agents.al_agent import ActiveLearningAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ActiveLearningAgent")
    parser.add_argument("--input", default="data/clean/cleaned_best.csv",
                        help="Path to cleaned dataset CSV")
    parser.add_argument("--model", default="logreg",
                        choices=["logreg", "svm", "nb"],
                        help="Base classifier backend")
    parser.add_argument("--strategy", default="entropy",
                        choices=["entropy", "margin", "random"],
                        help="Query strategy (ignored if --all-strategies)")
    parser.add_argument("--all-strategies", action="store_true",
                        help="Run all 3 strategies and compare")
    parser.add_argument("--n-initial", type=int, default=50,
                        help="Initial labeled set size")
    parser.add_argument("--n-test", type=int, default=200,
                        help="Test set size")
    parser.add_argument("--n-iter", type=int, default=5,
                        help="Number of AL iterations")
    parser.add_argument("--batch", type=int, default=20,
                        help="Examples queried per iteration")
    parser.add_argument("--metric", default="f1_macro",
                        choices=["accuracy", "f1_macro", "f1_positive"],
                        help="Metric to plot in learning curve")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="data/al",
                        help="Directory for output artefacts")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    df = pd.read_csv(args.input)
    print(f"Loaded: {len(df)} rows from {args.input}")

    df_trainpool, df_test = train_test_split(
        df, test_size=args.n_test, stratify=df["label"],
        random_state=args.random_state,
    )
    df_labeled_init, df_pool = train_test_split(
        df_trainpool, train_size=args.n_initial, stratify=df_trainpool["label"],
        random_state=args.random_state,
    )

    print(f"  Initial labeled : {len(df_labeled_init)}")
    print(f"  Unlabeled pool  : {len(df_pool)}")
    print(f"  Test set        : {len(df_test)}")

    # ------------------------------------------------------------------ #
    # 2. Instantiate agent
    # ------------------------------------------------------------------ #
    agent = ActiveLearningAgent(
        model=args.model,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )

    # ------------------------------------------------------------------ #
    # 3. Run AL cycle(s)
    # ------------------------------------------------------------------ #
    strategies = ["entropy", "margin", "random"] if args.all_strategies \
        else [args.strategy]

    histories = []
    for strategy in strategies:
        print(f"\n[AL] Running strategy: {strategy} ...")
        hist = agent.run_cycle(
            labeled_df=df_labeled_init,
            pool_df=df_pool,
            test_df=df_test,
            strategy=strategy,
            n_iterations=args.n_iter,
            batch_size=args.batch,
        )
        histories.append(hist)
        last = hist[-1]
        print(f"  Final   n_labeled={last['n_labeled']}  "
              f"acc={last['accuracy']:.4f}  f1={last['f1_macro']:.4f}")

    # ------------------------------------------------------------------ #
    # 4. Report
    # ------------------------------------------------------------------ #
    print("\n[Report] Generating learning curve plot ...")
    png = agent.report(
        history_list=histories,
        metric=args.metric,
        output_filename="learning_curve.png",
    )
    print(f"  → {png}")

    # ------------------------------------------------------------------ #
    # 5. Sample efficiency summary
    # ------------------------------------------------------------------ #
    if args.all_strategies:
        rand_hist = next(h for h in histories if h[0]["strategy"] == "random")
        import numpy as np
        rx = [h["n_labeled"] for h in rand_hist]
        ry = [h["f1_macro"] for h in rand_hist]
        target_f1 = ry[-1]

        print(f"\n[Efficiency] Target F1 = {target_f1:.4f}  "
              f"(random @ {rx[-1]} examples)")
        for hist in histories:
            strategy = hist[0]["strategy"]
            if strategy == "random":
                continue
            xs = [h["n_labeled"] for h in hist]
            ys = [h["f1_macro"] for h in hist]
            above = [i for i, y in enumerate(ys) if y >= target_f1]
            if above:
                n_needed = xs[above[0]]
                saved = rx[-1] - n_needed
                print(f"  {strategy:<8}: reaches target at {n_needed} examples "
                      f"→ saves {saved} labels ({saved/rx[-1]*100:.0f}% fewer)")
            else:
                print(f"  {strategy:<8}: did NOT reach target within {xs[-1]} examples")

    # ------------------------------------------------------------------ #
    # 6. BONUS: LLM explanation
    # ------------------------------------------------------------------ #
    print("\n[BONUS] Requesting LLM explanation ...")
    explanation = agent.explain_with_llm(
        history_list=histories,
        task_description="Binary sentiment classification (positive/negative) "
                         "on mixed-source text data.",
    )
    print(explanation)

    print("\nDone. All outputs in", args.output_dir)


if __name__ == "__main__":
    main()
