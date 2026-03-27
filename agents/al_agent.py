"""
ActiveLearningAgent — sample-efficient labeling via uncertainty sampling.

Skills
------
fit(labeled_df)                          → trains internal model
query(pool_df, strategy, n)              → np.ndarray of pool row indices
evaluate(test_df)                        → dict (accuracy, f1, f1_macro)
run_cycle(labeled_df, pool_df, test_df,
          strategy, n_iterations,
          batch_size)                    → list[dict] history
report(history_list)                     → path to learning_curve.png
explain_with_llm(history_list, task)     → str  [BONUS — requires ANTHROPIC_API_KEY]

Query strategies
----------------
'entropy'  : H(p) = -Σ p_i log p_i  (highest entropy = most uncertain)
'margin'   : smallest gap between top-2 probabilities
'random'   : random baseline

Model backends (``model`` parameter)
-------------------------------------
'logreg'   : TF-IDF + LogisticRegression  (default, fast)
'svm'      : TF-IDF + LinearSVC  (no probability → uses decision function)
'nb'       : TF-IDF + MultinomialNB
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class ActiveLearningAgent:
    """Active Learning agent for text classification.

    Parameters
    ----------
    model : str
        Base classifier: ``'logreg'``, ``'svm'``, ``'nb'``.
    text_col : str
        Column containing raw text.
    label_col : str
        Column containing class labels.
    random_state : int
        Seed for reproducibility.
    output_dir : str | Path
        Directory for all generated artefacts.
    dotenv_path : str | Path | None
        Path to .env file (loads ANTHROPIC_API_KEY for bonus skill).
    """

    def __init__(
        self,
        model: str = "logreg",
        text_col: str = "text",
        label_col: str = "label",
        random_state: int = 42,
        output_dir: str | Path = "data/al",
        dotenv_path: str | Path | None = ".env",
    ) -> None:
        self.model_name = model
        self.text_col = text_col
        self.label_col = label_col
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._pipeline: Pipeline | None = None
        self._classes: list[str] = []

        if dotenv_path and Path(str(dotenv_path)).exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(str(dotenv_path), override=False)
            except ImportError:
                pass

    # ------------------------------------------------------------------
    # Public skills
    # ------------------------------------------------------------------

    def fit(self, labeled_df: pd.DataFrame) -> None:
        """Train the internal pipeline on labeled data.

        Parameters
        ----------
        labeled_df : pd.DataFrame
            Must contain ``text_col`` and ``label_col``.
        """
        X = labeled_df[self.text_col].fillna("").tolist()
        y = labeled_df[self.label_col].tolist()
        self._classes = sorted(set(y))

        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, y)
        logger.debug("fit: %d examples, classes=%s", len(X), self._classes)

    def query(
        self,
        pool_df: pd.DataFrame,
        strategy: str = "entropy",
        n: int = 20,
    ) -> np.ndarray:
        """Select the ``n`` most informative examples from the pool.

        Parameters
        ----------
        pool_df : pd.DataFrame
            Unlabeled pool (same schema as training data, labels may be
            present but are not used).
        strategy : str
            ``'entropy'`` | ``'margin'`` | ``'random'``.
        n : int
            Number of examples to query.

        Returns
        -------
        np.ndarray — integer indices into pool_df (iloc positions).
        """
        if self._pipeline is None:
            raise RuntimeError("Call fit() before query().")

        n = min(n, len(pool_df))
        X_pool = pool_df[self.text_col].fillna("").tolist()

        if strategy == "random":
            rng = np.random.default_rng(self.random_state)
            return rng.choice(len(pool_df), size=n, replace=False)

        # Get probability estimates
        proba = self._predict_proba(X_pool)

        if strategy == "entropy":
            scores = self._entropy(proba)
        elif strategy == "margin":
            scores = self._margin(proba)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. "
                             "Choose 'entropy', 'margin', or 'random'.")

        # Return indices of top-n highest uncertainty
        return np.argsort(-scores)[:n]

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Evaluate the current model on a test set.

        Returns
        -------
        dict with keys: accuracy, f1_macro, f1_positive, f1_negative
        """
        if self._pipeline is None:
            raise RuntimeError("Call fit() before evaluate().")

        X_test = test_df[self.text_col].fillna("").tolist()
        y_test = test_df[self.label_col].tolist()
        y_pred = self._pipeline.predict(X_test)

        pos_label = "positive" if "positive" in self._classes else self._classes[0]

        return {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
            "f1_positive": round(
                float(f1_score(y_test, y_pred, average="binary",
                               pos_label=pos_label, zero_division=0)), 4
            ),
        }

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategy: str = "entropy",
        n_iterations: int = 5,
        batch_size: int = 20,
    ) -> list[dict]:
        """Run a full Active Learning cycle.

        Starting with ``labeled_df``, iteratively queries ``batch_size``
        examples from ``pool_df``, simulates annotation (moves them to
        labeled), retrains and evaluates.

        Parameters
        ----------
        labeled_df : pd.DataFrame
            Initial labeled set (e.g. 50 examples).
        pool_df : pd.DataFrame
            Unlabeled pool (labels present but treated as unknown to agent).
        test_df : pd.DataFrame
            Fixed test set — never used for training.
        strategy : str
            Query strategy.
        n_iterations : int
            Number of AL iterations.
        batch_size : int
            Examples queried per iteration.

        Returns
        -------
        list[dict] — history of metrics per iteration.
        """
        current_labeled = labeled_df.copy().reset_index(drop=True)
        current_pool = pool_df.copy().reset_index(drop=True)
        history: list[dict] = []

        # Iteration 0: performance of initial labeled set
        self.fit(current_labeled)
        m0 = self.evaluate(test_df)
        history.append({
            "iteration": 0,
            "n_labeled": len(current_labeled),
            "strategy": strategy,
            **m0,
        })
        logger.info(
            "AL [%s] iter 0/%d  n=%d  acc=%.3f  f1=%.3f",
            strategy, n_iterations, len(current_labeled),
            m0["accuracy"], m0["f1_macro"],
        )

        for i in range(1, n_iterations + 1):
            if len(current_pool) == 0:
                logger.warning("Pool exhausted at iteration %d.", i)
                break

            # Query
            query_idx = self.query(current_pool, strategy=strategy, n=batch_size)

            # Simulate annotation: move queried rows to labeled set
            new_examples = current_pool.iloc[query_idx].copy()
            current_labeled = pd.concat(
                [current_labeled, new_examples], ignore_index=True
            )
            current_pool = current_pool.drop(
                index=current_pool.index[query_idx]
            ).reset_index(drop=True)

            # Retrain on expanded labeled set
            self.fit(current_labeled)
            m = self.evaluate(test_df)

            history.append({
                "iteration": i,
                "n_labeled": len(current_labeled),
                "strategy": strategy,
                "query_indices": query_idx.tolist(),
                **m,
            })
            logger.info(
                "AL [%s] iter %d/%d  n=%d  acc=%.3f  f1=%.3f",
                strategy, i, n_iterations, len(current_labeled),
                m["accuracy"], m["f1_macro"],
            )

        return history

    def report(
        self,
        history_list: list[list[dict]] | list[dict],
        metric: str = "f1_macro",
        output_filename: str = "learning_curve.png",
        target_quality: float | None = None,
    ) -> str:
        """Plot learning curves for one or more AL runs.

        Parameters
        ----------
        history_list : list[dict] | list[list[dict]]
            A single history or a list of histories (one per strategy).
        metric : str
            Metric to plot: ``'accuracy'`` | ``'f1_macro'`` | ``'f1_positive'``.
        output_filename : str
            Filename for the saved PNG.
        target_quality : float | None
            Draw a horizontal dashed line at this quality level.

        Returns
        -------
        str — absolute path to the saved PNG.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # Normalise input: accept either a single history or a list of histories
        if history_list and isinstance(history_list[0], dict):
            histories = [history_list]  # wrap single history
        else:
            histories = history_list  # type: ignore

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = {"entropy": "#E53935", "margin": "#1E88E5", "random": "#43A047",
                  "unknown": "#9E9E9E"}
        styles = {"entropy": "-o", "margin": "-s", "random": "--^"}

        # --- Plot 1: learning curves ---
        ax = axes[0]
        for hist in histories:
            strategy = hist[0].get("strategy", "unknown")
            xs = [h["n_labeled"] for h in hist]
            ys = [h[metric] for h in hist]
            color = colors.get(strategy, "#9E9E9E")
            style = styles.get(strategy, "-x")
            ax.plot(xs, ys, style, color=color, lw=2, ms=7, label=strategy)

        if target_quality is not None:
            ax.axhline(target_quality, color="gray", ls=":", lw=1.5,
                       label=f"target={target_quality}")

        ax.set_title(f"Learning curves — {metric}", fontweight="bold")
        ax.set_xlabel("Number of labeled examples")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend()
        ax.grid(alpha=0.4)
        ax.set_ylim(bottom=max(0, min(
            h[metric] for hist in histories for h in hist) - 0.05
        ))

        # --- Plot 2: sample efficiency (area under curve relative to random) ---
        ax2 = axes[1]
        random_hist = next(
            (h for h in histories if h[0].get("strategy") == "random"), None
        )
        for hist in histories:
            strategy = hist[0].get("strategy", "unknown")
            xs = np.array([h["n_labeled"] for h in hist])
            ys = np.array([h[metric] for h in hist])

            if random_hist and strategy != "random":
                rx = np.array([h["n_labeled"] for h in random_hist])
                ry = np.array([h[metric] for h in random_hist])
                # Interpolate random on same x grid
                ry_interp = np.interp(xs, rx, ry)
                gain = ys - ry_interp
                color = colors.get(strategy, "#9E9E9E")
                ax2.bar(xs, gain, width=3, color=color, alpha=0.7, label=strategy)

        ax2.axhline(0, color="black", lw=1)
        ax2.set_title("Quality gain over random baseline", fontweight="bold")
        ax2.set_xlabel("Number of labeled examples")
        ax2.set_ylabel(f"Δ {metric} vs random")
        ax2.legend()
        ax2.grid(alpha=0.4)

        plt.tight_layout()
        out = self.output_dir / output_filename
        plt.savefig(out, bbox_inches="tight", dpi=130)
        plt.close(fig)
        logger.info("Learning curve saved → %s", out)
        return str(out)

    def explain_with_llm(
        self,
        history_list: list[list[dict]],
        task_description: str = "binary sentiment classification",
    ) -> str:
        """Use Claude API to analyse the learning curves and recommend stopping.

        Requires ANTHROPIC_API_KEY in environment / .env.
        """
        try:
            import anthropic
        except ImportError:
            return "❌ anthropic not installed.  pip install anthropic"

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "❌ ANTHROPIC_API_KEY not set in .env"

        # Build table
        rows = []
        for hist in history_list:
            for h in hist:
                rows.append({
                    "strategy": h.get("strategy", "?"),
                    "n_labeled": h["n_labeled"],
                    "accuracy": h.get("accuracy", "?"),
                    "f1_macro": h.get("f1_macro", "?"),
                })
        df = pd.DataFrame(rows)
        table_md = df.to_markdown(index=False)

        strategies = list({h.get("strategy") for hist in history_list for h in hist})

        prompt = f"""You are an expert in Active Learning for NLP.

**Task:** {task_description}

**Learning curve data:**
{table_md}

**Strategies compared:** {', '.join(strategies)}

Answer the following questions concisely:
1. Which strategy is most **sample-efficient** and why?
2. At what `n_labeled` does the model quality **plateau** (diminishing returns)?
3. How many **labeling examples are saved** by the best strategy compared to random baseline to reach the same quality?
4. Should we **continue labeling** beyond the final point, or is it sufficient for production?

Give a short, practical recommendation in bullet points."""

        try:
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=700,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return f"❌ LLM failed: {exc}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> Pipeline:
        """Build the sklearn Pipeline for the selected model."""
        vectorizer = TfidfVectorizer(
            max_features=15_000,
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
            strip_accents="unicode",
        )

        if self.model_name == "logreg":
            clf = LogisticRegression(
                C=1.0, max_iter=1000, random_state=self.random_state, n_jobs=-1
            )
        elif self.model_name == "svm":
            clf = LinearSVC(C=1.0, max_iter=2000, random_state=self.random_state)
        elif self.model_name == "nb":
            vectorizer = TfidfVectorizer(
                max_features=15_000, sublinear_tf=False,
                ngram_range=(1, 2), min_df=2
            )
            clf = MultinomialNB(alpha=0.1)
        else:
            raise ValueError(f"Unknown model: {self.model_name!r}")

        return Pipeline([("tfidf", vectorizer), ("clf", clf)])

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return probability matrix [n_samples × n_classes]."""
        if hasattr(self._pipeline["clf"], "predict_proba"):
            return self._pipeline.predict_proba(texts)
        # SVM: use decision function → softmax approximation
        scores = self._pipeline.decision_function(texts)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        # Soft-max
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    @staticmethod
    def _entropy(proba: np.ndarray) -> np.ndarray:
        p = np.clip(proba, 1e-9, 1.0)
        return -np.sum(p * np.log(p), axis=1)

    @staticmethod
    def _margin(proba: np.ndarray) -> np.ndarray:
        sorted_p = np.sort(proba, axis=1)[:, ::-1]
        return -(sorted_p[:, 0] - sorted_p[:, 1])  # negate → higher = more uncertain
