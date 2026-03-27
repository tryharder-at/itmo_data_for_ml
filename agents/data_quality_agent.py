"""
DataQualityAgent — detects and fixes data quality issues in text datasets.

Skills
------
detect_issues(df)               → QualityReport
fix(df, strategy)               → pd.DataFrame
compare(df_before, df_after)    → pd.DataFrame (comparison table)
explain_with_llm(report, task)  → str  [BONUS — requires ANTHROPIC_API_KEY in .env]

Detected issue types
--------------------
1. Missing values   (NaN in any column)
2. Duplicates       (exact duplicate rows)
3. Outliers         (text length via IQR or z-score)
4. Class imbalance  (unequal label distribution)

Fix strategies (per issue type)
--------------------------------
missing    : 'drop'         | 'fill'
duplicates : 'drop'         | 'keep'
outliers   : 'drop_iqr'     | 'drop_zscore' | 'clip_iqr' | 'keep'
imbalance  : 'undersample'  | 'oversample'  | 'none'
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MissingReport:
    counts: dict        # {col: n_missing}
    percentages: dict   # {col: pct_missing}
    total_affected_rows: int


@dataclass
class DuplicateReport:
    count: int
    percentage: float
    sample_indices: list   # up to 20 indices


@dataclass
class OutlierReport:
    column: str           # derived metric used (e.g. 'word_count')
    method: str           # 'iqr' or 'zscore'
    count: int
    percentage: float
    lower_bound: float
    upper_bound: float
    outlier_indices: list  # up to 100 indices


@dataclass
class ImbalanceReport:
    label_column: str
    class_counts: dict       # {label: count}
    class_percentages: dict  # {label: pct}
    majority_class: str
    minority_class: str
    imbalance_ratio: float   # majority / minority
    is_imbalanced: bool      # ratio > imbalance_threshold


@dataclass
class QualityReport:
    total_rows: int
    total_cols: int
    missing: MissingReport
    duplicates: DuplicateReport
    outliers: OutlierReport
    imbalance: ImbalanceReport

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        nr = self.total_rows
        lines = [
            f"{'='*52}",
            f"  Quality Report  ({nr:,} rows × {self.total_cols} cols)",
            f"{'='*52}",
            f"  Missing values  : {self.missing.total_affected_rows} rows  "
            f"({self.missing.total_affected_rows / nr * 100:.1f}%)",
            f"  Duplicates      : {self.duplicates.count} rows  "
            f"({self.duplicates.percentage:.1f}%)",
            f"  Outliers        : {self.outliers.count} rows  "
            f"({self.outliers.percentage:.1f}%)  [method={self.outliers.method}, "
            f"bounds=[{self.outliers.lower_bound:.0f}, {self.outliers.upper_bound:.0f}] words]",
            f"  Imbalance ratio : {self.imbalance.imbalance_ratio:.2f}×  "
            f"({self.imbalance.majority_class} / {self.imbalance.minority_class})"
            f"  ⚠ imbalanced={self.imbalance.is_imbalanced}",
            f"{'='*52}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DataQualityAgent:
    """Detect and fix data quality issues in text datasets.

    Parameters
    ----------
    label_col : str
        Column containing class labels.
    text_col : str
        Column containing raw text.
    outlier_method : str
        Default method: ``'iqr'`` or ``'zscore'``.
    zscore_threshold : float
        |z| threshold for z-score outlier detection.
    iqr_factor : float
        IQR fence multiplier (standard = 1.5).
    imbalance_threshold : float
        majority/minority ratio above which imbalance is flagged.
    dotenv_path : str | Path | None
        Path to .env file for loading ANTHROPIC_API_KEY (bonus skill).
    """

    def __init__(
        self,
        label_col: str = "label",
        text_col: str = "text",
        outlier_method: str = "iqr",
        zscore_threshold: float = 3.0,
        iqr_factor: float = 1.5,
        imbalance_threshold: float = 1.5,
        dotenv_path: str | Path | None = ".env",
    ) -> None:
        self.label_col = label_col
        self.text_col = text_col
        self.outlier_method = outlier_method
        self.zscore_threshold = zscore_threshold
        self.iqr_factor = iqr_factor
        self.imbalance_threshold = imbalance_threshold

        # Load .env if available (for ANTHROPIC_API_KEY)
        if dotenv_path and Path(dotenv_path).exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path, override=False)
                logger.info("Loaded .env from %s", dotenv_path)
            except ImportError:
                logger.debug("python-dotenv not installed — .env not loaded")

    # ------------------------------------------------------------------
    # Public skills
    # ------------------------------------------------------------------

    def detect_issues(self, df: pd.DataFrame) -> QualityReport:
        """Scan DataFrame and return a structured QualityReport.

        Detects:
        1. Missing values per column
        2. Duplicate rows
        3. Text-length outliers (IQR or z-score on word_count)
        4. Class imbalance ratio

        Returns
        -------
        QualityReport dataclass
        """
        logger.info("Detecting issues (%d rows) ...", len(df))
        report = QualityReport(
            total_rows=len(df),
            total_cols=len(df.columns),
            missing=self._detect_missing(df),
            duplicates=self._detect_duplicates(df),
            outliers=self._detect_outliers(df),
            imbalance=self._detect_imbalance(df),
        )
        logger.info(report.summary())
        return report

    def fix(
        self,
        df: pd.DataFrame,
        strategy: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Apply cleaning strategies and return a clean DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
        strategy : dict, optional
            Override any subset of defaults:
              'missing'    : 'drop'        | 'fill'
              'duplicates' : 'drop'        | 'keep'
              'outliers'   : 'drop_iqr'    | 'drop_zscore' | 'clip_iqr' | 'keep'
              'imbalance'  : 'undersample' | 'oversample'  | 'none'

        Returns
        -------
        pd.DataFrame — cleaned copy
        """
        defaults: dict[str, str] = {
            "missing": "drop",
            "duplicates": "drop",
            "outliers": "drop_iqr",
            "imbalance": "none",
        }
        s = {**defaults, **(strategy or {})}

        result = df.copy()
        result = self._fix_missing(result, s["missing"])
        result = self._fix_duplicates(result, s["duplicates"])
        result = self._fix_outliers(result, s["outliers"])
        result = self._fix_imbalance(result, s["imbalance"])
        result = result.reset_index(drop=True)

        logger.info(
            "Cleaned: %d → %d rows (removed %d)",
            len(df), len(result), len(df) - len(result),
        )
        return result

    def compare(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        strategy_label: str = "",
    ) -> pd.DataFrame:
        """Return a before/after comparison table as a DataFrame.

        Columns: metric, before, after, change, change_pct
        """
        r_b = self.detect_issues(df_before)
        r_a = self.detect_issues(df_after)

        def wc(df: pd.DataFrame) -> pd.Series:
            return df[self.text_col].dropna().str.split().str.len()

        wc_b, wc_a = wc(df_before), wc(df_after)

        rows: list[dict] = []

        def add(metric: str, before: float, after: float) -> None:
            diff = after - before
            pct = (diff / before * 100) if before != 0 else 0.0
            rows.append({
                "metric": metric,
                "before": _fmt(before),
                "after": _fmt(after),
                "change": _fmt(diff),
                "change_pct": f"{pct:+.1f}%",
            })

        add("total_rows",         r_b.total_rows,  r_a.total_rows)
        add("missing_rows",       r_b.missing.total_affected_rows, r_a.missing.total_affected_rows)
        add("missing_pct",
            r_b.missing.total_affected_rows / r_b.total_rows * 100,
            r_a.missing.total_affected_rows / r_a.total_rows * 100 if r_a.total_rows else 0)
        add("duplicate_rows",     r_b.duplicates.count, r_a.duplicates.count)
        add("duplicate_pct",      r_b.duplicates.percentage, r_a.duplicates.percentage)
        add("outlier_rows",       r_b.outliers.count, r_a.outliers.count)
        add("outlier_pct",        r_b.outliers.percentage, r_a.outliers.percentage)
        add("imbalance_ratio",    r_b.imbalance.imbalance_ratio, r_a.imbalance.imbalance_ratio)

        all_labels = sorted(
            set(r_b.imbalance.class_counts) | set(r_a.imbalance.class_counts)
        )
        for lbl in all_labels:
            add(f"count_{lbl}",
                r_b.imbalance.class_counts.get(lbl, 0),
                r_a.imbalance.class_counts.get(lbl, 0))

        add("avg_words",    wc_b.mean(),   wc_a.mean())
        add("median_words", wc_b.median(), wc_a.median())
        add("std_words",    wc_b.std(),    wc_a.std())
        add("max_words",    wc_b.max(),    wc_a.max())

        result = pd.DataFrame(rows)
        if strategy_label:
            result.insert(0, "strategy", strategy_label)
        return result

    def explain_with_llm(
        self,
        report: QualityReport,
        task_description: str = "binary sentiment classification (positive / negative)",
    ) -> str:
        """Use Claude API to explain detected issues and recommend a cleaning strategy.

        Requires ANTHROPIC_API_KEY set in .env or environment.

        Parameters
        ----------
        report : QualityReport
            Output of detect_issues().
        task_description : str
            Brief description of the ML task.

        Returns
        -------
        str — Claude's analysis
        """
        try:
            import anthropic
        except ImportError:
            return "❌ anthropic package not installed.  pip install anthropic"

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return (
                "❌ ANTHROPIC_API_KEY not found. "
                "Create a .env file with ANTHROPIC_API_KEY=sk-ant-..."
            )

        nr = report.total_rows

        prompt = f"""You are a senior ML data scientist. Analyse the following data quality report
for a dataset used in **{task_description}** and give actionable recommendations.

## Dataset
- Rows: {nr:,}
- Columns: {report.total_cols}

## Quality issues found

### 1. Missing values
{_dict_to_md(report.missing.counts) or '  None'}
Total affected rows: {report.missing.total_affected_rows} ({report.missing.total_affected_rows/nr*100:.1f}%)

### 2. Duplicates
{report.duplicates.count} rows ({report.duplicates.percentage:.1f}%)

### 3. Text-length outliers  (method: {report.outliers.method})
- Count: {report.outliers.count} rows ({report.outliers.percentage:.1f}%)
- IQR bounds: [{report.outliers.lower_bound:.0f}, {report.outliers.upper_bound:.0f}] words

### 4. Class imbalance
{_dict_to_md(report.imbalance.class_counts)}
Ratio: {report.imbalance.imbalance_ratio:.2f}× ({report.imbalance.majority_class} vs {report.imbalance.minority_class})

## Available strategies
| Issue | Options |
|-------|---------|
| missing | drop, fill |
| duplicates | drop, keep |
| outliers | drop_iqr, drop_zscore, clip_iqr, keep |
| imbalance | undersample, oversample, none |

## Your task
1. Briefly explain why each issue type is problematic for {task_description}.
2. Recommend the best strategy for each issue and explain why.
3. In 2-3 sentences, justify your overall recommended strategy dict:
   {{'missing': '?', 'duplicates': '?', 'outliers': '?', 'imbalance': '?'}}

Be concise and practical."""

        try:
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=900,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return f"❌ LLM call failed: {exc}"

    # ------------------------------------------------------------------
    # Private — detection
    # ------------------------------------------------------------------

    def _detect_missing(self, df: pd.DataFrame) -> MissingReport:
        counts: dict[str, int] = {}
        pcts: dict[str, float] = {}
        for col in df.columns:
            n = int(df[col].isna().sum())
            if n > 0:
                counts[col] = n
                pcts[col] = round(n / len(df) * 100, 2)
        affected = int(df.isna().any(axis=1).sum())
        return MissingReport(counts=counts, percentages=pcts, total_affected_rows=affected)

    def _detect_duplicates(self, df: pd.DataFrame) -> DuplicateReport:
        mask = df.duplicated(keep="first")
        count = int(mask.sum())
        pct = round(count / len(df) * 100, 2)
        indices = df.index[mask].tolist()[:20]
        return DuplicateReport(count=count, percentage=pct, sample_indices=indices)

    def _detect_outliers(
        self, df: pd.DataFrame, method: str | None = None
    ) -> OutlierReport:
        method = method or self.outlier_method
        wc = df[self.text_col].dropna().str.split().str.len()

        if method == "iqr":
            q1, q3 = wc.quantile(0.25), wc.quantile(0.75)
            iqr = q3 - q1
            lower = float(max(0.0, q1 - self.iqr_factor * iqr))
            upper = float(q3 + self.iqr_factor * iqr)
            mask = (wc < lower) | (wc > upper)
        else:  # zscore
            mean, std = wc.mean(), wc.std()
            lower = float(mean - self.zscore_threshold * std)
            upper = float(mean + self.zscore_threshold * std)
            mask = ((wc - mean) / std).abs() > self.zscore_threshold

        indices = wc[mask].index.tolist()
        count = len(indices)
        return OutlierReport(
            column="word_count",
            method=method,
            count=count,
            percentage=round(count / len(df) * 100, 2),
            lower_bound=round(lower, 1),
            upper_bound=round(upper, 1),
            outlier_indices=indices[:100],
        )

    def _detect_imbalance(self, df: pd.DataFrame) -> ImbalanceReport:
        if self.label_col not in df.columns:
            return ImbalanceReport(
                label_column=self.label_col,
                class_counts={}, class_percentages={},
                majority_class="N/A", minority_class="N/A",
                imbalance_ratio=1.0, is_imbalanced=False,
            )
        counts = df[self.label_col].value_counts().to_dict()
        total = sum(counts.values())
        pcts = {k: round(v / total * 100, 2) for k, v in counts.items()}
        majority = max(counts, key=counts.get)
        minority = min(counts, key=counts.get)
        ratio = round(counts[majority] / max(counts[minority], 1), 3)
        return ImbalanceReport(
            label_column=self.label_col,
            class_counts=counts,
            class_percentages=pcts,
            majority_class=majority,
            minority_class=minority,
            imbalance_ratio=ratio,
            is_imbalanced=ratio > self.imbalance_threshold,
        )

    # ------------------------------------------------------------------
    # Private — fixing
    # ------------------------------------------------------------------

    def _fix_missing(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        if strategy == "drop":
            before = len(df)
            df = df.dropna(subset=[self.text_col])
            logger.info("  missing → drop: removed %d rows", before - len(df))
        elif strategy == "fill":
            n = int(df[self.text_col].isna().sum())
            df = df.copy()
            df[self.text_col] = df[self.text_col].fillna("[MISSING]")
            logger.info("  missing → fill: replaced %d values with [MISSING]", n)
        return df

    def _fix_duplicates(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        if strategy == "drop":
            before = len(df)
            df = df.drop_duplicates(keep="first")
            logger.info("  duplicates → drop: removed %d rows", before - len(df))
        elif strategy == "keep":
            logger.info("  duplicates → keep: no action")
        return df

    def _fix_outliers(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        if strategy == "keep":
            logger.info("  outliers → keep: no action")
            return df

        wc = df[self.text_col].dropna().str.split().str.len()
        q1, q3 = wc.quantile(0.25), wc.quantile(0.75)
        iqr = q3 - q1
        iqr_lower = float(max(0.0, q1 - self.iqr_factor * iqr))
        iqr_upper = float(q3 + self.iqr_factor * iqr)

        if strategy == "drop_iqr":
            mask = (wc < iqr_lower) | (wc > iqr_upper)
            before = len(df)
            df = df.drop(index=wc[mask].index)
            logger.info("  outliers → drop_iqr: removed %d rows (bounds=[%.0f, %.0f])",
                        before - len(df), iqr_lower, iqr_upper)

        elif strategy == "drop_zscore":
            mean, std = wc.mean(), wc.std()
            z = (wc - mean) / std
            mask = z.abs() > self.zscore_threshold
            z_lower = mean - self.zscore_threshold * std
            z_upper = mean + self.zscore_threshold * std
            before = len(df)
            df = df.drop(index=wc[mask].index)
            logger.info("  outliers → drop_zscore: removed %d rows (bounds=[%.0f, %.0f])",
                        before - len(df), z_lower, z_upper)

        elif strategy == "clip_iqr":
            max_words = int(iqr_upper)

            def _truncate(text: Any) -> Any:
                if pd.isna(text):
                    return text
                words = str(text).split()
                return " ".join(words[:max_words]) if len(words) > max_words else text

            long_count = int((wc > iqr_upper).sum())
            df = df.copy()
            df[self.text_col] = df[self.text_col].apply(_truncate)
            logger.info("  outliers → clip_iqr: truncated %d texts to %d words",
                        long_count, max_words)

        return df

    def _fix_imbalance(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        if strategy == "none" or self.label_col not in df.columns:
            return df

        counts = df[self.label_col].value_counts()
        minority_n = int(counts.min())
        majority_n = int(counts.max())

        if strategy == "undersample":
            parts = [
                grp.sample(n=minority_n, random_state=42)
                for _, grp in df.groupby(self.label_col)
            ]
            df = pd.concat(parts)
            logger.info("  imbalance → undersample: %d → %d per class",
                        majority_n, minority_n)

        elif strategy == "oversample":
            parts = [
                grp.sample(n=majority_n, replace=True, random_state=42)
                if len(grp) < majority_n else grp
                for _, grp in df.groupby(self.label_col)
            ]
            df = pd.concat(parts)
            logger.info("  imbalance → oversample: %d → %d per class",
                        minority_n, majority_n)

        return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fmt(v: float) -> str | float:
    """Format number: int if whole, else 2 decimals."""
    if isinstance(v, float) and v == int(v):
        return int(v)
    return round(v, 2) if isinstance(v, float) else v


def _dict_to_md(d: dict) -> str:
    if not d:
        return ""
    return "\n".join(f"  - {k}: {v}" for k, v in d.items())
