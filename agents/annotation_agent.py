"""
AnnotationAgent — automatic labeling, spec generation, quality metrics,
LabelStudio export, and human-in-the-loop flagging.

Skills
------
auto_label(df, modality)        → pd.DataFrame  (adds predicted_label + confidence)
generate_spec(df, task)         → str  (writes annotation_spec.md, returns path)
check_quality(df_labeled)       → dict  (Cohen's κ, label dist, confidence stats)
export_to_labelstudio(df)       → dict  (writes labelstudio_import.json, returns data)
flag_low_confidence(df_labeled) → pd.DataFrame  [BONUS HITL]

Supported modalities
--------------------
text  : DistilBERT-SST2 (primary) → VADER fallback → keyword fallback
audio : Whisper stub (not auto-downloaded; raises NotImplementedError if called alone)
image : YOLO stub (same)
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections import Counter
from datetime import datetime
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
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TEXT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
# Maps the model's raw output labels → our schema labels
_SST2_LABEL_MAP = {"NEGATIVE": "negative", "POSITIVE": "positive",
                   "LABEL_0": "negative", "LABEL_1": "positive"}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AnnotationAgent:
    """Automatic annotation agent for text, audio, and image datasets.

    Parameters
    ----------
    modality : str
        Default modality: ``'text'`` | ``'audio'`` | ``'image'``.
    labels : list[str] | None
        Target label set. Defaults to ``['positive', 'negative']``.
    text_model : str
        HuggingFace model name for text classification.
    zero_shot : bool
        If True, use zero-shot classification instead of the fine-tuned model.
        Requires the ``candidate_labels`` to be identical to ``labels``.
    batch_size : int
        Inference batch size (for the neural pipeline).
    confidence_threshold : float
        HITL threshold — examples below this are flagged for human review.
    output_dir : str | Path
        Directory for all generated artefacts.
    """

    def __init__(
        self,
        modality: str = "text",
        labels: list[str] | None = None,
        text_model: str = DEFAULT_TEXT_MODEL,
        zero_shot: bool = False,
        batch_size: int = 32,
        confidence_threshold: float = 0.70,
        output_dir: str | Path = "data/annotations",
    ) -> None:
        self.modality = modality
        self.labels = labels or ["positive", "negative"]
        self.text_model = text_model
        self.zero_shot = zero_shot
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._pipeline: Any = None   # lazy-loaded neural pipeline
        self._backend: str = "unloaded"

    # ------------------------------------------------------------------
    # Public skills
    # ------------------------------------------------------------------

    def auto_label(
        self,
        df: pd.DataFrame,
        modality: str | None = None,
        text_col: str = "text",
    ) -> pd.DataFrame:
        """Automatically label every row and attach a confidence score.

        Adds columns:
          - ``predicted_label`` : predicted class
          - ``confidence``      : model's confidence for the predicted class

        Parameters
        ----------
        df : pd.DataFrame
        modality : str | None
            Override the agent's default modality.
        text_col : str
            Column containing text to classify.

        Returns
        -------
        pd.DataFrame — copy with added columns, NaN rows set to
        ``predicted_label='unknown', confidence=0.0``.
        """
        m = modality or self.modality
        logger.info("auto_label: modality=%s, rows=%d", m, len(df))

        result = df.copy()

        if m == "text":
            result = self._label_text(result, text_col)
        elif m == "audio":
            result = self._label_audio_stub(result)
        elif m == "image":
            result = self._label_image_stub(result)
        else:
            raise ValueError(f"Unknown modality: {m!r}")

        # Fill any remaining NaN predictions
        result["predicted_label"] = result.get(
            "predicted_label", pd.Series(["unknown"] * len(result))
        ).fillna("unknown")
        result["confidence"] = result.get(
            "confidence", pd.Series([0.0] * len(result))
        ).fillna(0.0)

        logger.info(
            "auto_label done: %s",
            dict(result["predicted_label"].value_counts()),
        )
        return result

    # ------------------------------------------------------------------

    def generate_spec(
        self,
        df: pd.DataFrame,
        task: str = "sentiment_classification",
        text_col: str = "text",
        label_col: str = "label",
        n_examples: int = 3,
        output_filename: str = "annotation_spec.md",
    ) -> str:
        """Generate a Markdown annotation specification and save it to disk.

        The spec contains:
        - Task description & objective
        - Class definitions
        - ≥3 representative examples per class
        - Edge cases (short texts, mixed sentiment, very long texts)
        - Labeling guidelines

        Parameters
        ----------
        df : pd.DataFrame
        task : str
            Task identifier (used as title and to tailor guidelines).
        text_col : str
            Column with raw text.
        label_col : str
            Column with existing labels (used to select examples).
        n_examples : int
            Minimum examples per class.

        Returns
        -------
        str — absolute path to the saved Markdown file.
        """
        label_col_use = label_col if label_col in df.columns else None
        spec_lines = self._build_spec(df, task, text_col, label_col_use, n_examples)
        spec_text = "\n".join(spec_lines)

        path = self.output_dir / output_filename
        path.write_text(spec_text, encoding="utf-8")
        logger.info("Annotation spec saved → %s", path)
        return str(path)

    # ------------------------------------------------------------------

    def check_quality(
        self,
        df_labeled: pd.DataFrame,
        reference_col: str = "label",
        pred_col: str = "predicted_label",
        conf_col: str = "confidence",
    ) -> dict:
        """Compute quality metrics for auto-labeled data.

        Metrics returned
        ----------------
        kappa                 : Cohen's κ (if reference_col present, else None)
        percent_agreement     : simple accuracy (if reference_col present, else None)
        label_dist            : {label: count} of predicted labels
        label_dist_pct        : {label: pct} of predicted labels
        confidence_mean       : mean confidence
        confidence_std        : std confidence
        confidence_min        : min confidence
        low_confidence_count  : rows below self.confidence_threshold
        low_confidence_pct    : % rows below threshold
        total_rows            : total rows analyzed

        Returns
        -------
        dict
        """
        from sklearn.metrics import cohen_kappa_score

        metrics: dict = {}
        n = len(df_labeled)
        metrics["total_rows"] = n

        # Confidence stats
        if conf_col in df_labeled.columns:
            c = df_labeled[conf_col].dropna()
            metrics["confidence_mean"] = round(float(c.mean()), 4)
            metrics["confidence_std"] = round(float(c.std()), 4)
            metrics["confidence_min"] = round(float(c.min()), 4)
            below = (c < self.confidence_threshold).sum()
            metrics["low_confidence_count"] = int(below)
            metrics["low_confidence_pct"] = round(float(below / n * 100), 2)
        else:
            metrics.update({
                "confidence_mean": None, "confidence_std": None,
                "confidence_min": None, "low_confidence_count": 0,
                "low_confidence_pct": 0.0,
            })

        # Label distribution
        if pred_col in df_labeled.columns:
            cnt = df_labeled[pred_col].value_counts().to_dict()
            metrics["label_dist"] = cnt
            metrics["label_dist_pct"] = {
                k: round(v / n * 100, 2) for k, v in cnt.items()
            }
        else:
            metrics["label_dist"] = {}
            metrics["label_dist_pct"] = {}

        # Agreement / Cohen's κ
        if reference_col in df_labeled.columns and pred_col in df_labeled.columns:
            valid = df_labeled[[reference_col, pred_col]].dropna()
            valid = valid[valid[pred_col] != "unknown"]
            if len(valid) >= 2:
                try:
                    kappa = cohen_kappa_score(
                        valid[reference_col].astype(str),
                        valid[pred_col].astype(str),
                    )
                    agree_pct = (
                        (valid[reference_col].astype(str) ==
                         valid[pred_col].astype(str)).mean() * 100
                    )
                    metrics["kappa"] = round(float(kappa), 4)
                    metrics["percent_agreement"] = round(float(agree_pct), 2)
                except Exception as exc:
                    logger.warning("Cohen's κ failed: %s", exc)
                    metrics["kappa"] = None
                    metrics["percent_agreement"] = None
            else:
                metrics["kappa"] = None
                metrics["percent_agreement"] = None
        else:
            metrics["kappa"] = None
            metrics["percent_agreement"] = None

        # Log summary
        logger.info(
            "Quality: κ=%.3f  agree=%.1f%%  conf_mean=%.3f  low_conf=%d",
            metrics.get("kappa") or 0,
            metrics.get("percent_agreement") or 0,
            metrics.get("confidence_mean") or 0,
            metrics.get("low_confidence_count", 0),
        )
        return metrics

    # ------------------------------------------------------------------

    def export_to_labelstudio(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        pred_col: str = "predicted_label",
        conf_col: str = "confidence",
        task_name: str = "sentiment",
        output_filename: str = "labelstudio_import.json",
        config_filename: str = "labelstudio_config.xml",
    ) -> dict:
        """Export annotated DataFrame to LabelStudio import format.

        Produces:
          - ``labelstudio_import.json`` — tasks with pre-annotations
          - ``labelstudio_config.xml``  — label interface configuration

        Parameters
        ----------
        df : pd.DataFrame
        task_name : str
            Name used in LabelStudio UI for the label control.

        Returns
        -------
        dict with keys ``tasks``, ``json_path``, ``config_path``.
        """
        tasks = []
        for idx, row in df.iterrows():
            text = row.get(text_col, "")
            if pd.isna(text):
                text = ""
            pred = row.get(pred_col, None)
            conf = float(row.get(conf_col, 0.0)) if conf_col in row.index else 0.0

            task: dict = {
                "id": int(idx) + 1,
                "data": {"text": str(text)},
            }

            # Add pre-annotation as prediction (visible to annotator as suggestion)
            if pred and pred != "unknown":
                task["predictions"] = [
                    {
                        "model_version": f"AnnotationAgent/{self.text_model.split('/')[-1]}",
                        "score": round(conf, 4),
                        "result": [
                            {
                                "id": str(uuid.uuid4())[:8],
                                "type": "choices",
                                "from_name": task_name,
                                "to_name": "text",
                                "value": {"choices": [pred]},
                            }
                        ],
                    }
                ]
            tasks.append(task)

        # LabelStudio XML label config
        choices_xml = "\n    ".join(
            f'<Choice value="{lbl}"/>' for lbl in self.labels
        )
        xml_config = f"""<View>
  <Text name="text" value="$text" granularity="word"/>
  <Header value="Sentiment Classification"/>
  <Choices name="{task_name}" toName="text" choice="single" showInLine="true">
    {choices_xml}
  </Choices>
</View>"""

        # Save artefacts
        json_path = self.output_dir / output_filename
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(tasks, fh, ensure_ascii=False, indent=2)

        config_path = self.output_dir / config_filename
        config_path.write_text(xml_config, encoding="utf-8")

        logger.info(
            "LabelStudio export → %s (%d tasks)", json_path, len(tasks)
        )
        return {
            "tasks": tasks,
            "json_path": str(json_path),
            "config_path": str(config_path),
            "n_tasks": len(tasks),
            "n_with_predictions": sum(1 for t in tasks if "predictions" in t),
        }

    # ------------------------------------------------------------------
    # BONUS: Human-in-the-loop
    # ------------------------------------------------------------------

    def flag_low_confidence(
        self,
        df_labeled: pd.DataFrame,
        threshold: float | None = None,
        conf_col: str = "confidence",
        output_filename: str = "hitl_review.csv",
    ) -> pd.DataFrame:
        """Flag examples with confidence < threshold for human review.

        Saves the flagged rows to ``data/annotations/hitl_review.csv``
        and the high-confidence rows to ``hitl_auto_accepted.csv``.

        Returns
        -------
        pd.DataFrame — the low-confidence (needs review) subset.
        """
        thr = threshold if threshold is not None else self.confidence_threshold

        if conf_col not in df_labeled.columns:
            logger.warning("No confidence column found — returning empty DataFrame.")
            return pd.DataFrame(columns=df_labeled.columns)

        low_mask = df_labeled[conf_col] < thr
        low_df = df_labeled[low_mask].copy()
        high_df = df_labeled[~low_mask].copy()

        low_path = self.output_dir / output_filename
        low_df.to_csv(low_path, index=False)

        high_path = self.output_dir / output_filename.replace(
            "hitl_review", "hitl_auto_accepted"
        )
        high_df.to_csv(high_path, index=False)

        logger.info(
            "HITL: %d low-conf (< %.2f) → %s  |  %d auto-accepted → %s",
            len(low_df), thr, low_path,
            len(high_df), high_path,
        )
        return low_df

    # ------------------------------------------------------------------
    # Private — text labeling
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> str:
        """Load (and cache) the HuggingFace pipeline. Return backend name."""
        if self._pipeline is not None:
            return self._backend

        # --- Try neural pipeline ---
        try:
            from transformers import pipeline as hf_pipeline
            import torch

            if self.zero_shot:
                # Zero-shot: works for any label set without fine-tuning
                zs_model = "typeform/distilbart-mnli-12-3"
                logger.info("Loading zero-shot pipeline: %s ...", zs_model)
                self._pipeline = hf_pipeline(
                    "zero-shot-classification",
                    model=zs_model,
                    device=_best_device(),
                )
                self._backend = "zero-shot"
            else:
                logger.info("Loading text-classification pipeline: %s ...", self.text_model)
                self._pipeline = hf_pipeline(
                    "text-classification",
                    model=self.text_model,
                    device=_best_device(),
                    truncation=True,
                    max_length=512,
                )
                self._backend = "neural"
            return self._backend

        except Exception as exc:
            logger.warning(
                "Neural pipeline unavailable (%s). Falling back to VADER.", exc
            )

        # --- VADER fallback ---
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._pipeline = SentimentIntensityAnalyzer()
            self._backend = "vader"
            logger.info("Using VADER sentiment analyser.")
            return self._backend
        except ImportError:
            pass

        # --- Keyword fallback ---
        logger.warning("VADER not available. Using keyword classifier.")
        self._pipeline = None
        self._backend = "keyword"
        return self._backend

    def _label_text(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Run text classification and add predicted_label + confidence."""
        backend = self._load_pipeline()
        texts = df[text_col].fillna("").tolist()

        if backend in ("neural", "zero-shot"):
            df = self._label_neural(df, texts, text_col)
        elif backend == "vader":
            df = self._label_vader(df, texts)
        else:
            df = self._label_keyword(df, texts)

        return df

    def _label_neural(
        self, df: pd.DataFrame, texts: list[str], text_col: str
    ) -> pd.DataFrame:
        """Batch inference with HuggingFace pipeline."""
        predicted, confidence = [], []

        if self._backend == "neural":
            # text-classification pipeline
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                results = self._pipeline(batch)
                for r in results:
                    raw = r["label"].upper()
                    lbl = _SST2_LABEL_MAP.get(raw, raw.lower())
                    predicted.append(lbl)
                    confidence.append(round(float(r["score"]), 4))

        else:
            # zero-shot pipeline
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                results = self._pipeline(batch, candidate_labels=self.labels)
                for r in results:
                    idx_best = r["scores"].index(max(r["scores"]))
                    predicted.append(r["labels"][idx_best])
                    confidence.append(round(float(r["scores"][idx_best]), 4))

        df = df.copy()
        df["predicted_label"] = predicted
        df["confidence"] = confidence
        return df

    def _label_vader(
        self, df: pd.DataFrame, texts: list[str]
    ) -> pd.DataFrame:
        """VADER-based sentiment labeling."""
        predicted, confidence = [], []
        for text in texts:
            score = self._pipeline.polarity_scores(str(text))
            compound = score["compound"]
            if compound >= 0.05:
                predicted.append("positive")
                confidence.append(round(min(1.0, 0.5 + compound * 0.5), 4))
            elif compound <= -0.05:
                predicted.append("negative")
                confidence.append(round(min(1.0, 0.5 + abs(compound) * 0.5), 4))
            else:
                predicted.append("negative")
                confidence.append(round(0.5 + abs(compound), 4))

        df = df.copy()
        df["predicted_label"] = predicted
        df["confidence"] = confidence
        return df

    def _label_keyword(
        self, df: pd.DataFrame, texts: list[str]
    ) -> pd.DataFrame:
        """Simple keyword-based fallback."""
        pos_kw = {"good", "great", "excellent", "amazing", "love", "best", "beautiful",
                  "fantastic", "wonderful", "perfect", "brilliant", "outstanding"}
        neg_kw = {"bad", "terrible", "awful", "worst", "hate", "horrible", "boring",
                  "disappointing", "dreadful", "poor", "mediocre", "waste"}
        predicted, confidence = [], []
        for text in texts:
            words = set(re.findall(r"\b\w+\b", str(text).lower()))
            pos = len(words & pos_kw)
            neg = len(words & neg_kw)
            total = pos + neg + 1e-9
            if pos >= neg:
                predicted.append("positive")
                confidence.append(round(0.5 + 0.5 * pos / total, 4))
            else:
                predicted.append("negative")
                confidence.append(round(0.5 + 0.5 * neg / total, 4))
        df = df.copy()
        df["predicted_label"] = predicted
        df["confidence"] = confidence
        return df

    # ------------------------------------------------------------------
    # Private — audio / image stubs
    # ------------------------------------------------------------------

    def _label_audio_stub(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stub for Whisper-based audio transcription + classification.

        To activate: install ``openai-whisper`` and load an audio column
        with file paths. This stub returns placeholder labels.
        """
        logger.warning(
            "Audio modality: Whisper stub active. Install openai-whisper "
            "and pass a column with audio file paths to enable real inference."
        )
        df = df.copy()
        df["predicted_label"] = "unknown"
        df["confidence"] = 0.0
        df["transcription"] = "[STUB — Whisper not loaded]"
        return df

    def _label_image_stub(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stub for YOLO-based image classification.

        To activate: install ``ultralytics`` and pass a column with image paths.
        """
        logger.warning(
            "Image modality: YOLO stub active. Install ultralytics "
            "and pass a column with image file paths."
        )
        df = df.copy()
        df["predicted_label"] = "unknown"
        df["confidence"] = 0.0
        df["bbox"] = None
        return df

    # ------------------------------------------------------------------
    # Private — spec generation
    # ------------------------------------------------------------------

    def _build_spec(
        self,
        df: pd.DataFrame,
        task: str,
        text_col: str,
        label_col: str | None,
        n_examples: int,
    ) -> list[str]:
        ts = datetime.now().strftime("%Y-%m-%d")

        lines = [
            f"# Annotation Specification: {task.replace('_', ' ').title()}",
            f"**Generated:** {ts}  |  **Agent:** AnnotationAgent v1.0",
            "",
            "---",
            "",
            "## 1. Task Description",
            "",
            f"**Goal:** Classify each text sample into one of "
            f"`{len(self.labels)}` predefined classes.",
            "",
            "**ML use case:** Binary sentiment classification of multi-source text data "
            "(movie reviews, book titles). The resulting labels will be used to train and "
            "evaluate a text classifier (e.g. DistilBERT fine-tuning).",
            "",
            "**Input:** A single text string (review body or book title).",
            "**Output:** One label from the set below.",
            "",
            "---",
            "",
            "## 2. Label Definitions",
            "",
        ]

        label_defs = {
            "positive": (
                "The text expresses a favourable, enthusiastic, or satisfied opinion. "
                "The author recommends the item or describes it in clearly positive terms. "
                "Joy, excitement, admiration, or praise are typical markers."
            ),
            "negative": (
                "The text expresses an unfavourable, critical, or disappointed opinion. "
                "The author would not recommend the item or describes clear flaws, "
                "boredom, or frustration."
            ),
        }

        for lbl in self.labels:
            definition = label_defs.get(
                lbl, f"Text belonging to the '{lbl}' category."
            )
            lines += [f"### `{lbl}`", "", f"**Definition:** {definition}", ""]

            # Select examples
            if label_col and label_col in df.columns:
                subset = df[df[label_col] == lbl][text_col].dropna()
                # Prefer medium-length examples
                wc = subset.str.split().str.len()
                medium = subset[(wc >= 15) & (wc <= 120)]
                pool = medium if len(medium) >= n_examples else subset
                examples = pool.sample(
                    min(n_examples, len(pool)), random_state=42
                ).tolist()
            else:
                examples = []

            if examples:
                lines.append("**Examples:**")
                for i, ex in enumerate(examples, 1):
                    # Truncate very long examples for readability
                    short = " ".join(str(ex).split()[:60])
                    if len(str(ex).split()) > 60:
                        short += " ..."
                    lines.append(f"{i}. *\"{short}\"*")
                lines.append("")
            else:
                lines += [
                    "**Examples:**",
                    "1. *(no examples available — run auto_label first)*",
                    "",
                ]

        lines += [
            "---",
            "",
            "## 3. Edge Cases",
            "",
            "| Scenario | Recommended label | Notes |",
            "|---|---|---|",
            "| Sarcasm / irony | `negative` | Treat the intended meaning, not the literal words. |",
            "| Mixed positive + negative | majority sentiment | Label the dominant tone of the text. |",
            "| Very short text (1–3 words) | most likely label | Low confidence — flag for review. |",
            "| Lists without commentary | `negative` (default) | No sentiment signal present. |",
            "| All-caps emotional text | as written | Intensity doesn't change valence. |",
            "| Foreign-language text | `negative` (default) | Cannot be reliably classified. |",
            "",
            "---",
            "",
            "## 4. Annotation Guidelines",
            "",
            "1. **Read the full text** before deciding — first and last sentences often "
            "carry the main sentiment.",
            "2. **Ignore topic** — you are labelling *sentiment*, not topic or quality of writing.",
            "3. **Avoid anchoring** — do not let previous labels influence the current decision.",
            "4. **Flag uncertainty** — if you are less than 70% confident, mark the example "
            "for expert review (use the LabelStudio 'flag' button).",
            "5. **Target rate:** aim to annotate 30–50 examples per hour.",
            "6. **Inter-annotator check:** a sample of 10% will be double-annotated to "
            "measure Cohen's κ. Target κ ≥ 0.70.",
            "",
            "---",
            "",
            "## 5. LabelStudio Setup",
            "",
            "1. Import `labelstudio_import.json` via **Import** in LabelStudio.",
            "2. Copy the content of `labelstudio_config.xml` into your project's "
            "**Labeling Interface** editor.",
            "3. Pre-annotations are loaded as *predictions* — you can accept, reject, or "
            "correct each one.",
            "",
            "---",
            "",
            f"*Document generated automatically by AnnotationAgent on {ts}.*",
        ]
        return lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _best_device() -> int | str:
    """Return the best available device identifier for HuggingFace pipelines."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0  # first CUDA GPU
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
    except ImportError:
        pass
    return -1  # CPU
