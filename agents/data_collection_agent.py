"""
DataCollectionAgent — collects sentiment-labelled text from multiple sources
and returns a unified pandas DataFrame.

Unified output schema
---------------------
text        : str   — raw text (review, book title, etc.)
label       : str   — "positive" or "negative"
source      : str   — origin identifier
collected_at: str   — ISO-8601 timestamp

Sources
-------
- hf_dataset  : HuggingFace Datasets (load_dataset)
- scrape      : HTML scraping with CSS selectors
- api         : REST API; built-in handlers for "openlibrary" and "hackernews";
                generic JSON-list fallback also available
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["text", "label", "source", "collected_at"]

# books.toscrape.com uses word ratings; map to binary sentiment
STAR_MAP = {
    "one": "negative",
    "two": "negative",
    "three": None,        # neutral → skip
    "four": "positive",
    "five": "positive",
}


class DataCollectionAgent:
    """Multi-source text data collection agent.

    Parameters
    ----------
    config : str | Path
        Path to config.yaml.
    """

    def __init__(self, config: str | Path = "config.yaml") -> None:
        with open(config, "r", encoding="utf-8") as fh:
            self.cfg = yaml.safe_load(fh)

        self.request_delay: float = self.cfg["agent"].get("request_delay", 0.4)
        out_cfg = self.cfg.get("output", {})
        self.out_dir = Path(out_cfg.get("dir", "data/raw"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # ------------------------------------------------------------------
    # Public skills
    # ------------------------------------------------------------------

    def scrape(
        self,
        url: str,
        selectors: dict[str, str],
        pages: int = 1,
        source_name: str = "scraped",
        label_transform: str | None = None,
    ) -> pd.DataFrame:
        """Scrape text + label from paginated HTML pages.

        Parameters
        ----------
        url : str
            URL of page 1. For subsequent pages the agent replaces the
            page number in the path (books.toscrape.com convention).
        selectors : dict
            CSS selectors with keys ``item``, ``text``, ``label``.
        pages : int
            Total number of pages to scrape.
        source_name : str
            Fills the ``source`` column.
        label_transform : str | None
            ``"star_rating"`` → convert word-rating class to pos/neg.

        Returns
        -------
        pd.DataFrame
        """
        records: list[dict] = []
        collected_at = datetime.now().isoformat()
        headers = {"User-Agent": "Mozilla/5.0 (DataCollectionAgent/1.0)"}

        for page in tqdm(range(1, pages + 1), desc=f"Scraping {source_name}"):
            # Build paginated URL: replace the page number in the template
            page_url = url.replace("page-1.html", f"page-{page}.html")
            try:
                resp = requests.get(page_url, headers=headers, timeout=15)
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("Failed to fetch %s: %s", page_url, exc)
                break

            soup = BeautifulSoup(resp.text, "lxml")
            items = soup.select(selectors["item"])
            if not items:
                logger.info("No items on page %d, stopping.", page)
                break

            for item in items:
                text_el = item.select_one(selectors["text"])
                label_el = item.select_one(selectors["label"])

                # books.toscrape: title is in the 'title' attribute of <a>
                if text_el is not None:
                    text = text_el.get("title") or text_el.get_text(strip=True)
                else:
                    text = None

                if label_transform == "star_rating" and label_el is not None:
                    # class list looks like ['star-rating', 'Four']
                    classes = [c.lower() for c in label_el.get("class", [])]
                    star_word = next(
                        (c for c in classes if c in STAR_MAP), None
                    )
                    label = STAR_MAP.get(star_word) if star_word else None
                elif label_el is not None:
                    label = label_el.get_text(strip=True)
                else:
                    label = None

                # Skip neutral / missing entries
                if text and label:
                    records.append(
                        {
                            "text": text.strip(),
                            "label": label,
                            "source": source_name,
                            "collected_at": collected_at,
                        }
                    )

            time.sleep(self.request_delay)

        df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
        logger.info("Scraped %d records from %s", len(df), source_name)
        return df

    # ------------------------------------------------------------------

    def fetch_api(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        api_type: str = "generic",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch data from a REST API.

        Built-in handlers
        -----------------
        ``api_type='openlibrary'`` : OpenLibrary search — uses
            ``ratings_average`` to assign positive/negative labels.
        ``api_type='generic'``     : JSON list → needs ``text_field`` and
            ``label_field`` kwargs.

        Returns
        -------
        pd.DataFrame
        """
        if api_type == "openlibrary":
            return self._fetch_openlibrary(endpoint, params=params, **kwargs)

        # Generic fallback
        resp = requests.get(endpoint, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        text_field = kwargs.get("text_field", "text")
        label_field = kwargs.get("label_field", "label")
        source_name = kwargs.get("source_name", "api")
        collected_at = datetime.now().isoformat()

        records = [
            {
                "text": str(item.get(text_field, "")),
                "label": str(item.get(label_field, "unknown")),
                "source": source_name,
                "collected_at": collected_at,
            }
            for item in data
            if item.get(text_field)
        ]
        df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
        logger.info("Fetched %d records from %s", len(df), endpoint)
        return df

    # ------------------------------------------------------------------

    def load_dataset(
        self,
        name: str,
        source: str = "hf",
        split: str = "train",
        sample_size: int | None = None,
        text_column: str = "text",
        label_column: str = "label",
        label_map: dict | None = None,
        source_name: str | None = None,
    ) -> pd.DataFrame:
        """Load a dataset from HuggingFace (``source='hf'``) or Kaggle.

        Returns
        -------
        pd.DataFrame with columns ``text, label, source, collected_at``.
        """
        if source == "hf":
            return self._load_hf(
                name,
                split=split,
                sample_size=sample_size,
                text_column=text_column,
                label_column=label_column,
                label_map=label_map,
                source_name=source_name or name,
            )
        if source == "kaggle":
            return self._load_kaggle(name, source_name=source_name or name)
        raise ValueError(f"Unknown dataset source: {source!r}")

    # ------------------------------------------------------------------

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate DataFrames, enforce the standard schema, drop empties.

        Returns
        -------
        pd.DataFrame with columns ``text, label, source, collected_at``.
        """
        combined = pd.concat(sources, ignore_index=True)
        combined = combined[OUTPUT_COLUMNS].copy()
        combined["text"] = combined["text"].astype(str).str.strip()
        combined = combined[combined["text"].str.len() > 0].reset_index(drop=True)
        logger.info("Merged dataset: %d rows total", len(combined))
        return combined

    # ------------------------------------------------------------------

    def run(self, sources: list[dict] | None = None) -> pd.DataFrame:
        """Collect data from all sources and return a unified DataFrame.

        If ``sources`` is None the list from config.yaml is used.

        Returns
        -------
        pd.DataFrame
        """
        source_cfgs = sources or self.cfg.get("sources", [])
        frames: list[pd.DataFrame] = []

        for cfg in source_cfgs:
            src_type = cfg.get("type", "")
            logger.info("Processing source type=%s name=%s ...",
                        src_type, cfg.get("source_name", "?"))

            df: pd.DataFrame | None = None

            if src_type == "hf_dataset":
                df = self.load_dataset(
                    name=cfg["name"],
                    source="hf",
                    split=cfg.get("split", "train"),
                    sample_size=cfg.get("sample_size"),
                    text_column=cfg.get("text_column", "text"),
                    label_column=cfg.get("label_column", "label"),
                    label_map=cfg.get("label_map"),
                    source_name=cfg.get("source_name", cfg["name"]),
                )

            elif src_type == "scrape":
                df = self.scrape(
                    url=cfg["url"],
                    selectors=cfg["selectors"],
                    pages=cfg.get("pages", 1),
                    source_name=cfg.get("source_name", "scraped"),
                    label_transform=cfg.get("label_transform"),
                )

            elif src_type == "api":
                df = self.fetch_api(
                    endpoint=cfg["endpoint"],
                    params=cfg.get("params"),
                    api_type=cfg.get("api_type", "generic"),
                    label_threshold=cfg.get("label_threshold", 3.8),
                    min_ratings_count=cfg.get("min_ratings_count", 5),
                    source_name=cfg.get("source_name", "api"),
                )

            else:
                logger.warning("Unknown source type %r — skipping.", src_type)
                continue

            if df is not None and not df.empty:
                out_cfg = self.cfg.get("output", {})
                if out_cfg.get("save_individual", False):
                    fname = self.out_dir / f"{cfg.get('source_name', src_type)}.csv"
                    df.to_csv(fname, index=False)
                    logger.info("Saved  %s  (%d rows)", fname, len(df))
                frames.append(df)

        if not frames:
            logger.warning("No data collected!")
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        unified = self.merge(frames)

        out_cfg = self.cfg.get("output", {})
        unified_path = self.out_dir / out_cfg.get("unified_file", "unified_dataset.csv")
        unified.to_csv(unified_path, index=False)
        logger.info(
            "Unified dataset saved → %s  (%d rows)", unified_path, len(unified)
        )
        return unified

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_hf(
        self,
        name: str,
        split: str,
        sample_size: int | None,
        text_column: str,
        label_column: str,
        label_map: dict | None,
        source_name: str,
        shuffle_seed: int = 42,
    ) -> pd.DataFrame:
        try:
            from datasets import load_dataset as hf_load  # type: ignore
        except ImportError as exc:
            raise ImportError("Install `datasets`:  pip install datasets") from exc

        logger.info("Loading HuggingFace '%s' (split=%s) ...", name, split)
        ds = hf_load(name, split=split)

        # Always shuffle so that sample_size gives a balanced class distribution
        ds = ds.shuffle(seed=shuffle_seed)

        if sample_size is not None:
            ds = ds.select(range(min(sample_size, len(ds))))

        raw_df = ds.to_pandas()
        label_series = raw_df[label_column].astype(str)
        if label_map:
            label_series = label_series.map(label_map).fillna(label_series)

        df = pd.DataFrame(
            {
                "text": raw_df[text_column],
                "label": label_series,
                "source": source_name,
                "collected_at": datetime.now().isoformat(),
            }
        )
        logger.info("Loaded %d rows from HuggingFace '%s'", len(df), name)
        return df

    def _load_kaggle(self, name: str, source_name: str) -> pd.DataFrame:
        import subprocess

        target = self.out_dir / name.replace("/", "_")
        target.mkdir(exist_ok=True)
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", name,
             "-p", str(target), "--unzip"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"kaggle CLI failed: {result.stderr}")

        csvs = list(target.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found after downloading {name}")

        raw_df = pd.read_csv(csvs[0])
        text_col = next(
            (c for c in raw_df.columns if "text" in c.lower()), raw_df.columns[0]
        )
        label_col = next(
            (c for c in raw_df.columns
             if "label" in c.lower() or "target" in c.lower()), None
        )
        df = pd.DataFrame(
            {
                "text": raw_df[text_col],
                "label": raw_df[label_col].astype(str) if label_col else "unknown",
                "source": source_name,
                "collected_at": datetime.now().isoformat(),
            }
        )
        logger.info("Loaded %d rows from Kaggle '%s'", len(df), name)
        return df

    def _fetch_openlibrary(
        self,
        endpoint: str,
        params: dict | None = None,
        label_threshold: float = 3.8,
        min_ratings_count: int = 5,
        source_name: str = "openlibrary_api",
        max_retries: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        """Fetch books from OpenLibrary search API; label by community rating."""
        logger.info("Fetching OpenLibrary books (endpoint=%s) ...", endpoint)
        resp = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(endpoint, params=params, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                logger.warning("OpenLibrary attempt %d/%d failed: %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("OpenLibrary unavailable after %d retries — returning empty DataFrame.", max_retries)
                    return pd.DataFrame(columns=OUTPUT_COLUMNS)
        docs = resp.json().get("docs", [])

        records: list[dict] = []
        collected_at = datetime.now().isoformat()

        for doc in docs:
            title: str = doc.get("title", "").strip()
            avg: float | None = doc.get("ratings_average")
            count: int = doc.get("ratings_count", 0) or 0

            if not title or avg is None or count < min_ratings_count:
                continue

            label = "positive" if avg >= label_threshold else "negative"
            records.append(
                {
                    "text": title,
                    "label": label,
                    "source": source_name,
                    "collected_at": collected_at,
                }
            )

        df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
        logger.info("Fetched %d books from OpenLibrary", len(df))
        return df
