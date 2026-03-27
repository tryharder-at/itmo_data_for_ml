"""
Streamlit HITL App — Human-in-the-Loop label review interface.

Run:
    streamlit run hitl_app.py

Workflow:
    1. Load review_queue.csv (written by run_pipeline.py step 4)
    2. Inspect each example; change predicted_label if wrong
    3. Click "Save & Export" → writes review_queue_corrected.csv
    4. Return to run_pipeline.py and press Enter to continue
"""

from pathlib import Path
import pandas as pd
import streamlit as st

REVIEW_QUEUE = Path("review_queue.csv")
REVIEW_DONE  = Path("review_queue_corrected.csv")

st.set_page_config(
    page_title="HITL Label Review",
    page_icon="🏷️",
    layout="wide",
)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🏷️ Human-in-the-Loop Label Review")
st.caption(
    "Review low-confidence auto-labelled examples. "
    "Edit the **predicted_label** column directly in the table below, "
    "then click **Save & Export**."
)

# ─── Load data ────────────────────────────────────────────────────────────────
if not REVIEW_QUEUE.exists():
    st.warning(
        f"`{REVIEW_QUEUE}` not found. "
        "Run `python run_pipeline.py` first (it will pause at the HITL step)."
    )
    st.stop()

df = pd.read_csv(REVIEW_QUEUE)

# ─── Sidebar stats ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Queue Statistics")
    st.metric("Total rows to review", len(df))
    if "confidence" in df.columns:
        st.metric("Mean confidence", f"{df['confidence'].mean():.3f}")
        st.metric("Min confidence",  f"{df['confidence'].min():.3f}")
    if "predicted_label" in df.columns:
        st.subheader("Predicted distribution")
        st.bar_chart(df["predicted_label"].value_counts())
    if "label" in df.columns:
        st.subheader("Original distribution")
        st.bar_chart(df["label"].value_counts())

# ─── Instructions ─────────────────────────────────────────────────────────────
with st.expander("ℹ️ How to use", expanded=False):
    st.markdown("""
1. The table below shows all examples where the model was **unsure** (confidence < 0.70).
2. The `predicted_label` column contains the model's guess.
3. The `label` column shows the **original** label from the dataset (ground-truth reference).
4. **If the prediction is wrong**, change `predicted_label` to the correct value.
5. Click **Save & Export** at the bottom when done.
6. Return to the terminal running `run_pipeline.py` and press **Enter**.
    """)

# ─── Editable table ───────────────────────────────────────────────────────────
cols_to_show = [c for c in ["text", "label", "predicted_label", "confidence", "source"]
                if c in df.columns]

st.subheader(f"📋 Review queue — {len(df)} rows")

edited_df = st.data_editor(
    df[cols_to_show],
    column_config={
        "text": st.column_config.TextColumn("Text", width="large"),
        "label": st.column_config.TextColumn("Original label", disabled=True),
        "predicted_label": st.column_config.SelectboxColumn(
            "Predicted label ✏️",
            options=["positive", "negative"],
            required=True,
        ),
        "confidence": st.column_config.NumberColumn(
            "Confidence", format="%.3f", disabled=True
        ),
        "source": st.column_config.TextColumn("Source", disabled=True),
    },
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
)

# ─── Diff summary ─────────────────────────────────────────────────────────────
if "predicted_label" in df.columns:
    orig = df["predicted_label"].values[: len(edited_df)]
    changed_mask = edited_df["predicted_label"].values != orig
    n_changed = changed_mask.sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows reviewed", len(edited_df))
    col2.metric("Labels changed", int(n_changed))
    col3.metric("Agreement rate",
                f"{(1 - n_changed / max(len(edited_df), 1)) * 100:.1f}%")

    if n_changed > 0:
        st.subheader("🔄 Changed labels")
        changed_df = df[cols_to_show].copy()
        changed_df["corrected_label"] = edited_df["predicted_label"].values
        st.dataframe(
            changed_df[changed_mask][
                [c for c in ["text", "label", "predicted_label", "corrected_label",
                              "confidence"] if c in changed_df.columns]
            ],
            use_container_width=True,
            hide_index=True,
        )

# ─── Save ─────────────────────────────────────────────────────────────────────
st.divider()
col_save, col_msg = st.columns([1, 3])

with col_save:
    if st.button("💾 Save & Export", type="primary", use_container_width=True):
        # Merge edits back into full df
        full_corrected = df.copy()
        full_corrected["predicted_label"] = edited_df["predicted_label"].values
        full_corrected.to_csv(REVIEW_DONE, index=False)
        st.success(f"Saved {len(full_corrected)} rows → `{REVIEW_DONE}`")
        st.info("Return to the terminal and press **Enter** to continue the pipeline.")

with col_msg:
    if REVIEW_DONE.exists():
        done_df = pd.read_csv(REVIEW_DONE)
        st.success(
            f"✅ `{REVIEW_DONE}` already exists ({len(done_df)} rows). "
            "The pipeline will use it automatically."
        )
