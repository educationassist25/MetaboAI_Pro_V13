"""
MetaboAI Pro — Automated LC-MS Metabolomics Statistical Analysis and Reporting Platform
Streamlit application entry point.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as _plt
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from modules import utils, qc, normalization, imputation_module, dataset_manager, stats_analysis, pca_module, volcano, biomarker, heatmap_module, boxplot_module

st.set_page_config(page_title="MetaboAI Pro", layout="wide", page_icon="🧪")

# Compact heatmap annotation-color controls.
st.markdown(
    """
    <style>
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        padding-top: 0.35rem;
        padding-bottom: 0.45rem;
    }
    div[data-testid="stColorPicker"] {
        margin-top: -0.25rem;
        margin-bottom: 0.15rem;
    }
    div[data-testid="stColorPicker"] button {
        min-height: 30px;
        height: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
for key, default in [
    ("raw_peak_df", None), ("meta", None), ("data_type", "Untargeted Metabolomics"),
    ("row_annotations", None),
    ("heatmap_annotation_colors_applied", None), ("heatmap_colors_reset_pending", False),
    ("qc_cols", []), ("sample_cols", []), ("istd_normalized", None),
    ("iqr_normalized", None), ("log2_data", None), ("log2_raw", None),
    ("stats_result", None), ("stats_result_groups", None),
    ("anova_result", None), ("posthoc_result", None), ("anova_groups_used", None),
    ("stats_result_group_col", None),
    ("processing_notes", []), ("qc_figs", {}), ("viz_figs", {}), ("heatmap_fig", None), ("heatmap_zscore", None),
    ("volcano_fig", None), ("volcano_annotated", None), ("volcano_settings", None),
    ("boxplot_fig", None), ("boxplot_stats", None),
    ("missingness_table", None), ("cleaned_data", None), ("imputed_data", None), ("imputation_method_used", None),
    ("data_mode", "single"), ("datasets", {}), ("combined_done", False),
    ("raw_peak_df_qc", None), ("cv_table", None), ("qc_log_data", None),
    ("combined_normalized_prelog2", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


class _SingleDatasetState(dict):
    """Adapter so the shared render_cleaning_ui/render_qc_ui/render_normalization_ui
    functions can read/write the flat single-dataset session_state keys through the
    same dict-like interface used for multi-dataset mode (where each dataset is its
    own plain dict)."""
    _MAP = {
        "cleaned_data": "cleaned_data", "imputed_data": "imputed_data",
        "missingness_table": "missingness_table", "imputation_method_used": "imputation_method_used",
        "raw_peak_df_qc": "raw_peak_df_qc", "cv_table": "cv_table", "qc_figs": "qc_figs",
        "qc_log_data": "qc_log_data",
        "istd_normalized": "istd_normalized", "iqr_normalized": "iqr_normalized",
        "log2_data": "log2_data",
    }

    def __getitem__(self, key):
        if key == "processing_notes":
            return st.session_state.processing_notes
        return st.session_state[self._MAP[key]]

    def __setitem__(self, key, value):
        if key == "processing_notes":
            st.session_state.processing_notes = value
        else:
            st.session_state[self._MAP[key]] = value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def reset_downstream_analysis_state():
    """
    Clear every downstream-analysis result (Statistics, ANOVA, post-hoc, Biomarkers)
    whenever new data is loaded. Without this, a stale stats_result from a previous
    dataset/mode can persist across a reload, and tabs that read it (Heatmap, Volcano,
    Biomarker Discovery) can then reference a log2 matrix that no longer matches --
    e.g. crashing on mismatched columns -- since those tabs assume stats_result is
    None until freshly (re)computed against the currently-loaded data.
    """
    st.session_state.stats_result = None
    st.session_state.stats_result_groups = None
    st.session_state.stats_result_group_col = None
    st.session_state.anova_result = None
    st.session_state.posthoc_result = None
    st.session_state.anova_groups_used = None
    st.session_state.biomarkers = None


def render_dataset_dropdown(datasets: dict, stage_key: str, stage_col: str) -> str:
    """
    Shared multi-dataset selector for Tabs 2-4: a status dashboard (one row per
    dataset, showing Cleaning/QC/Normalization progress) followed by a dropdown to
    pick which single dataset to configure below — replaces the old nested-tabs UI
    (which meant up to 5 tab levels deep with 4+ datasets) with one flat, always-visible
    overview plus a single selection control. Returns the selected dataset's key.
    `stage_col` (e.g. "2. QC") names which stage this tab is currently working on.
    """
    status_df = dataset_manager.dataset_status_table(datasets)
    st.dataframe(status_df, width='stretch', hide_index=True)
    st.caption(f"This tab configures the **{stage_col}** column above, one dataset at a time.")

    labels = {ds_id: f"{ds['label']} ({ds['data_type']})" for ds_id, ds in datasets.items()}
    ids = list(labels.keys())
    state_key = f"multi_select_{stage_key}"
    if st.session_state.get(state_key) not in ids:
        st.session_state[state_key] = ids[0]
    selected_id = st.selectbox(
        "Select a dataset/method to configure", options=ids,
        format_func=lambda k: labels[k], key=state_key
    )
    return selected_id


st.title("MetaboAI Pro")
st.caption("LC-MS based Metabolomics & Lipidomics Statistical Analysis")

TABS = st.tabs([
    "**Data Upload**", "**Data Cleaning & Imputation**", "**QC Validation**", "**Combined QC**",
    "**Normalization**", "**Combined Normalized Data**", "**PCA**", "**Statistics**",
    "**Volcano Plot**", "**Biomarker Discovery**", "**Heatmap**", "**Boxplot**"
])

# ===========================================================================
# TAB 1 — DATA UPLOAD
# ===========================================================================
with TABS[0]:
    st.header("Data Upload & Study Configuration")

    st.session_state.data_mode = st.radio(
        "How many datasets are you uploading?",
        ["Single dataset", "Multiple datasets (combine methods/modes)"],
        index=0 if st.session_state.data_mode == "single" else 1,
        horizontal=True,
        help="Real LC-MS studies often run the same samples through multiple chromatography "
             "methods and/or ionization modes (e.g. Method1-Positive, Method1-Negative, "
             "Method2-Positive, Method2-Negative) since no single method captures the full "
             "metabolome. Choose 'Multiple datasets' to clean, QC, and normalize each one "
             "independently, then combine them into one unified matrix for downstream analysis."
    )
    st.session_state.data_mode = "single" if st.session_state.data_mode == "Single dataset" else "multi"

    # =======================================================================
    # SINGLE-DATASET MODE (original workflow, unchanged)
    # =======================================================================
    if st.session_state.data_mode == "single":
        st.session_state.data_type = st.selectbox(
            "Analysis type",
            ["Untargeted Metabolomics", "Targeted Metabolomics", "Untargeted Lipidomics", "Targeted Lipidomics"],
        )
        is_targeted_demo = "Targeted" in st.session_state.data_type
        demo_prefix = "targeted" if is_targeted_demo else "untargeted"
        demo_desc = (
            "60 features incl. an ISTD row, 24 biological samples across 4 groups "
            "(Control/Mild/Moderate/Severe), 6 QC replicates"
            if is_targeted_demo else
            "300 features (no ISTD — untargeted has no single spiked standard), "
            "24 biological samples across 4 groups (Control/Mild/Moderate/Severe), 6 QC replicates"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Peak Area Matrix")
            st.caption("First column = Metabolite/Lipid ID, remaining columns = sample peak areas (incl. QC).")
            peak_file = st.file_uploader("Upload peak area matrix (CSV/XLSX)", type=["csv", "xlsx"], key="peak_upload")
            use_demo = st.checkbox("Use built-in demo dataset instead", value=(peak_file is None))
            demo_choice = "Standard"
            if use_demo or peak_file is None:
                demo_choice = st.radio(
                    "Demo dataset", ["Standard", "Rich Clinical Demo"], horizontal=True,
                    key="single_demo_choice"
                )
                if demo_choice == "Standard":
                    st.caption(
                        f"Suggested demo for **{st.session_state.data_type}**: {demo_desc}. "
                        f"The matching dataset is selected automatically."
                    )
                else:
                    st.caption(
                        "200 metabolites, 36 biological samples across 4 diagnosis groups "
                        "(Healthy Control, Prediabetic, Type 2 Diabetes, Metabolic Syndrome, "
                        "9 each) + 6 QC replicates = 42 total. Metadata includes Diagnosis, "
                        "Age, Gender, Treatment, Ethnicity, and Body Weight — any of these can "
                        "be picked as the grouping/coloring variable throughout the app, not "
                        "just Diagnosis. Also loads metabolite Method + Pathway row annotations "
                        "for the Heatmap tab's row annotation tracks."
                    )
        with col2:
            st.subheader("Sample Metadata")
            st.caption("Columns: Sample, Group, IsQC (True/False), Batch (optional) — any additional "
                       "columns (Diagnosis, Age, Gender, Treatment, Ethnicity, ...) are kept and become "
                       "selectable grouping/coloring variables in PCA, Statistics, Heatmap, and Boxplot.")
            meta_file = st.file_uploader("Upload metadata table (CSV/XLSX)", type=["csv", "xlsx"], key="meta_upload")

        st.subheader("Metabolite Row Annotations (optional)")
        st.caption("Columns: Metabolite, then any annotation columns (e.g. Method, Pathway) — enables "
                   "the Heatmap tab's row annotation tracks. Not required for PCA/Statistics/Volcano/"
                   "Biomarker/Boxplot.")
        row_annot_file = st.file_uploader("Upload row annotation table (CSV/XLSX)",
                                           type=["csv", "xlsx"], key="row_annot_upload")

        if st.button("Load & Validate Data", type="primary"):
            try:
                row_annotations = None
                if use_demo or peak_file is None:
                    if demo_choice == "Rich Clinical Demo":
                        peak_path = os.path.join(os.path.dirname(__file__), "sample_peak_area_matrix_richdemo.csv")
                        meta_path = os.path.join(os.path.dirname(__file__), "sample_metadata_richdemo.csv")
                        annot_path = os.path.join(os.path.dirname(__file__), "metabolite_row_annotations_richdemo.csv")
                        peak_raw = pd.read_csv(peak_path)
                        meta_raw = pd.read_csv(meta_path)
                        row_annotations = pd.read_csv(annot_path).set_index("Metabolite")
                        st.info("Using the built-in **Rich Clinical Demo** dataset (200 metabolites, "
                                "42 samples, 6 metadata variables, Method + Pathway row annotations).")
                    else:
                        peak_path = os.path.join(os.path.dirname(__file__), f"sample_peak_area_matrix_{demo_prefix}.csv")
                        meta_path = os.path.join(os.path.dirname(__file__), f"sample_metadata_{demo_prefix}.csv")
                        peak_raw = pd.read_csv(peak_path)
                        meta_raw = pd.read_csv(meta_path)
                        st.info(f"Using the built-in **{demo_prefix}** demo dataset, matching your "
                                f"'{st.session_state.data_type}' selection: {demo_desc}.")
                else:
                    peak_raw = utils.load_table(peak_file)
                    meta_raw = utils.load_table(meta_file) if meta_file is not None else None
                    if meta_raw is None:
                        st.error("Please upload a metadata table (or check 'use demo dataset').")
                        st.stop()

                peak_df = utils.validate_peak_matrix(peak_raw)
                meta = utils.validate_metadata(meta_raw, peak_df.columns)
                qc_cols, sample_cols = utils.split_qc_and_samples(peak_df, meta)

                if row_annot_file is not None and row_annotations is None:
                    row_annot_raw = utils.load_table(row_annot_file)
                    if "Metabolite" not in row_annot_raw.columns:
                        st.warning("Row annotation file has no 'Metabolite' column — ignoring it.")
                    else:
                        row_annotations = row_annot_raw.set_index("Metabolite")

                st.session_state.raw_peak_df = peak_df
                st.session_state.meta = meta
                st.session_state.qc_cols = qc_cols
                st.session_state.sample_cols = sample_cols
                st.session_state.row_annotations = row_annotations
                st.session_state.processing_notes = [f"Analysis type: {st.session_state.data_type}"]
                st.session_state.log2_data = None
                reset_downstream_analysis_state()

                st.success(f"Loaded {peak_df.shape[0]} features × {peak_df.shape[1]} samples "
                           f"({len(qc_cols)} QC, {len(sample_cols)} study samples)."
                           + (f" Row annotations: {', '.join(row_annotations.columns)}."
                              if row_annotations is not None else ""))
            except utils.DataValidationError as e:
                st.error(f"Validation error: {e}")
            except Exception as e:
                st.error(f"Unexpected error loading data: {e}")

        if st.session_state.raw_peak_df is not None:
            st.subheader("Preview")
            st.dataframe(st.session_state.raw_peak_df.head(10), width='stretch')
            st.write("**Metadata:**")
            st.dataframe(st.session_state.meta, width='stretch')
            if st.session_state.get("row_annotations") is not None:
                st.write("**Row Annotations:**")
                st.dataframe(st.session_state.row_annotations.head(10), width='stretch')

    # =======================================================================
    # MULTI-DATASET MODE (multiple methods/modes, combined downstream)
    # =======================================================================
    else:
        st.caption(
            "Each dataset (method/ionization mode) is cleaned, QC'd, and normalized "
            "**independently** in the tabs that follow — their raw intensity scales and "
            "technical characteristics differ and shouldn't be mixed before normalization. "
            "Once every dataset is normalized, click **🔗 Generate Combined Normalized Data** "
            "in the Normalization tab — combination only ever happens on already-normalized "
            "data, never raw data, and never automatically. That combined table then becomes "
            "the single input for every downstream tab (PCA, Statistics, Volcano, Biomarker "
            "Discovery, Heatmap, Boxplot)."
        )

        data_source = st.radio(
            "Data source", ["Demo Data", "Real Data"], horizontal=True, key="multi_data_source"
        )

        st.subheader("Metabolite Row Annotations (optional)")
        st.caption("Columns: Metabolite, then any annotation columns (e.g. Method, Pathway) — enables "
                   "the Heatmap tab's row annotation tracks, same as single-dataset mode. Since combined "
                   "features are prefixed by dataset (e.g. `Targeted Metabolomics::Glucose`), matching "
                   "tries the full prefixed name first, then falls back to matching just the part after "
                   "`::` — so a plain annotation file (Metabolite names with no dataset prefix) still "
                   "matches correctly against the combined data.")
        multi_row_annot_file = st.file_uploader("Upload row annotation table (CSV/XLSX)",
                                                 type=["csv", "xlsx"], key="multi_row_annot_upload")

        # ===================================================================
        # MULTIPLE DATASETS — DEMO DATA
        # ===================================================================
        if data_source == "Demo Data":
            st.caption(
                "Loads 4 built-in demo datasets — **one per analysis type** (Untargeted "
                "Metabolomics, Targeted Metabolomics, Untargeted Lipidomics, Targeted "
                "Lipidomics) — sharing 24 biological samples, so you can explore the full "
                "multi-dataset workflow immediately."
            )
            if st.button("Load Demo Datasets", type="primary"):
                meta_path = os.path.join(os.path.dirname(__file__), "sample_metadata_multimethod.csv")
                meta_raw = pd.read_csv(meta_path)
                demo_types = ["Untargeted Metabolomics", "Targeted Metabolomics",
                              "Untargeted Lipidomics", "Targeted Lipidomics"]
                datasets = {}
                first_peak_df = None
                for dtype in demo_types:
                    prefix = dtype.lower().replace(" ", "_")
                    peak_path = os.path.join(
                        os.path.dirname(__file__),
                        f"sample_peak_area_matrix_multimethod_{prefix}.csv"
                    )
                    peak_raw = pd.read_csv(peak_path)
                    peak_df = utils.validate_peak_matrix(peak_raw)
                    if first_peak_df is None:
                        first_peak_df = peak_df
                    entry = dataset_manager.make_dataset_entry(dtype, dtype)
                    entry["raw_df"] = peak_df
                    datasets[dtype] = entry
                meta = utils.validate_metadata(meta_raw, first_peak_df.columns)
                qc_cols, sample_cols = utils.split_qc_and_samples(first_peak_df, meta)
                for entry in datasets.values():
                    entry["qc_cols"] = qc_cols
                    entry["sample_cols"] = sample_cols
                st.session_state.datasets = datasets
                st.session_state.meta = meta
                st.session_state.qc_cols = qc_cols
                st.session_state.sample_cols = sample_cols
                st.session_state.combined_done = False
                st.session_state.log2_data = None
                row_annotations = None
                if multi_row_annot_file is not None:
                    row_annot_raw = utils.load_table(multi_row_annot_file)
                    if "Metabolite" not in row_annot_raw.columns:
                        st.warning("Row annotation file has no 'Metabolite' column — ignoring it.")
                    else:
                        row_annotations = row_annot_raw.set_index("Metabolite")
                st.session_state.row_annotations = row_annotations
                reset_downstream_analysis_state()
                st.session_state.processing_notes = [
                    f"Multi-dataset mode (demo): {len(datasets)} datasets loaded "
                    f"({', '.join(datasets.keys())}), sharing {len(sample_cols)} biological samples."
                ]
                st.success(f"Loaded {len(datasets)} demo datasets sharing {len(sample_cols)} biological "
                           f"samples across {meta['Group'].nunique() - 1} groups."
                           + (f" Row annotations: {', '.join(row_annotations.columns)}."
                              if row_annotations is not None else ""))

        # ===================================================================
        # MULTIPLE DATASETS — REAL DATA
        # ===================================================================
        else:
            st.subheader("Upload Your Datasets")
            n_datasets = st.number_input("Number of datasets", min_value=1, max_value=8, value=4, step=1)
            st.caption("Typical: 2 methods × 2 ionization modes = 4. Add more for additional methods.")

            upload_entries = []
            for i in range(int(n_datasets)):
                with st.expander(f"Dataset {i+1}", expanded=(i < 2)):
                    c1, c2 = st.columns(2)
                    label = c1.text_input(f"Dataset {i+1} label", value=f"Method{i+1}", key=f"ds_label_{i}")
                    dtype = c2.selectbox(f"Analysis type", ["Untargeted Metabolomics", "Targeted Metabolomics",
                                                             "Untargeted Lipidomics", "Targeted Lipidomics"],
                                          key=f"ds_type_{i}")
                    ds_file = st.file_uploader(f"Peak area matrix for {label}", type=["csv", "xlsx"], key=f"ds_file_{i}")
                    upload_entries.append((label, dtype, ds_file))

            st.subheader("Sample Metadata")
            st.caption(
                "One metadata table applies to ALL datasets above — the same biological samples "
                "(by name) must appear in every peak area matrix you uploaded. Columns: Sample, "
                "Group, IsQC (True/False), Batch (optional)."
            )
            shared_meta_file = st.file_uploader("Upload shared metadata table (CSV/XLSX)",
                                                 type=["csv", "xlsx"], key="shared_meta_upload")

            if st.button("Load All Datasets", type="primary"):
                if shared_meta_file is None:
                    st.error("Please upload the shared metadata table above — it's required to "
                             "identify QC vs. biological samples and group assignments for every "
                             "dataset.")
                    st.stop()
                try:
                    meta_raw = utils.load_table(shared_meta_file)
                    datasets = {}
                    first_peak_df = None
                    for label, dtype, ds_file in upload_entries:
                        if ds_file is None:
                            st.warning(f"Skipping '{label}' — no file uploaded.")
                            continue
                        peak_raw = utils.load_table(ds_file)
                        peak_df = utils.validate_peak_matrix(peak_raw)
                        if first_peak_df is None:
                            first_peak_df = peak_df
                        entry = dataset_manager.make_dataset_entry(label, dtype)
                        entry["raw_df"] = peak_df
                        datasets[label] = entry

                    if not datasets:
                        st.error("No datasets were uploaded.")
                        st.stop()

                    meta = utils.validate_metadata(meta_raw, first_peak_df.columns)
                    warnings = dataset_manager.validate_shared_samples(datasets, meta)
                    for w in warnings:
                        st.warning(w)

                    qc_cols, sample_cols = utils.split_qc_and_samples(first_peak_df, meta)
                    for entry in datasets.values():
                        entry["qc_cols"] = [c for c in qc_cols if c in entry["raw_df"].columns]
                        entry["sample_cols"] = [c for c in sample_cols if c in entry["raw_df"].columns]

                    st.session_state.datasets = datasets
                    st.session_state.meta = meta
                    st.session_state.qc_cols = qc_cols
                    st.session_state.sample_cols = sample_cols
                    st.session_state.combined_done = False
                    st.session_state.log2_data = None
                    row_annotations = None
                    if multi_row_annot_file is not None:
                        row_annot_raw = utils.load_table(multi_row_annot_file)
                        if "Metabolite" not in row_annot_raw.columns:
                            st.warning("Row annotation file has no 'Metabolite' column — ignoring it.")
                        else:
                            row_annotations = row_annot_raw.set_index("Metabolite")
                    st.session_state.row_annotations = row_annotations
                    reset_downstream_analysis_state()
                    st.session_state.processing_notes = [
                        f"Multi-dataset mode (real data): {len(datasets)} datasets loaded "
                        f"({', '.join(datasets.keys())}), sharing {len(sample_cols)} biological samples."
                    ]
                    st.success(f"Loaded {len(datasets)} datasets."
                               + (f" Row annotations: {', '.join(row_annotations.columns)}."
                                  if row_annotations is not None else ""))
                except utils.DataValidationError as e:
                    st.error(f"Validation error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error loading data: {e}")

        if st.session_state.datasets:
            st.subheader("Loaded Datasets")
            summary_rows = [{"Label": ds["label"], "Analysis Type": ds["data_type"],
                             "Features": ds["raw_df"].shape[0] if ds["raw_df"] is not None else 0,
                             "Samples": ds["raw_df"].shape[1] if ds["raw_df"] is not None else 0}
                            for ds in st.session_state.datasets.values()]
            st.dataframe(pd.DataFrame(summary_rows), width='stretch')
            st.write("**Shared Metadata:**")
            st.dataframe(st.session_state.meta, width='stretch')

            st.divider()
            st.subheader("🚀 Quick Start")
            already_processed = dataset_manager.all_datasets_ready(st.session_state.datasets)
            if already_processed:
                st.success(
                    "All datasets are cleaned, QC'd, and normalized — head to **Tab 6 → 🔗 "
                    "Generate Combined Normalized Data** to combine them. That combined table "
                    "is required before any downstream tab (PCA, Statistics, Volcano, Biomarker "
                    "Discovery, Heatmap, Boxplot) can be used."
                )
            st.caption(
                "Applies the recommended defaults to **every** loaded dataset independently in one "
                "click — missingness filter (50%) → Half-Minimum imputation → QC CV filter (20%) → "
                "ISTD/Median-IQR normalization → Log2 — instead of stepping through Tabs 2-4 for each "
                "dataset by hand. This does **not** combine them — combining is always a separate, "
                "explicit step (Tab 6 → Generate Combined Normalized Data), required before any "
                "downstream analysis. You can still fine-tune any individual dataset afterward in "
                "Tabs 2-4 — re-running there overwrites just that one (re-combine afterward)."
            )
            if st.button("🚀 Auto-Process All Datasets (recommended defaults)", type="primary"):
                with st.spinner("Cleaning, QC'ing, and normalizing every dataset independently..."):
                    notes = dataset_manager.auto_process_all_datasets(st.session_state.datasets)
                st.session_state.processing_notes = (
                    [f"Multi-dataset mode: {len(st.session_state.datasets)} datasets loaded "
                     f"({', '.join(ds['label'] for ds in st.session_state.datasets.values())}), "
                     f"sharing {len(st.session_state.sample_cols)} biological samples."]
                    + notes
                )
                st.success(
                    f"All {len(st.session_state.datasets)} datasets cleaned, QC'd, and normalized "
                    "independently. Head to **Tab 6 → 🔗 Generate Combined Normalized Data** to "
                    "combine them — required before PCA, Statistics, or any other downstream tab."
                )
                st.dataframe(dataset_manager.dataset_status_table(st.session_state.datasets),
                             width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Reusable QC Validation UI, shared between single-dataset mode and each
# dataset in multi-dataset mode.
# ---------------------------------------------------------------------------
def render_qc_ui(peak_df, qc_cols, sample_cols, key_prefix, state):
    if len(qc_cols) < 2:
        st.warning("At least 2 QC samples are required for CV calculation. Skipping QC module.")
        return

    cv_table = qc.calculate_cv(peak_df[qc_cols])
    state["cv_table"] = cv_table
    n_accept = (cv_table["Quality"] == "Acceptable").sum()
    n_var = (cv_table["Quality"] == "Variable").sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Features", len(cv_table))
    m2.metric("Acceptable (CV≤20%)", n_accept)
    m3.metric("Variable (CV>20%)", n_var)

    st.markdown("**QC CV Distribution**")
    fig_cv_dist = qc.cv_distribution_plot(cv_table)
    st.pyplot(fig_cv_dist)
    dataset_manager.render_figure_download(st, fig_cv_dist, f"{key_prefix}_QC_CV_Distribution",
                                            key_prefix=f"{key_prefix}_cvdist")

    st.markdown("**Feature Counts by CV Quality**")
    fig_cv_hist = qc.cv_histogram(cv_table)
    st.pyplot(fig_cv_hist)
    dataset_manager.render_figure_download(st, fig_cv_hist, f"{key_prefix}_Feature_Counts_by_Quality",
                                            key_prefix=f"{key_prefix}_cvhist")

    st.markdown("**Peak Area Distribution**")
    st.caption("Raw peak-area values across all features and samples (QC + biological), log10 scale.")
    fig_peak_hist = qc.peak_area_histogram(peak_df, title=f"{key_prefix} — Peak Area Distribution")
    st.pyplot(fig_peak_hist)
    dataset_manager.render_figure_download(st, fig_peak_hist, f"{key_prefix}_Peak_Area_Histogram",
                                            key_prefix=f"{key_prefix}_peakhist")

    st.subheader("CV Filtering Table")
    st.dataframe(cv_table, width='stretch', height=250)
    st.download_button(
        f"Download {key_prefix.replace(' ', '_')}_CV_Table.csv",
        utils.to_download_bytes_csv(cv_table),
        file_name=f"{key_prefix.replace(' ', '_')}_CV_Table.csv", mime="text/csv",
        key=f"{key_prefix}_dl_cv_table"
    )

    filter_cv = st.checkbox(
        "Filter out 'Variable' features (CV>20%) from downstream analysis",
        value=False, key=f"{key_prefix}_filter_cv"
    )
    st.caption(
        "Unchecked by default — **all features are kept** unless you explicitly opt in to "
        "removing high-CV features before normalization/statistics."
    )

    st.subheader("QC Visualization")
    st.caption(
        "Shown below using **QC replicates only** — biological samples are excluded, since this "
        "module evaluates analytical reproducibility and instrument stability, not biology."
    )
    qc_log = normalization.log2_transform(peak_df[qc_cols])
    state["qc_log_data"] = qc_log
    fig_corr, corr_df = qc.sample_correlation_matrix(qc_log)
    st.pyplot(fig_corr)
    dataset_manager.render_figure_download(st, fig_corr, f"{key_prefix}_QC_Sample_Correlation",
                                            key_prefix=f"{key_prefix}_corr")

    if st.button("Confirm QC & Proceed", key=f"{key_prefix}_confirm_qc"):
        if state.get("imputed_data") is not None:
            working_df = state["imputed_data"].copy()
            source_desc = "cleaned & imputed data from Tab 2"
        else:
            working_df = peak_df[sample_cols].copy()
            source_desc = "raw biological samples (no cleaning/imputation was applied in Tab 2)"
        if filter_cv:
            keep = cv_table.index[cv_table["Quality"] == "Acceptable"]
            working_df = working_df.loc[working_df.index.intersection(keep)]
        state["raw_peak_df_qc"] = working_df
        state["qc_figs"] = {"QC CV Distribution": fig_cv_dist, "QC Sample Correlation Matrix": fig_corr}
        state["processing_notes"].append(
            f"QC validation: {len(cv_table)} features assessed via {len(qc_cols)} QC replicates; "
            f"{'variable features (CV>20%) removed' if filter_cv else 'no CV-based filtering applied — all features kept'}, "
            f"applied on top of {source_desc}. "
            f"{len(qc_cols)} QC samples excluded from all downstream normalization/statistics/visualization "
            f"({working_df.shape[0]} features, {len(sample_cols)} biological samples proceed)."
        )
        st.success(
            f"QC step confirmed — {len(qc_cols)} QC samples excluded going forward. "
            f"Proceed to Normalization with {working_df.shape[0]} features across "
            f"{len(sample_cols)} biological samples."
        )


# ===========================================================================
# TAB 3 — QC VALIDATION
# ===========================================================================
with TABS[2]:
    st.header("Quality Control (QC) Validation")
    if st.session_state.data_mode == "single":
        if st.session_state.raw_peak_df is None:
            st.warning("Load data in Tab 1 first.")
        else:
            render_qc_ui(st.session_state.raw_peak_df, st.session_state.qc_cols,
                         st.session_state.sample_cols, "single", _SingleDatasetState())
    else:
        if not st.session_state.datasets:
            st.warning("Load datasets in Tab 1 first.")
        else:
            selected_id = render_dataset_dropdown(st.session_state.datasets, "qc", "2. QC")
            ds = st.session_state.datasets[selected_id]
            st.divider()
            st.subheader(f"QC for {ds['label']} ({ds['data_type']})")
            render_qc_ui(ds["raw_df"], ds["qc_cols"], ds["sample_cols"], selected_id, ds)
            st.caption("See **Tab 4 — Combined QC** for QC diagnostics pooled across all datasets.")

# ===========================================================================
# TAB 4 — COMBINED QC
# ===========================================================================
with TABS[3]:
    st.header("Combined QC — Across All Datasets")
    if st.session_state.data_mode == "single":
        st.info("Combined QC only applies in multi-dataset mode (Tab 1). In single-dataset mode, "
                "see Tab 3 for QC diagnostics.")
    elif not st.session_state.datasets:
        st.warning("Load datasets in Tab 1 first.")
    else:
        st.subheader("QC Comparison Across All Datasets")
        st.caption(
            "Once you've reviewed each dataset's CV table in Tab 3, this compares "
            "analytical reproducibility side by side."
        )
        comparison_table = dataset_manager.qc_comparison_table(st.session_state.datasets)
        if comparison_table.empty:
            st.info("Run QC on at least one dataset in Tab 3 to populate this comparison.")
        else:
            st.dataframe(comparison_table, width='stretch')
            fig_compare = dataset_manager.qc_comparison_plot(st.session_state.datasets)
            if fig_compare is not None:
                st.pyplot(fig_compare)
                dataset_manager.render_figure_download(st, fig_compare, "QC_Comparison_Across_Datasets",
                                                        key_prefix="qc_compare")

        st.divider()
        st.subheader("Combined QC Analysis")
        st.caption(
            "Each dataset's per-feature CV table (and, for the correlation matrix, its "
            "QC-only log2 data) is concatenated into one combined table, with metabolite "
            "names prefixed by dataset label — the same analyses computed per-dataset "
            "in Tab 3, now on all datasets pooled together."
        )
        combined_cv = dataset_manager.combine_cv_tables(st.session_state.datasets)
        if combined_cv is None:
            st.info("Run QC on at least one dataset in Tab 3 to populate the combined analysis.")
        else:
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Total Features (all datasets)", len(combined_cv))
            cc2.metric("Acceptable (CV≤20%)", int((combined_cv["Quality"] == "Acceptable").sum()))
            cc3.metric("Variable (CV>20%)", int((combined_cv["Quality"] == "Variable").sum()))

            st.markdown("**QC CV Distribution**")
            fig_cv_dist_c = qc.cv_distribution_plot(combined_cv)
            st.pyplot(fig_cv_dist_c)
            st.download_button(
                "Download Combined_QC_CV_Table.csv", utils.to_download_bytes_csv(combined_cv),
                file_name="Combined_QC_CV_Table.csv", mime="text/csv", key="dl_combined_cv_dist_data"
            )
            dataset_manager.render_figure_download(st, fig_cv_dist_c, "Combined_QC_CV_Distribution",
                                                    key_prefix="combined_cvdist")

            st.markdown("**Feature Counts by CV Quality**")
            fig_cv_hist_c = qc.cv_histogram(combined_cv)
            st.pyplot(fig_cv_hist_c)
            quality_counts = combined_cv["Quality"].value_counts().rename_axis("Quality").reset_index(name="Count")
            st.download_button(
                "Download Combined_Feature_Counts_by_Quality.csv",
                utils.to_download_bytes_csv(quality_counts),
                file_name="Combined_Feature_Counts_by_Quality.csv", mime="text/csv",
                key="dl_combined_cv_hist_data"
            )
            dataset_manager.render_figure_download(st, fig_cv_hist_c, "Combined_Feature_Counts_by_Quality",
                                                    key_prefix="combined_cvhist")

            st.markdown("**QC Sample Correlation Matrix**")
            combined_qc_log = dataset_manager.combine_qc_log_data(
                st.session_state.datasets, st.session_state.qc_cols
            )
            if combined_qc_log is None:
                st.info("Run QC on at least one dataset in Tab 3 to populate the correlation matrix.")
            else:
                fig_corr_c, corr_df_c = qc.sample_correlation_matrix(combined_qc_log)
                st.pyplot(fig_corr_c)
                st.download_button(
                    "Download Combined_QC_Sample_Correlation.csv",
                    utils.to_download_bytes_csv(corr_df_c),
                    file_name="Combined_QC_Sample_Correlation.csv", mime="text/csv",
                    key="dl_combined_corr_data"
                )
                dataset_manager.render_figure_download(st, fig_corr_c, "Combined_QC_Sample_Correlation",
                                                        key_prefix="combined_corr")

            st.divider()
            st.markdown("**Combined Peak Area Distribution**")
            st.caption(
                "Raw peak-area values across all features, samples, AND datasets pooled together "
                "(computed fresh for this plot — not a stored/downloadable combined peak area "
                "table, since combining happens after normalization, not before)."
            )
            combined_imputed, _ = dataset_manager.combine_imputed_datasets(
                st.session_state.datasets, st.session_state.sample_cols
            )
            if combined_imputed is None:
                st.info("Run Cleaning & Imputation on at least one dataset (Tab 2) first.")
            else:
                fig_peak_hist_c = qc.peak_area_histogram(
                    combined_imputed, title="Combined — Peak Area Distribution (all datasets)"
                )
                st.pyplot(fig_peak_hist_c)
                dataset_manager.render_figure_download(st, fig_peak_hist_c, "Combined_Peak_Area_Histogram",
                                                        key_prefix="combined_peakhist")

# ===========================================================================
# ---------------------------------------------------------------------------
# Reusable Data Cleaning & Imputation UI, shared between single-dataset mode
# and each dataset in multi-dataset mode.
# ---------------------------------------------------------------------------
def render_cleaning_ui(source_df, key_prefix, state):
    treat_zero_as_missing = st.checkbox(
        "Treat exact-zero values as missing (common LC-MS convention for 'not detected')",
        value=True, key=f"{key_prefix}_treat_zero"
    )

    st.subheader("Step 1: Assess & Filter by Missingness")
    st.caption(
        "Guidance: **<20%** missing → keep as-is · **20-50%** → keep, impute carefully · "
        "**50-70%** → usually remove · **>70-80%** → remove unless biologically essential."
    )
    missingness_table = imputation_module.compute_missingness(source_df, treat_zero_as_missing)
    state["missingness_table"] = missingness_table

    m1, m2, m3, m4 = st.columns(4)
    counts = missingness_table["Quality"].value_counts()
    m1.metric("Keep (<20%)", int(counts.get("Keep", 0)))
    m2.metric("Careful imputation (20-50%)", int(counts.get("Keep (careful imputation)", 0)))
    m3.metric("Usually remove (50-70%)", int(counts.get("Usually remove", 0)))
    m4.metric("Remove unless essential (>70%)", int(counts.get("Remove unless essential", 0)))

    fig_miss, ax_miss = _plt.subplots(figsize=(7, 3.5))
    ax_miss.hist(missingness_table["Pct_Missing"], bins=30, color="#4C72B0", edgecolor="white")
    for x, lbl in [(20, "20%"), (50, "50%"), (70, "70%")]:
        ax_miss.axvline(x, color="red", linestyle="--", lw=0.8)
    ax_miss.set_xlabel("% Missing")
    ax_miss.set_ylabel("Number of features")
    ax_miss.set_title("Missingness Distribution Across Features")
    st.pyplot(fig_miss)

    with st.expander("View missingness table"):
        st.dataframe(missingness_table, width='stretch', height=250)

    max_pct_missing = st.slider(
        "Remove features with missingness above this threshold (%)",
        min_value=0, max_value=100, value=50, step=5, key=f"{key_prefix}_max_pct"
    )
    cleaned_preview = imputation_module.filter_by_missingness(source_df, missingness_table, max_pct_missing)
    st.caption(f"At this threshold: **{cleaned_preview.shape[0]} of {source_df.shape[0]}** "
               f"features retained ({source_df.shape[0] - cleaned_preview.shape[0]} removed).")

    if st.button("Apply Missingness Filter", type="primary", key=f"{key_prefix}_apply_filter"):
        state["cleaned_data"] = cleaned_preview
        state["processing_notes"].append(
            f"Missingness filtering: removed {source_df.shape[0] - cleaned_preview.shape[0]} of "
            f"{source_df.shape[0]} features exceeding {max_pct_missing}% missingness; "
            f"{cleaned_preview.shape[0]} features retained."
        )
        st.success(f"Filter applied — {cleaned_preview.shape[0]} features retained.")

    if state.get("cleaned_data") is not None:
        remaining_missing = imputation_module.count_missing(state["cleaned_data"], treat_zero_as_missing)

        st.subheader("Step 2: Choose an Imputation Method")
        if remaining_missing == 0:
            st.success("No missing values remain in the filtered data — imputation isn't needed. "
                       "You can skip ahead.")
            state["imputed_data"] = state["cleaned_data"]
        else:
            st.caption(f"{remaining_missing} missing values remain across the filtered features.")
            method = st.selectbox("Imputation method", list(imputation_module.METHOD_INFO.keys()),
                                   key=f"{key_prefix}_method")
            st.caption(imputation_module.METHOD_INFO[method])

            kwargs = {}
            if method == "K-Nearest Neighbors (KNN)":
                kwargs["n_neighbors"] = st.slider("Number of neighbors (k)", 2, 15, 5, key=f"{key_prefix}_knn_k")
            elif method == "Random Forest (MissForest-style)":
                c1, c2 = st.columns(2)
                kwargs["n_estimators"] = c1.slider("Number of trees", 5, 50, 10, key=f"{key_prefix}_rf_n")
                kwargs["max_iter"] = c2.slider("Iterations", 1, 10, 3, key=f"{key_prefix}_rf_iter")
            elif method == "BPCA (approximated)":
                kwargs["max_iter"] = st.slider("Iterations", 1, 15, 5, key=f"{key_prefix}_bpca_iter")

            if st.button("Apply Imputation", type="primary", key=f"{key_prefix}_apply_impute"):
                with st.spinner(f"Running {method}... this may take a moment for Random Forest or BPCA."):
                    imputed = imputation_module.impute(
                        state["cleaned_data"], method, treat_zero_as_missing=treat_zero_as_missing, **kwargs
                    )
                state["imputed_data"] = imputed
                state["imputation_method_used"] = method
                state["processing_notes"].append(
                    f"Missing value imputation: {method} applied to "
                    f"{state['cleaned_data'].shape[0]} features ({remaining_missing} missing values filled)."
                )
                st.success(f"Imputation complete using {method}.")

        if state.get("imputed_data") is not None:
            st.subheader("Result: Complete Post-Imputation Data")
            st.caption(f"This cleaned & imputed dataset ({state['imputed_data'].shape[0]} features × "
                       f"{state['imputed_data'].shape[1]} samples) now feeds QC validation and "
                       "normalization. Full table (scroll to see all rows):")
            st.dataframe(state["imputed_data"], width='stretch', height=400)
            imputed_filename = (f"{key_prefix.replace(' ', '_')}_Imputed_Data.csv"
                                 if key_prefix != "single" else "Imputed_Data.csv")
            st.download_button(
                f"Download {imputed_filename}",
                utils.to_download_bytes_csv(state["imputed_data"]),
                file_name=imputed_filename, mime="text/csv", key=f"{key_prefix}_dl_imputed"
            )


# ===========================================================================
# TAB 2 — DATA CLEANING & MISSING VALUE IMPUTATION
# ===========================================================================
with TABS[1]:
    st.header("Data Cleaning & Missing Value Imputation")
    st.caption(
        "Optional, but recommended if your data has missing values (encoded as blank/NaN or "
        "zero, both common for 'not detected' in LC-MS peak areas). Runs on the biological "
        "samples (QC replicates are never included here), before QC validation and normalization."
    )
    if st.session_state.data_mode == "single":
        if st.session_state.raw_peak_df is None:
            st.warning("Load data in Tab 1 first.")
        else:
            source_df = st.session_state.raw_peak_df[st.session_state.sample_cols]
            render_cleaning_ui(source_df, "single", _SingleDatasetState())
    else:
        if not st.session_state.datasets:
            st.warning("Load datasets in Tab 1 first.")
        else:
            selected_id = render_dataset_dropdown(st.session_state.datasets, "clean", "1. Cleaning")
            ds = st.session_state.datasets[selected_id]
            st.divider()
            st.subheader(f"Configuring: {ds['label']} ({ds['data_type']})")
            source_df = ds["raw_df"][ds["sample_cols"]]
            render_cleaning_ui(source_df, selected_id, ds)

# ===========================================================================
# ---------------------------------------------------------------------------
# Reusable Normalization UI, shared between single-dataset mode and each
# dataset in multi-dataset mode.
# ---------------------------------------------------------------------------
def render_normalization_ui(base_df, data_type, key_prefix, state, meta):
    is_targeted = "Targeted" in data_type

    if is_targeted:
        st.subheader(f"{data_type}: ISTD Normalization → Log2 Pipeline")
    else:
        st.subheader(f"{data_type}: Log2 → Median-IQR Normalization Pipeline")

    override = st.checkbox(
        "Advanced: override the recommended pipeline for this analysis type", value=False,
        key=f"{key_prefix}_override",
        help="By default, Targeted assays use ISTD normalization and Untargeted assays use "
             "Median-IQR robust scaling, matching standard practice. Check this to access both."
    )

    if is_targeted or override:
        st.markdown("**Step 1: ISTD Normalization**")
        st.caption(
            "Formula: **Normalized Peak Area = Endogenous Metabolite Peak Area ÷ ISTD Peak Area**"
            + ("" if is_targeted else " — optional since this is an untargeted workflow.")
        )
        use_istd = st.checkbox("Apply internal standard (ISTD) normalization",
                                value=is_targeted, key=f"{key_prefix}_use_istd")
        working = base_df
        if use_istd:
            istd_choice = st.selectbox("Select internal standard feature", base_df.index.tolist(),
                                        key=f"{key_prefix}_istd_choice")
            if st.button("Apply ISTD Normalization", key=f"{key_prefix}_apply_istd"):
                try:
                    working = normalization.istd_normalize(base_df, istd_choice)
                    state["istd_normalized"] = working
                    state["processing_notes"].append(f"ISTD normalization applied using '{istd_choice}'.")
                    st.success(f"ISTD normalization complete using {istd_choice}.")
                except Exception as e:
                    st.error(str(e))
        working = state.get("istd_normalized") if state.get("istd_normalized") is not None else base_df

        st.markdown("**Step 2: Log2 Transformation**")
        st.caption(
            "Formula: **Log2(x)** — strict log2 transformation of the "
            "ISTD-normalized (endogenous ÷ ISTD) ratio. No pseudocount, "
            "constant addition, zero replacement, or shifting is applied. "
            "Values ≤ 0 are converted to NaN."
        )
        if st.button("Apply Log2 Transformation", key=f"{key_prefix}_log2_targeted_btn"):
            log2_df = normalization.log2_transform(working)
            state["log2_data"] = log2_df
            n_nonpositive = int((working <= 0).sum().sum())
            state["processing_notes"].append(
                f"Strict Log2 transformation applied to ISTD-normalized ratio "
                f"(log2(x)); {n_nonpositive} values ≤ 0 were converted to NaN."
            )
            st.success(
                f"Strict Log2 transformation complete. "
                f"{n_nonpositive} values ≤ 0 were converted to NaN."
            )
            fig_dist = normalization.distribution_plots(working, log2_df)
            st.pyplot(fig_dist)

    if (not is_targeted) or override:
        st.markdown("**Step 1: Median-IQR Normalization (Robust Scaling)**")
        st.caption(
            "Formula: **X_norm = (X − Median(X)) ÷ IQR(X)**, computed per metabolite across "
            "samples (feature-based), applied to the **raw (QC-validated) peak areas**. "
            "Robust scaling is less sensitive to outliers than mean/SD scaling — the standard "
            "normalization for untargeted metabolomics/lipidomics, where no single ISTD applies "
            "across all features."
        )
        iqr_axis = st.radio("Normalization axis", ["feature", "sample", "batch"], horizontal=True,
                             index=0, key=f"{key_prefix}_iqr_axis")
        if st.button("Apply Median-IQR Normalization", key=f"{key_prefix}_iqr_untargeted_btn"):
            batch_map = meta["Batch"] if iqr_axis == "batch" else None
            scaled = normalization.iqr_normalize(base_df, axis=iqr_axis, batch_map=batch_map)
            state["iqr_normalized"] = scaled
            state["processing_notes"].append(
                f"Median-IQR robust scaling applied to raw abundance ({iqr_axis}-based)."
            )
            st.success(f"Median-IQR normalization complete ({iqr_axis}-based).")

        st.markdown("**Step 2: Log2 Transformation**")
        st.caption(
            "Formula: **Log2(x)** — strict log2 transformation of the "
            "Median-IQR normalized values. No pseudocount, constant addition, "
            "zero replacement, or shifting is applied. Values ≤ 0 are "
            "converted to NaN."
        )
        if state.get("iqr_normalized") is None:
            st.info("Run Step 1 (Median-IQR Normalization) first.")
        elif st.button("Apply Log2 Transformation", key=f"{key_prefix}_log2_untargeted_btn"):
            log2_df = normalization.log2_transform(state["iqr_normalized"])
            state["log2_data"] = log2_df
            n_nonpositive = int((state["iqr_normalized"] <= 0).sum().sum())
            state["processing_notes"].append(
                f"Strict Log2 transformation applied to Median-IQR normalized data "
                f"(log2(x)); {n_nonpositive} values ≤ 0 were converted to NaN."
            )
            st.success(
                f"Strict Log2 transformation complete. "
                f"{n_nonpositive} values ≤ 0 were converted to NaN."
            )
            fig_dist = normalization.distribution_plots(
                state["iqr_normalized"], log2_df
            )
            st.pyplot(fig_dist)

    prelog2_data = state.get("istd_normalized") if state.get("istd_normalized") is not None else state.get("iqr_normalized")
    if prelog2_data is not None or state.get("log2_data") is not None:
        st.markdown("**Current Working Matrix**")
        file_prefix = "" if key_prefix == "single" else f"{key_prefix.replace(' ', '_')}_"

        if prelog2_data is not None:
            st.markdown("*Normalized — Without Log2 Transformation*")
            st.dataframe(prelog2_data, width='stretch', height=300)
            st.download_button(
                f"Download {file_prefix}Normalized_NoLog2.csv",
                utils.to_download_bytes_csv(prelog2_data),
                file_name=f"{file_prefix}Normalized_NoLog2.csv", mime="text/csv",
                key=f"{key_prefix}_dl_norm_nolog2"
            )
        else:
            st.info("Run Step 1 (normalization) above to enable the pre-Log2 download.")

        if state.get("log2_data") is not None:
            st.markdown("*Normalized — With Log2 Transformation*")
            st.dataframe(state["log2_data"], width='stretch', height=300)
            st.download_button(
                f"Download {file_prefix}Normalized_Log2.csv",
                utils.to_download_bytes_csv(state["log2_data"]),
                file_name=f"{file_prefix}Normalized_Log2.csv", mime="text/csv",
                key=f"{key_prefix}_dl_norm_log2"
            )
        else:
            st.info("Run Step 2 (Log2 transformation) above to enable the Log2 download.")


# ===========================================================================
# TAB 5 — NORMALIZATION
# ===========================================================================
with TABS[4]:
    if st.session_state.data_mode == "single":
        if st.session_state.raw_peak_df is None:
            st.warning("Load data in Tab 1 first.")
        else:
            if st.session_state.get("raw_peak_df_qc") is not None:
                base_df = st.session_state.raw_peak_df_qc
            else:
                base_df = st.session_state.raw_peak_df[st.session_state.sample_cols]
                st.info("QC tab wasn't confirmed yet — proceeding with biological samples only "
                         "(QC columns excluded automatically). Visit Tab 3 to review QC first.")
            render_normalization_ui(base_df, st.session_state.data_type, "single",
                                     _SingleDatasetState(), st.session_state.meta)
    else:
        if not st.session_state.datasets:
            st.warning("Load datasets in Tab 1 first.")
        else:
            selected_id = render_dataset_dropdown(st.session_state.datasets, "norm", "3. Normalization")
            ds = st.session_state.datasets[selected_id]
            st.divider()
            st.subheader(f"Configuring: {ds['label']} ({ds['data_type']})")
            if ds.get("raw_peak_df_qc") is not None:
                base_df = ds["raw_peak_df_qc"]
            else:
                base_df = ds["raw_df"][ds["sample_cols"]]
                st.info(f"QC tab wasn't confirmed yet for {ds['label']} — proceeding with "
                         "biological samples only.")
            render_normalization_ui(base_df, ds["data_type"], selected_id, ds, st.session_state.meta)
            st.caption("Once every dataset is normalized, go to **Tab 6 — Combined Normalized Data** "
                       "to combine them — required before any downstream analysis tab.")

# ===========================================================================
# TAB 6 — COMBINED NORMALIZED DATA
# ===========================================================================
with TABS[5]:
    st.header("Combined Normalized Data")
    if st.session_state.data_mode == "single":
        st.info("Combining only applies in multi-dataset mode (Tab 1). In single-dataset mode, "
                "Tab 5's normalized data already feeds every downstream tab directly.")
    elif not st.session_state.datasets:
        st.warning("Load datasets in Tab 1 first.")
    else:
        st.caption(
            "This is the **required final step** for multi-dataset analysis: it combines every "
            "dataset's **already-normalized** matrix into one unified table, which then becomes "
            "the single input every downstream tab (PCA, Statistics, Volcano, Biomarker "
            "Discovery, Heatmap, Boxplot) analyzes — never raw or unnormalized data, and never "
            "just one dataset by itself. Metabolite names are prefixed with their dataset label "
            "(e.g. `Targeted Metabolomics::Glucose`) so the same compound name across methods/"
            "ionization modes — a genuinely different measurement each time (different adducts, "
            "different chromatographic behavior) — is never silently merged, and so you can still "
            "tell which dataset each row came from. This works the same way regardless of how "
            "many datasets you loaded — 2, 3, 4, or more."
        )
        summary = dataset_manager.normalization_summary_table(st.session_state.datasets)
        st.dataframe(summary, width='stretch')

        n_ready = (summary["Status"] == "Complete").sum()
        n_total = len(summary)
        if n_ready < n_total:
            st.warning(
                f"⏳ {n_ready} of {n_total} datasets are normalized. **Generate Combined "
                f"Normalized Data** stays disabled until all {n_total} are complete — finish "
                "each remaining dataset's Log2 Transformation step in Tab 5 (select it from the "
                "dropdown), or use the 🚀 Quick Start button in Tab 1 to normalize all of "
                "them at once."
            )
        generate_clicked = st.button(
            "🔗 Generate Combined Normalized Data", type="primary", disabled=(n_ready < n_total)
        )
        if generate_clicked:
            combined, counts = dataset_manager.combine_normalized_datasets(
                st.session_state.datasets, st.session_state.sample_cols, use_log2=True
            )
            combined_prelog2, prelog2_counts = dataset_manager.combine_normalized_datasets(
                st.session_state.datasets, st.session_state.sample_cols, use_log2=False
            )
            st.session_state.log2_data = combined
            st.session_state.combined_normalized_prelog2 = combined_prelog2
            st.session_state.combined_done = True
            reset_downstream_analysis_state()

            # Aggregate each dataset's QC figures into the flat structure downstream
            # exports read from, prefixed by dataset label.
            combined_qc_figs = {}
            for ds2 in st.session_state.datasets.values():
                for fig_name, fig in ds2.get("qc_figs", {}).items():
                    combined_qc_figs[f"{ds2['label']}: {fig_name}"] = fig
            st.session_state.qc_figs = combined_qc_figs

            st.session_state.processing_notes.append(
                f"Generated combined normalized data from {len(counts)} independently-"
                f"normalized datasets into one Log2 matrix ({combined.shape[0]} total "
                "features): " + ", ".join(f"{lbl} ({n})" for lbl, n in counts.items()) + ". "
                f"A pre-Log2 normalized combined matrix "
                f"({combined_prelog2.shape[0] if combined_prelog2 is not None else 0} "
                "total features) was also generated. This combined table is now the single "
                "input for every downstream analysis tab."
            )
            st.success(f"Combined into {combined.shape[0]} total features across "
                       f"{len(counts)} datasets. Every downstream tab (PCA, Statistics, "
                       "Volcano, Biomarker Discovery, Heatmap, Boxplot) now uses this table.")

        if st.session_state.get("combined_done") and st.session_state.log2_data is not None:
            st.markdown("### Combined Normalized Data Table")
            st.markdown("**Normalized Without Log2 Transformation**")
            prelog2 = st.session_state.get("combined_normalized_prelog2")
            if prelog2 is not None:
                st.dataframe(prelog2, width='stretch', height=400)
                st.download_button(
                    "Download Combined_Normalized_NoLog2.csv",
                    utils.to_download_bytes_csv(prelog2),
                    file_name="Combined_Normalized_NoLog2.csv", mime="text/csv",
                    key="dl_combined_norm_nolog2"
                )
            else:
                st.info("No pre-Log2 normalized data available to combine yet.")

            st.markdown("**Normalized With Log2 Transformation**")
            st.dataframe(st.session_state.log2_data, width='stretch', height=400)
            st.download_button(
                "Download Combined_Normalized_Log2.csv",
                utils.to_download_bytes_csv(st.session_state.log2_data),
                file_name="Combined_Normalized_Log2.csv", mime="text/csv",
                key="dl_combined_norm_log2"
            )
            st.caption(
                f"{st.session_state.log2_data.shape[0]} total features (dataset-prefixed) × "
                f"{st.session_state.log2_data.shape[1]} samples. This is the table every "
                "downstream tab now uses — head to **Tab 7 (PCA)** to continue."
            )

# ===========================================================================
# TAB 8 — STATISTICS
# ===========================================================================
with TABS[7]:
    st.header("Statistical Comparison")
    if st.session_state.log2_data is None:
        st.warning(
            "Complete the Normalization tab (Log2 transform) first."
            if st.session_state.data_mode == "single" else
            "Complete Normalization for every dataset, then click **🔗 Generate Combined "
            "Normalized Data** in Tab 6 first — that combined table is what every "
            "downstream tab (PCA, Statistics, Volcano, Biomarker Discovery, Heatmap, "
            "Boxplot) analyzes."
        )
    else:
        if st.session_state.data_mode == "multi":
            st.caption("Analyzing the **combined normalized data** across all datasets.")
        log2_df = st.session_state.log2_data
        meta = st.session_state.meta
        sample_cols = log2_df.columns.tolist()
        stats_cat_cols = utils.get_categorical_metadata_columns(meta, sample_cols)
        grouping_var = st.selectbox(
            "Grouping variable:", stats_cat_cols,
            index=stats_cat_cols.index("Group") if "Group" in stats_cat_cols else 0,
            key="stats_grouping_var"
        )
        groups_available = meta.loc[meta.index.intersection(sample_cols), grouping_var].unique().tolist()

        st.caption(
            "All statistics below (mean abundance, fold change, p-value, FDR, 95% CI) are computed "
            "from the log2-transformed, normalized data — raw peak areas are not used for inference."
        )

        mode = st.radio("Comparison type", ["Two-group comparison", "ANOVA (≥3 groups)"], horizontal=True)

        if mode == "Two-group comparison":
            c1, c2, c3 = st.columns(3)
            group_a = c1.selectbox("Group A", groups_available, index=0)
            group_b = c2.selectbox("Group B", groups_available, index=min(1, len(groups_available) - 1))
            method = c3.selectbox("Method", ["Student's t-test", "Wilcoxon rank-sum"])
            method_key = "ttest" if method.startswith("Student") else "wilcoxon"

           if st.button("Run Two-Group Test"):

    # --------------------------------------------------------
    # Get samples for Group A and Group B
    # --------------------------------------------------------

    a_samples = meta.index[
        meta[grouping_var] == group_a
    ].tolist()

    b_samples = meta.index[
        meta[grouping_var] == group_b
    ].tolist()

    # Keep only samples present in the normalized log2 matrix
    a_samples = [
        s for s in a_samples
        if s in log2_df.columns
    ]

    b_samples = [
        s for s in b_samples
        if s in log2_df.columns
    ]

    # --------------------------------------------------------
    # Validate sample numbers
    # --------------------------------------------------------

    if len(a_samples) < 2:
        st.error(
            f"Group A ({group_a}) has only {len(a_samples)} "
            "valid samples. At least 2 samples are required."
        )
        st.stop()

    if len(b_samples) < 2:
        st.error(
            f"Group B ({group_b}) has only {len(b_samples)} "
            "valid samples. At least 2 samples are required."
        )
        st.stop()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # log2_df is already the normalized log2-scale dataset.
    #
    # NO additional log2 transformation here.
    #
    # Fold change is calculated inside stats_analysis.py:
    #
    # Log2FC = Mean(Group B) - Mean(Group A)
    #
    # Linear FC = 2 ** Log2FC
    # --------------------------------------------------------

    result = stats_analysis.two_group_test(
        data_log2=log2_df,
        group_a_samples=a_samples,
        group_b_samples=b_samples,
        method=method_key
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    st.session_state.stats_result = result

    st.session_state.stats_result_groups = (
        group_a,
        group_b
    )

    st.session_state.stats_result_group_col = grouping_var

    # --------------------------------------------------------
    # Processing note
    # --------------------------------------------------------

    significant_count = int(
        result["Significant"].sum()
    )

    st.session_state.processing_notes.append(
        f"Statistical comparison ({grouping_var}): "
        f"{group_a} vs {group_b} using {method} "
        f"(BH-FDR correction); "
        f"{significant_count} significant metabolites "
        f"(p<0.05 & FDR<0.25)."
    )

    st.success(
        f"Test complete: {significant_count} "
        "significant metabolites found.")

            if st.session_state.stats_result is not None and st.session_state.stats_result_groups is not None:
                result = st.session_state.stats_result
                g_a, g_b = st.session_state.stats_result_groups
                comparison_name = f"{g_a}_vs_{g_b}"
                st.dataframe(result, width='stretch', height=400)
                st.download_button(
                    f"Download {comparison_name}_Statistics.csv",
                    utils.to_download_bytes_csv(result),
                    file_name=f"{comparison_name}_Statistics.csv", mime="text/csv",
                    key="dl_stats_twogroup"
                )
                st.caption(f"Filename includes the comparison ({comparison_name}) so results from "
                           "different comparisons stay distinguishable.")

        else:
            if len(groups_available) < 3:
                st.warning(f"Need at least 3 values of '{grouping_var}' (excluding QC) for ANOVA. "
                           "Pick a different grouping variable, or add more groups in metadata.")
            else:
                anova_groups = st.multiselect(
                    f"{grouping_var} values to include in ANOVA (select 3 or more)",
                    groups_available, default=groups_available, key="anova_group_select"
                )
                posthoc_method = st.selectbox("Post-hoc test", ["tukey", "dunnett", "pairwise"])

                if len(anova_groups) < 3:
                    st.warning(f"Select at least 3 groups to run ANOVA (currently {len(anova_groups)} selected).")
                elif st.button("Run ANOVA"):
                    group_map = meta.loc[sample_cols, grouping_var]
                    group_map = group_map[group_map.isin(anova_groups)]
                    group_map = group_map[group_map.index.isin(log2_df.columns)]
                    anova_table, posthoc_results = stats_analysis.anova_test(
                        log2_df[group_map.index], group_map, posthoc=posthoc_method
                    )
                    st.session_state.anova_result = anova_table
                    st.session_state.posthoc_result = posthoc_results
                    st.session_state.anova_groups_used = list(anova_groups)
                    st.session_state.stats_result_group_col = grouping_var
                    st.session_state.processing_notes.append(
                        f"One-way ANOVA across {len(anova_groups)} selected {grouping_var} values "
                        f"({', '.join(anova_groups)}) with {posthoc_method} post-hoc; "
                        f"{(anova_table['FDR'] < 0.25).sum()} significant metabolites (FDR<0.25)."
                    )
                    st.success(f"ANOVA complete: {(anova_table['FDR'] < 0.25).sum()} significant metabolites (FDR<0.25).")

                if st.session_state.anova_result is not None:
                    anova_table = st.session_state.anova_result
                    anova_groups_used = st.session_state.get("anova_groups_used") or anova_groups
                    anova_comparison_name = "ANOVA_" + "_vs_".join(anova_groups_used)
                    st.dataframe(anova_table, width='stretch', height=350)
                    st.download_button(
                        f"Download {anova_comparison_name}.csv",
                        utils.to_download_bytes_csv(anova_table),
                        file_name=f"{anova_comparison_name}.csv", mime="text/csv",
                        key="dl_stats_anova"
                    )
                    st.caption(f"Filename includes the groups compared ({', '.join(anova_groups_used)}) "
                               "so results from different ANOVA runs stay distinguishable.")

                    posthoc_results = st.session_state.get("posthoc_result")
                    if posthoc_results:
                        feat_choice = st.selectbox("View post-hoc results for feature:", list(posthoc_results.keys()))
                        st.dataframe(posthoc_results[feat_choice], width='stretch')
                        st.download_button(
                            f"Download {anova_comparison_name}_Posthoc_{feat_choice}.csv",
                            utils.to_download_bytes_csv(posthoc_results[feat_choice]),
                            file_name=f"{anova_comparison_name}_Posthoc_{feat_choice}.csv", mime="text/csv",
                            key="dl_stats_posthoc_one"
                        )
                        all_posthoc = pd.concat(
                            [df.assign(Feature=feat) for feat, df in posthoc_results.items()],
                            ignore_index=True
                        )
                        st.download_button(
                            f"Download {anova_comparison_name}_Posthoc_All_Features.csv",
                            utils.to_download_bytes_csv(all_posthoc),
                            file_name=f"{anova_comparison_name}_Posthoc_All_Features.csv", mime="text/csv",
                            key="dl_stats_posthoc_all"
                        )

# ===========================================================================
# TAB 7 — PCA
# ===========================================================================
with TABS[6]:
    st.header("PCA Visualization (Biological Samples Only)")
    if st.session_state.log2_data is None:
        st.warning(
            "Complete the Normalization tab (Log2 transform) first."
            if st.session_state.data_mode == "single" else
            "Complete Normalization for every dataset, then click **🔗 Generate Combined "
            "Normalized Data** in Tab 6 first — that combined table is what every "
            "downstream tab analyzes."
        )
    else:
        if st.session_state.data_mode == "multi":
            st.caption("Analyzing the **combined normalized data** across all datasets.")
        log2_df_full = st.session_state.log2_data
        meta = st.session_state.meta
        pca_cat_cols = utils.get_categorical_metadata_columns(meta, log2_df_full.columns)
        pca_group_col = st.selectbox(
            "Color/group samples by:", pca_cat_cols,
            index=pca_cat_cols.index("Group") if "Group" in pca_cat_cols else 0,
            key="pca_group_col"
        )
        groups_all_pca = meta.loc[meta.index.intersection(log2_df_full.columns), pca_group_col].unique().tolist()

        st.subheader("Select Groups")
        selected_pca_groups = st.multiselect(
            f"{pca_group_col} values to include in PCA (choose any subset — 2, 3, 4, or more)",
            groups_all_pca, default=groups_all_pca, key="pca_group_select"
        )
        pca_cols = [c for c in log2_df_full.columns if meta.loc[c, pca_group_col] in selected_pca_groups]
        log2_df = log2_df_full[pca_cols]

        if len(selected_pca_groups) < 1 or len(pca_cols) < 3:
            st.warning("Select at least one group with enough samples (≥3 total) to run PCA.")
        else:
            n_comp = st.slider("Number of components", 2, min(10, log2_df.shape[1] - 1 if log2_df.shape[1] > 2 else 2), 5)

            st.subheader("Customize Appearance")
            c1, c2 = st.columns(2)
            palette_choice = c1.selectbox("Color palette", list(pca_module.PALETTES.keys()))
            show_ellipse = c2.checkbox("Show 95% confidence ellipses", value=True)

            st.caption("Optional: assign a marker style per group (defaults to circles for all).")
            marker_map = {}
            marker_cols = st.columns(min(4, len(selected_pca_groups)) or 1)
            for i, g in enumerate(selected_pca_groups):
                with marker_cols[i % len(marker_cols)]:
                    marker_map[g] = st.selectbox(f"Marker: {g}", pca_module.MARKER_STYLES, key=f"marker_{g}")

            pca, scores_df, cols = pca_module.run_pca(log2_df, n_components=n_comp)
            fig_score = pca_module.pca_score_plot(pca, scores_df, meta, palette=palette_choice,
                                                    marker_map=marker_map, show_ellipse=show_ellipse,
                                                    group_col=pca_group_col)
            fig_loading, top_loadings = pca_module.pca_loading_plot(pca, log2_df.index, top_n=20)
            fig_var = pca_module.pca_variance_plot(pca)

            st.session_state.viz_figs["PCA Score Plot"] = fig_score
            st.session_state.viz_figs["PCA Loading Plot"] = fig_loading
            st.session_state.viz_figs["PCA Variance Plot"] = fig_var

            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(fig_score)
                dataset_manager.render_figure_download(st, fig_score, "PCA_Score_Plot", key_prefix="pca_score")
            with c2:
                st.pyplot(fig_loading)
                dataset_manager.render_figure_download(st, fig_loading, "PCA_Loading_Plot", key_prefix="pca_loading")
            st.pyplot(fig_var)
            dataset_manager.render_figure_download(st, fig_var, "PCA_Variance_Plot", key_prefix="pca_var")

            st.subheader("Top Contributing Metabolites (Loadings)")
            st.dataframe(top_loadings, width='stretch')

            st.divider()
            st.subheader("Download PCA Data")
            st.caption(
                "The underlying data behind the plots above: sample scores per principal "
                "component (with group assignment), % variance explained per component, "
                "the metabolite loadings, and the exact normalized data matrix PCA was run on."
            )
            explained_var_df = pd.DataFrame({
                "Component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
                "Variance Explained (%)": (pca.explained_variance_ratio_ * 100).round(3),
                "Cumulative Variance Explained (%)": (np.cumsum(pca.explained_variance_ratio_) * 100).round(3),
            })
            scores_with_group = scores_df.copy()
            scores_with_group.insert(0, pca_group_col, meta.loc[scores_with_group.index, pca_group_col].values)

            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "Download PCA_Scores.csv", utils.to_download_bytes_csv(scores_with_group),
                file_name="PCA_Scores.csv", mime="text/csv", key="dl_pca_scores"
            )
            dl2.download_button(
                "Download PCA_Explained_Variance.csv", utils.to_download_bytes_csv(explained_var_df),
                file_name="PCA_Explained_Variance.csv", mime="text/csv", key="dl_pca_variance"
            )
            dl3, dl4 = st.columns(2)
            dl3.download_button(
                "Download PCA_Loadings.csv", utils.to_download_bytes_csv(top_loadings),
                file_name="PCA_Loadings.csv", mime="text/csv", key="dl_pca_loadings"
            )
            dl4.download_button(
                "Download PCA_Input_Data.csv", utils.to_download_bytes_csv(log2_df),
                file_name="PCA_Input_Data.csv", mime="text/csv", key="dl_pca_input"
            )

# ===========================================================================
# TAB 9 — VOLCANO PLOT
# ===========================================================================
with TABS[8]:
    st.header("Volcano Plot")
    if st.session_state.stats_result is None:
        st.warning("Run a two-group statistical comparison in the Statistics tab first.")
    else:
        result = st.session_state.stats_result
        groups_used = st.session_state.get("stats_result_groups")
        group_col_used = st.session_state.get("stats_result_group_col") or "Group"
        if groups_used:
            st.caption(
                f"Reflects the two {group_col_used} values compared in the Statistics tab: "
                f"**{groups_used[0]}** vs **{groups_used[1]}**"
                + (" (combined normalized data, all datasets)."
                   if st.session_state.data_mode == "multi" else ".")
                + " To compare a different pair or variable, go back to Statistics and "
                "re-run the two-group test."
            )

        # ---- 1. Thresholds ----
        with st.expander("1 — Statistical Thresholds", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            y_metric_label = c1.radio("Y-axis metric", ["p-value", "FDR"], horizontal=False)
            y_metric = "pvalue" if y_metric_label == "p-value" else "fdr"
            sig_cutoff = c2.number_input("Significance cutoff", value=0.05 if y_metric == "pvalue" else 0.25,
                                          min_value=0.0001, max_value=1.0, step=0.01)
            fc_threshold = c3.number_input("Fold change threshold (log2 units)", value=1.0,
                                            min_value=0.0, max_value=5.0, step=0.1)
            use_fdr_secondary = c4.checkbox("Also require FDR <", value=False)
            fdr_cutoff = c4.number_input("Secondary FDR cutoff", value=0.25, min_value=0.0001, max_value=1.0,
                                          step=0.01, disabled=not use_fdr_secondary) if use_fdr_secondary else None
            c5, c6 = st.columns(2)
            use_fc_axis = c5.checkbox("Show linear FC on x-axis instead of log2FC", value=False)
            threshold_line_style = c6.selectbox("Threshold line style", list(volcano.LINE_STYLES.keys()))

        # ---- 2 & 3. Colors and point style ----
        with st.expander("2-3 — Colors & Point Style"):
            c1, c2 = st.columns(2)
            palette = c1.selectbox("Color palette (colorblind-friendly options included)",
                                    list(volcano.COLORBLIND_PALETTES.keys()))
            override_colors = c2.checkbox("Override individual colors", value=False)
            up_color = down_color = ns_color = None
            if override_colors:
                cc1, cc2, cc3 = st.columns(3)
                up_color = cc1.color_picker("Upregulated color", volcano.COLORBLIND_PALETTES[palette]["up"])
                down_color = cc2.color_picker("Downregulated color", volcano.COLORBLIND_PALETTES[palette]["down"])
                ns_color = cc3.color_picker("Not significant color", volcano.COLORBLIND_PALETTES[palette]["ns"])
            c3, c4, c5 = st.columns(3)
            alpha = c3.slider("Point transparency (alpha)", 0.2, 1.0, 0.75, 0.05)
            point_size = c4.slider("Point size", 1, 100, 14, 1)
            point_shape = c5.selectbox("Point shape", list(volcano.MARKER_SHAPES.keys()))
            c6, c7 = st.columns(2)
            edge_width = c6.slider("Point border width", 0.0, 2.0, 0.0, 0.1)
            edge_color = c7.color_picker("Point border color", "#000000") if edge_width > 0 else "none"

        # ---- 4. Labels ----
        with st.expander("4 — Metabolite Labels"):
            label_mode_label = st.radio(
                "Label mode",
                ["Top N significant", "All significant", "Manually selected", "None"],
                horizontal=True
            )
            label_mode = {"Top N significant": "top_n", "All significant": "significant_only",
                          "Manually selected": "manual", "None": "none"}[label_mode_label]
            top_label_n = 10
            manual_labels = []
            if label_mode == "top_n":
                top_label_n = st.number_input("Label top N metabolites", value=10, min_value=0, max_value=100, step=1)
            elif label_mode == "manual":
                manual_labels = st.multiselect("Search and select metabolites to label by name",
                                                result.index.tolist())
                st.caption("Note: labeling by HMDB ID isn't available — this dataset identifies "
                           "metabolites by name only (no compound-database annotation).")
            c1, c2, c3 = st.columns(3)
            label_font_size = c1.slider("Label font size", 5.0, 16.0, 7.5, 0.5)
            label_bold = c2.checkbox("Bold labels", value=False)
            label_italic = c3.checkbox("Italic labels", value=False)
            c4, c5 = st.columns(2)
            override_label_color = c4.checkbox("Override label color (default: match point color)", value=False)
            label_color = c5.color_picker("Label color", "#333333") if override_label_color else None
            repel_labels = st.checkbox("Repel overlapping labels automatically", value=True)

        # ---- 5. Legend ----
        with st.expander("5 — Legend"):
            c1, c2 = st.columns(2)
            legend_position = c1.selectbox("Legend position", ["Right", "Left", "Top", "Bottom", "Hidden"])
            legend_format = c2.radio("Legend label format",
                                      ["Full (\"Upregulated (n=5)\")", "Short (\"Up (5)\")"], horizontal=False)
            legend_format_key = "full" if legend_format.startswith("Full") else "short"

        # ---- 6 & 7. Axes and title ----
        with st.expander("6-7 — Axes & Title"):
            st.markdown("**Axes**")
            c1, c2, c3 = st.columns(3)
            axis_font_size = c1.slider("Axis font size", 8.0, 20.0, 12.0, 0.5)
            axis_bold = c2.checkbox("Bold axis titles", value=False)
            y_decimals = c3.selectbox("Y-axis decimals", [None, 0, 1, 2, 3], index=0,
                                      format_func=lambda x: "Auto" if x is None else str(x))
            c4, c5 = st.columns(2)
            custom_x_limits = c4.checkbox("Set custom X-axis limits", value=False)
            x_limits = None
            if custom_x_limits:
                xlo, xhi = st.columns(2)
                x_limits = (xlo.number_input("X min", value=-6.0), xhi.number_input("X max", value=6.0))
            custom_y_limits = c5.checkbox("Set custom Y-axis limits", value=False)
            y_limits = None
            if custom_y_limits:
                ylo, yhi = st.columns(2)
                y_limits = (ylo.number_input("Y min", value=0.0), yhi.number_input("Y max", value=10.0))
            x_tick_spacing = st.number_input("X-axis tick spacing (0 = auto)", value=0.0, min_value=0.0, step=0.5)
            x_tick_spacing = x_tick_spacing or None

            st.markdown("**Title**")
            c6, c7 = st.columns(2)
            default_title = f"{groups_used[0]} vs {groups_used[1]}" if groups_used else "Volcano Plot"
            custom_title = c6.text_input("Title", value=default_title)
            subtitle = c7.text_input("Subtitle (optional)", value="")
            c8, c9, c10, c11 = st.columns(4)
            title_bold = c8.checkbox("Bold title", value=True)
            title_italic = c9.checkbox("Italic title", value=False)
            title_align = c10.selectbox("Title alignment", ["center", "left", "right"])
            hide_title = c11.checkbox("Hide title", value=False)

        # ---- 8 & 9. Gridlines and threshold line style ----
        with st.expander("8-9 — Gridlines & Threshold Line Style"):
            c1, c2, c3 = st.columns(3)
            grid_mode = c1.selectbox("Gridlines", ["none", "major", "both"],
                                     format_func=lambda x: {"none": "No grid", "major": "Major only",
                                                              "both": "Major + minor"}[x])
            grid_color = c2.color_picker("Grid color", "#D9D9D9")
            grid_style = c3.selectbox("Grid line style", list(volcano.LINE_STYLES.keys()), index=0)
            c4, c5 = st.columns(2)
            threshold_line_color = c4.color_picker("Threshold line color", "#808080")
            threshold_line_width = c5.slider("Threshold line width", 0.2, 3.0, 0.7, 0.1)

        # ---- 10. Highlight specific metabolites ----
        with st.expander("10 — Highlight Specific Metabolites"):
            highlight_names = st.multiselect(
                "Search and select metabolites to highlight (e.g. known markers)",
                result.index.tolist()
            )
            c1, c2, c3 = st.columns(3)
            highlight_color = c1.color_picker("Highlight color", "#FFD700")
            highlight_size_mult = c2.slider("Highlight size multiplier", 1.0, 5.0, 2.0, 0.25)
            highlight_shape = c3.selectbox("Highlight shape", ["Star", "Circle", "Triangle", "Square", "Diamond"])

        # ---- 12 & 13. Background and figure size ----
        with st.expander("12-13 — Background & Figure Size"):
            c1, c2 = st.columns(2)
            background_choice = c1.selectbox("Background", ["White", "Transparent", "Gray", "Custom"])
            background = {"White": "white", "Transparent": "transparent", "Gray": "gray"}.get(background_choice)
            if background_choice == "Custom":
                background = c2.color_picker("Custom background color", "#FFFFFF")
            c3, c4 = st.columns(2)
            size_preset = c3.selectbox("Figure size preset", list(volcano.FIGURE_SIZE_PRESETS.keys()), index=2)
            if size_preset == "Custom":
                cw, ch = st.columns(2)
                fig_width_in = cw.number_input("Width (in)", value=7.2, min_value=2.0, max_value=20.0)
                fig_height_in = ch.number_input("Height (in)", value=6.4, min_value=2.0, max_value=20.0)
            else:
                fig_width_in, fig_height_in = volcano.FIGURE_SIZE_PRESETS[size_preset]

        # ---- 15. Statistics box, 18. Theme ----
        with st.expander("Statistics Display & Publication Theme"):
            c1, c2 = st.columns(2)
            show_stats_box = c1.checkbox("Show statistics box on figure (total/up/down/cutoffs)", value=False)
            theme_choice = c2.selectbox("Publication theme (stylistic approximation, not an official spec)",
                                        list(volcano.THEMES.keys()))

        fig_volc, annotated = volcano.volcano_plot(
            result,
            use_fc_not_log2=use_fc_axis, fc_threshold=fc_threshold, y_metric=y_metric,
            sig_cutoff=sig_cutoff, fdr_cutoff=fdr_cutoff, threshold_line_style=threshold_line_style,
            palette=palette, up_color=up_color, down_color=down_color, ns_color=ns_color,
            alpha=alpha, edge_color=edge_color, edge_width=edge_width,
            point_size=point_size, point_shape=point_shape,
            label_mode=label_mode, top_label_n=top_label_n, manual_labels=manual_labels,
            label_font_size=label_font_size, label_color=label_color, label_bold=label_bold,
            label_italic=label_italic, repel_labels=repel_labels,
            legend_position=legend_position, legend_format=legend_format_key,
            axis_font_size=axis_font_size, axis_bold=axis_bold,
            x_limits=x_limits, y_limits=y_limits, x_tick_spacing=x_tick_spacing, y_decimals=y_decimals,
            title=custom_title, subtitle=subtitle or None, title_bold=title_bold, title_italic=title_italic,
            title_align=title_align, hide_title=hide_title,
            grid_mode=grid_mode, grid_style=grid_style, grid_color=grid_color,
            threshold_line_color=threshold_line_color, threshold_line_width=threshold_line_width,
            highlight_names=highlight_names, highlight_color=highlight_color,
            highlight_size_mult=highlight_size_mult, highlight_shape=highlight_shape,
            background=background, fig_width_in=fig_width_in, fig_height_in=fig_height_in,
            show_stats_box=show_stats_box, theme=theme_choice,
        )
        st.session_state.viz_figs[f"Volcano Plot ({y_metric_label})"] = fig_volc
        st.session_state.volcano_fig = fig_volc
        st.session_state.volcano_annotated = annotated
        st.session_state.volcano_settings = {
            "y_metric": y_metric, "sig_cutoff": sig_cutoff, "fc_threshold": fc_threshold,
            "fdr_cutoff": fdr_cutoff, "palette": palette, "point_size": point_size,
            "point_shape": point_shape, "label_mode": label_mode, "legend_position": legend_position,
            "theme": theme_choice, "groups": groups_used,
        }
        st.pyplot(fig_volc)

        sig_col = "p-value" if y_metric == "pvalue" else "FDR"

        st.subheader("Top 20 Biomarkers")
        st.dataframe(volcano.top_biomarker_labels(annotated, sig_col=sig_col, n=20), width='stretch')

        st.subheader("Export")
        exp_tab1, exp_tab2 = st.tabs(["Figure", "Data & Settings"])

        with exp_tab1:
            ec1, ec2, ec3 = st.columns(3)
            export_fmt = ec1.selectbox("Format", ["PNG", "PDF", "SVG", "EPS", "JPEG", "TIFF"],
                                        key="volcano_export_fmt")
            export_dpi = ec2.selectbox("Resolution (DPI)", [300, 600, 1200], index=0,
                                        disabled=export_fmt in ("PDF", "SVG", "EPS"), key="volcano_export_dpi")
            mime_map = {"PNG": "image/png", "PDF": "application/pdf", "SVG": "image/svg+xml",
                        "EPS": "application/postscript", "JPEG": "image/jpeg", "TIFF": "image/tiff"}
            ext_map = {"PNG": "png", "PDF": "pdf", "SVG": "svg", "EPS": "eps", "JPEG": "jpg", "TIFF": "tiff"}
            fmt_key = "jpeg" if export_fmt == "JPEG" else export_fmt.lower()
            file_bytes = volcano.export_figure(fig_volc, fmt=fmt_key, dpi=export_dpi)
            with ec3:
                st.write("")
                st.write("")
                st.download_button(
                    f"Download VolcanoPlot.{ext_map[export_fmt]}", file_bytes,
                    file_name=f"VolcanoPlot.{ext_map[export_fmt]}", mime=mime_map[export_fmt]
                )

        with exp_tab2:
            dc1, dc2, dc3 = st.columns(3)
            up_tbl = volcano.get_direction_table(annotated, "Up")
            down_tbl = volcano.get_direction_table(annotated, "Down")
            sig_tbl = volcano.get_direction_table(annotated, None)
            with dc1:
                st.download_button("Download Upregulated.csv", utils.to_download_bytes_csv(up_tbl),
                                    "Upregulated_Metabolites.csv", "text/csv")
            with dc2:
                st.download_button("Download Downregulated.csv", utils.to_download_bytes_csv(down_tbl),
                                    "Downregulated_Metabolites.csv", "text/csv")
            with dc3:
                st.download_button("Download Complete_Volcano_Data.csv", utils.to_download_bytes_csv(annotated),
                                    "Complete_Volcano_Data.csv", "text/csv")
            st.caption("Figure settings (for reproducibility):")
            settings_json = volcano.export_settings_json(st.session_state.volcano_settings)
            st.download_button("Download Figure_Settings.json", settings_json.encode(),
                                "Volcano_Figure_Settings.json", "application/json")

# ===========================================================================
# TAB 10 — BIOMARKER DISCOVERY
# ===========================================================================
with TABS[9]:
    st.header("Biomarker Discovery")
    if st.session_state.stats_result is None:
        st.warning("Run a two-group statistical comparison in the Statistics tab first.")
    else:
        result = st.session_state.stats_result
        groups_used = st.session_state.get("stats_result_groups")
        group_col_used = st.session_state.get("stats_result_group_col") or "Group"
        if groups_used:
            st.caption(
                f"Reflects the two {group_col_used} values compared in the Statistics tab: "
                f"**{groups_used[0]}** vs **{groups_used[1]}**"
                + (" (combined normalized data, all datasets)."
                   if st.session_state.data_mode == "multi" else ".")
                + " To compare a different pair or variable, go back to Statistics and "
                "re-run the two-group test."
            )
        c1, c2, c3 = st.columns(3)
        criterion = c1.selectbox("Filtering criterion", ["combined", "pvalue", "fdr"],
                                  format_func=lambda x: {"combined": "p<0.05 AND FDR<0.25",
                                                          "pvalue": "p<0.05 only", "fdr": "FDR<0.25 only"}[x])
        p_cutoff = c2.number_input("p-value cutoff", value=0.05, min_value=0.0001, max_value=1.0, step=0.01)
        fdr_cutoff = c3.number_input("FDR cutoff", value=0.25, min_value=0.0001, max_value=1.0, step=0.01)

        if st.button("Discover Biomarkers"):
            biomarkers = biomarker.discover_biomarkers(result, criterion=criterion,
                                                        p_cutoff=p_cutoff, fdr_cutoff=fdr_cutoff)
            st.session_state.biomarkers = biomarkers
            st.success(f"{len(biomarkers)} biomarker candidates identified.")

        if st.session_state.get("biomarkers") is not None:
            biomarkers = st.session_state.biomarkers
            st.dataframe(biomarkers, width='stretch', height=450)
            comparison_name = (f"{groups_used[0]}_vs_{groups_used[1]}" if groups_used else "Comparison")
            biomarker_filename = f"{comparison_name}_Biomarkers.csv"
            st.download_button(
                f"Download {biomarker_filename}", utils.to_download_bytes_csv(biomarkers),
                file_name=biomarker_filename, mime="text/csv", key="dl_biomarkers"
            )
            st.caption(f"Filename includes the comparison ({comparison_name}) so results from "
                       "different comparisons stay distinguishable.")

# ===========================================================================
# TAB 11 — HEATMAP
# ===========================================================================
with TABS[10]:
    st.header("Clustered Heatmap")
    if st.session_state.stats_result is None:
        st.warning(
            "Run a two-group comparison in the Statistics tab first — the heatmap needs "
            "that result to know which metabolites are significant."
        )
    else:
        log2_df_full = st.session_state.log2_data
        result = st.session_state.stats_result
        meta = st.session_state.meta

        heat_cat_cols = utils.get_categorical_metadata_columns(meta, log2_df_full.columns)
        default_group_col = st.session_state.get("stats_result_group_col") or "Group"
        heat_group_col = st.selectbox(
            "Filter samples by:", heat_cat_cols,
            index=heat_cat_cols.index(default_group_col) if default_group_col in heat_cat_cols else 0,
            key="heatmap_group_col"
        )
        groups_all_heat = meta.loc[meta.index.intersection(log2_df_full.columns), heat_group_col].unique().tolist()
        st.subheader("Select Groups")
        selected_heat_groups = st.multiselect(
            f"{heat_group_col} values to include in the heatmap (choose any subset — 2, 3, 4, or more)",
            groups_all_heat, default=groups_all_heat, key="heatmap_group_select"
        )
        heat_cols = [c for c in log2_df_full.columns if meta.loc[c, heat_group_col] in selected_heat_groups]
        log2_df = log2_df_full[heat_cols]

        st.subheader("Column (Sample) Annotation")
        st.caption(
            "Pick any number of metadata columns to show as stacked annotation tracks above the "
            "heatmap — categorical columns (Diagnosis, Gender, Treatment, ...) render as discrete "
            "color blocks; numeric columns with many distinct values (Age, Body Weight, ...) render "
            "as a color gradient with its own scale."
        )
        col_annot_all_options = [c for c in meta.columns if c != "IsQC"]
        default_col_annot = [heat_group_col] if heat_group_col in col_annot_all_options else []
        col_annot_cols = st.multiselect(
            "Column annotation tracks:", col_annot_all_options, default=default_col_annot,
            key="heatmap_col_annot_cols"
        )

        row_annotations = st.session_state.get("row_annotations")
        row_annot_cols = []
        if row_annotations is not None:
            st.subheader("Row (Metabolite) Annotation")
            row_annot_cols = st.multiselect(
                "Row annotation tracks:", list(row_annotations.columns),
                default=list(row_annotations.columns)[:1], key="heatmap_row_annot_cols"
            )
            if row_annot_cols:
                n_matched = heatmap_module.count_row_annotation_matches(row_annotations, log2_df_full.index)
                st.caption(f"{n_matched}/{len(log2_df_full.index)} features matched to row annotations "
                           f"by name; unmatched features show as '(unannotated)' in the row bar(s).")

        # Stage-then-apply color model: editing pickers below never itself
        # triggers a full heatmap re-render (which can be slow with many tracks) --
        # only "Apply Colors" copies the staged picks into what the heatmap below
        # actually uses. "Reset Colors" clears both the applied colors AND each
        # widget's own state, so pickers visually revert to their auto defaults --
        # since widgets already exist earlier in this same run by the time the
        # buttons are clicked, that reversion needs a rerun to take visual effect,
        # which is why reset sets a pending flag + calls st.rerun() rather than
        # popping keys immediately.
        all_annotation_tracks = (
            [(c, meta, "(missing)") for c in col_annot_cols]
            + [(c, row_annotations, "(unannotated)") for c in row_annot_cols]
        )
        if st.session_state.get("heatmap_colors_reset_pending"):
            for track_col, lookup_df, _ in all_annotation_tracks:
                kind = heatmap_module.classify_annotation_series(lookup_df[track_col])
                if kind == "continuous":
                    st.session_state.pop(f"heatmap_annot_cmap_{track_col}", None)
                else:
                    for val in lookup_df[track_col].dropna().unique():
                        st.session_state.pop(f"heatmap_annot_color_{track_col}_{str(val)}", None)
            st.session_state.heatmap_colors_reset_pending = False
            st.session_state.heatmap_annotation_colors_applied = {}

        if all_annotation_tracks:
            with st.expander("🎨 Annotation Colors", expanded=False):
                annotation_colors_live = {}

                # Compact layout:
                # - one small expander per annotation track
                # - categorical colors displayed in a 4-column grid
                # - continuous tracks use one compact colormap selector
                # - no full-width divider after every value
                st.caption(
                    "Customize categorical annotation colors or continuous color maps. "
                    "Changes are staged until you click Apply Colors."
                )

                for track_col, lookup_df, missing_label in all_annotation_tracks:
                    kind = heatmap_module.classify_annotation_series(
                        lookup_df[track_col]
                    )

                    values = sorted(
                        str(v)
                        for v in lookup_df[track_col].dropna().unique()
                    )

                    with st.expander(
                        f"**{track_col}**  ·  "
                        f"{'Continuous' if kind == 'continuous' else f'{len(values)} categories'}",
                        expanded=False,
                    ):
                        if kind == "continuous":
                            cmap_choice = st.selectbox(
                                "Color map",
                                heatmap_module.CONTINUOUS_ANNOT_CMAPS,
                                key=f"heatmap_annot_cmap_{track_col}",
                            )
                            annotation_colors_live[track_col] = cmap_choice
                        else:
                            track_colors = {}

                            if values:
                                # Four compact swatches per row keeps the panel short.
                                ncols = min(4, max(1, len(values)))
                                color_cols = st.columns(ncols)

                                for j, val in enumerate(values):
                                    default_hex = (
                                        heatmap_module.GROUP_PALETTE[
                                            j % len(heatmap_module.GROUP_PALETTE)
                                        ]
                                    )

                                    with color_cols[j % ncols]:
                                        st.caption(
                                            val,
                                            help=f"Annotation value: {val}",
                                        )
                                        track_colors[val] = st.color_picker(
                                            f"Color for {val}",
                                            value=default_hex,
                                            key=f"heatmap_annot_color_{track_col}_{val}",
                                            label_visibility="collapsed",
                                        )

                            annotation_colors_live[track_col] = track_colors

                st.markdown("---")
                fc1, fc2, _ = st.columns([1, 1, 6])

                with fc1:
                    apply_clicked = st.button(
                        "✓ Apply",
                        type="primary",
                        key="heatmap_apply_colors",
                        use_container_width=True,
                    )

                with fc2:
                    reset_clicked = st.button(
                        "↺ Reset",
                        key="heatmap_reset_colors",
                        use_container_width=True,
                    )

                if apply_clicked:
                    st.session_state.heatmap_annotation_colors_applied = (
                        annotation_colors_live
                    )

                if reset_clicked:
                    st.session_state.heatmap_colors_reset_pending = True
                    st.session_state.heatmap_annotation_colors_applied = {}
                    st.rerun()


        annotation_colors = (
            st.session_state.get("heatmap_annotation_colors_applied") or {}
        )

        CUTOFF_OPTIONS = {
            "FDR ≤ 1 (no filter)": ("FDR", 1.0),
            "FDR ≤ 0.25": ("FDR", 0.25),
            "P-value ≤ 1 (no filter)": ("p-value", 1.0),
            "P-value ≤ 0.05": ("p-value", 0.05),
        }

        st.subheader("Feature Selection")
        c1, c2 = st.columns(2)
        cutoff_label = c1.selectbox("Significant feature cutoff", list(CUTOFF_OPTIONS.keys()), index=1)
        col_preview, cutoff_preview = CUTOFF_OPTIONS[cutoff_label]
        n_preview = (result[col_preview] <= cutoff_preview).sum()
        with c2:
            st.metric("Metabolites at this cutoff", n_preview)
        if n_preview > heatmap_module.HIDE_ROW_LABELS_ABOVE:
            st.warning(
                f"{n_preview} metabolites selected — beyond {heatmap_module.HIDE_ROW_LABELS_ABOVE}, "
                f"row labels become unreadable and will be hidden automatically. Tighten the cutoff "
                f"above for a labeled figure, or proceed for an unlabeled overview heatmap."
            )

        st.subheader("Clustering Options")
        c3, c4, c5 = st.columns(3)
        cluster_mode = c3.selectbox("Clustering", ["Both rows and columns", "Rows only", "Columns only", "No clustering"])
        distance = c4.selectbox("Distance metric", ["euclidean", "correlation"])
        linkage_m = c5.selectbox("Linkage method", ["ward", "average", "complete"])
        cluster_rows = cluster_mode in ("Both rows and columns", "Rows only")
        cluster_cols = cluster_mode in ("Both rows and columns", "Columns only")

        st.subheader("Color Scale Customization")
        c6, c7 = st.columns(2)
        use_custom_gradient = c6.checkbox("Use a custom color chart instead of a preset palette", value=False)
        reverse_cmap = c7.checkbox("Reverse color chart", value=False)

        if use_custom_gradient:
            gc1, gc2, gc3 = st.columns(3)
            color_low = gc1.color_picker("Low color", "#2166AC")
            color_mid = gc2.color_picker("Mid color", "#FFFFFF")
            color_high = gc3.color_picker("High color", "#B2182B")
            custom_colors = [color_low, color_mid, color_high]
            cmap_name = None
        else:
            cmap_name = st.selectbox("Predefined color palette", heatmap_module.PREDEFINED_PALETTES)
            custom_colors = None

        c8, c9 = st.columns(2)
        vmin = c8.number_input("Z-score minimum", value=-2.5, step=0.1)
        vmax = c9.number_input("Z-score maximum", value=2.5, step=0.1)

        use_breakpoints = st.checkbox("Use custom discrete color breakpoints (instead of a continuous scale)", value=False)
        breakpoints = None
        if use_breakpoints:
            bp_text = st.text_input("Comma-separated breakpoints (e.g. -3,-1,0,1,3)",
                                     value=f"{vmin},{vmin/2:.2g},0,{vmax/2:.2g},{vmax}")
            try:
                breakpoints = [float(x.strip()) for x in bp_text.split(",") if x.strip()]
            except ValueError:
                st.warning("Couldn't parse breakpoints — using a continuous scale instead.")
                breakpoints = None

        st.subheader("Font")
        cf1, cf2 = st.columns(2)
        heatmap_font_family = cf1.selectbox("Font family", ["sans-serif", "serif", "monospace"],
                                             key="heatmap_font_family")
        heatmap_font_size = cf2.slider("Base font size", 6, 20, 10, 1, key="heatmap_font_size")
        st.caption("Title, tick labels, legends, and annotation track labels all scale from this "
                   "one size.")

        st.subheader("Figure Size")
        c_size1, c_size2, c_size3 = st.columns(3)
        use_custom_size = c_size1.checkbox("Customize heatmap panel size", value=False)
        heatmap_width_in = heatmap_height_in = None
        if use_custom_size:
            heatmap_width_in = c_size2.number_input("Width (inches)", value=8.0, min_value=2.0, max_value=30.0, step=0.5)
            heatmap_height_in = c_size3.number_input("Height (inches)", value=6.0, min_value=2.0, max_value=30.0, step=0.5)
            st.caption("All other elements (dendrograms, annotation bar, legend, colorbar, labels) "
                       "scale automatically with the size you set here.")

        if len(heat_cols) < 2:
            st.warning("Select at least one group with 2+ samples to build a heatmap.")
        else:
            col, cutoff = CUTOFF_OPTIONS[cutoff_label]
            sig_feats = result[result[col] <= cutoff].index

            if len(sig_feats) < 2:
                st.warning(
                    f"Only {len(sig_feats)} significant feature(s) at this cutoff — "
                    "clustering needs at least 2. Try relaxing the threshold in the Statistics or "
                    "Biomarker Discovery tab."
                )
            else:
                try:
                    fig_heat, z_ordered, notes = heatmap_module.clustered_heatmap(
                        log2_df, sig_feats, meta=meta, col_annot_cols=col_annot_cols,
                        cluster_rows=cluster_rows, cluster_cols=cluster_cols,
                        distance=distance, linkage_method=linkage_m,
                        cmap_name=cmap_name, custom_colors=custom_colors, reverse_cmap=reverse_cmap,
                        vmin=vmin, vmax=vmax, breakpoints=breakpoints,
                        heatmap_width_in=heatmap_width_in, heatmap_height_in=heatmap_height_in,
                        row_meta=row_annotations, row_annot_cols=row_annot_cols,
                        annotation_colors=annotation_colors, font_family=heatmap_font_family,
                        font_size=heatmap_font_size,
                        title_suffix=f"{len(sig_feats)} features, {cutoff_label}",
                    )
                    st.session_state.viz_figs["Clustered Heatmap"] = fig_heat
                    st.session_state.heatmap_fig = fig_heat
                    st.session_state.heatmap_zscore = z_ordered
                    for note in notes:
                        st.info(f"ℹ️ {note}")
                    st.pyplot(fig_heat)
                    st.caption(f"{len(sig_feats)} metabolites shown ({cutoff_label}), "
                               f"{len(heat_cols)} samples across {len(selected_heat_groups)} selected group(s), "
                               f"row-scaled (z-score).")

                    st.subheader("Export Heatmap")
                    ec1, ec2, ec3 = st.columns(3)
                    export_fmt = ec1.selectbox("Format", ["PNG", "PDF", "SVG", "JPEG", "TIFF"])
                    export_dpi = ec2.selectbox("Resolution (DPI)", [150, 300, 600], index=1)
                    st.caption(
                        "Note: unlike typical vector plots, this heatmap's cells are drawn as an image "
                        "internally, so DPI affects sharpness even for PDF/SVG — use 300+ to avoid a "
                        "blurry/smeared appearance when zoomed in or printed."
                    )
                    mime_map = {"PNG": "image/png", "PDF": "application/pdf", "SVG": "image/svg+xml",
                                "JPEG": "image/jpeg", "TIFF": "image/tiff"}
                    ext_map = {"PNG": "png", "PDF": "pdf", "SVG": "svg", "JPEG": "jpg", "TIFF": "tiff"}
                    fmt_key = "jpeg" if export_fmt == "JPEG" else export_fmt.lower()
                    file_bytes = heatmap_module.export_figure(fig_heat, fmt=fmt_key, dpi=export_dpi)
                    with ec3:
                        st.write("")
                        st.write("")
                        st.download_button(
                            f"Download Heatmap.{ext_map[export_fmt]}", file_bytes,
                            file_name=f"Heatmap.{ext_map[export_fmt]}", mime=mime_map[export_fmt]
                        )

                    st.subheader("Z-score Data Table")
                    st.caption(
                        "The exact row-scaled Z-score values used to render the heatmap above "
                        "(rows/columns in the same clustered order shown)."
                    )
                    st.dataframe(z_ordered, width='stretch', height=300)
                    st.download_button(
                        "Download Zscore_Table.csv", utils.to_download_bytes_csv(z_ordered),
                        file_name="Zscore_Table.csv", mime="text/csv"
                    )
                except ValueError as e:
                    st.warning(str(e))

# ===========================================================================
# TAB 12 — BOXPLOT OF METABOLITES
# ===========================================================================
with TABS[11]:
    st.header("Boxplot of Metabolites")
    if st.session_state.log2_data is None:
        st.warning(
            "Complete the Normalization tab (Log2 transform) first."
            if st.session_state.data_mode == "single" else
            "Complete Normalization for every dataset, then click **🔗 Generate Combined "
            "Normalized Data** in the Normalization tab first — that combined table is what "
            "every downstream tab analyzes."
        )
    else:
        if st.session_state.data_mode == "multi":
            st.caption("Analyzing the **combined normalized data** across all datasets.")
        log2_df_full = st.session_state.log2_data
        meta = st.session_state.meta

        st.subheader("Select Metabolites & Groups")
        selected_metabolites = st.multiselect(
            "Search and select one or more metabolites",
            log2_df_full.index.tolist()
        )
        box_cat_cols = utils.get_categorical_metadata_columns(meta, log2_df_full.columns)
        box_group_col = st.selectbox(
            "Grouping variable:", box_cat_cols,
            index=box_cat_cols.index("Group") if "Group" in box_cat_cols else 0,
            key="boxplot_group_col"
        )
        groups_all_box = meta.loc[meta.index.intersection(log2_df_full.columns), box_group_col].unique().tolist()
        selected_box_groups = st.multiselect(
            f"{box_group_col} values to include in the comparison (choose any subset — 2, 3, 4, or more)",
            groups_all_box, default=groups_all_box, key="boxplot_group_select"
        )

        if selected_metabolites and len(selected_box_groups) >= 2:
            test_name = "Welch's t-test" if len(selected_box_groups) == 2 else "one-way ANOVA"
            st.caption(
                f"Statistics computed on log2-transformed data: {test_name} "
                f"across the {len(selected_box_groups)} selected group(s). FDR here is corrected across "
                f"only the {len(selected_metabolites)} metabolite(s) shown, not the full feature panel — "
                f"for a panel-wide FDR, use the Statistics tab."
            )

        st.subheader("Customize Appearance")
        c1, c2, c3 = st.columns(3)
        font_family = c1.selectbox("Font family", ["sans-serif", "serif", "monospace"])
        font_size = c2.slider("Font size", 6, 20, 10, 1)
        ncols = c3.number_input("Panels per row", min_value=1, max_value=6, value=3, step=1)

        c4, c5 = st.columns(2)
        use_custom_box_size = c4.checkbox("Customize figure size", value=False)
        fig_width_in = fig_height_in = None
        if use_custom_box_size:
            fig_width_in = c5.number_input("Width (inches)", value=10.0, min_value=3.0, max_value=30.0, step=0.5)
            fig_height_in = st.number_input("Height (inches)", value=6.0, min_value=3.0, max_value=30.0, step=0.5)

        c6, c7, c8 = st.columns(3)
        show_points = c6.checkbox("Show individual data points", value=True)
        show_mean = c7.checkbox("Show mean (dashed line)", value=False)
        show_median = c8.checkbox("Show median (solid line)", value=True)

        st.caption("Optional: assign a color per group (defaults to a standard palette).")
        group_colors = {}
        if selected_box_groups:
            color_cols = st.columns(min(4, len(selected_box_groups)) or 1)
            for i, g in enumerate(selected_box_groups):
                with color_cols[i % len(color_cols)]:
                    default_c = boxplot_module.GROUP_PALETTE[i % len(boxplot_module.GROUP_PALETTE)]
                    group_colors[g] = st.color_picker(f"Color: {g}", default_c, key=f"boxcolor_{g}")

        if not selected_metabolites:
            st.info("Select at least one metabolite above to generate a boxplot.")
        elif len(selected_box_groups) < 2:
            st.warning("Select at least 2 groups to compare.")
        else:
            box_cols = [c for c in log2_df_full.columns if meta.loc[c, box_group_col] in selected_box_groups]
            log2_df_box = log2_df_full[box_cols]

            stats_table = boxplot_module.compute_stats_for_metabolites(
                log2_df_box, meta, selected_metabolites, selected_box_groups, group_col=box_group_col
            )
            fig_box = boxplot_module.boxplot_metabolites(
                log2_df_box, meta, selected_metabolites, selected_box_groups,
                stats_table=stats_table, fig_width_in=fig_width_in, fig_height_in=fig_height_in,
                font_size=font_size, font_family=font_family, group_colors=group_colors,
                show_points=show_points, show_mean=show_mean, show_median=show_median, ncols=ncols,
                group_col=box_group_col,
            )
            st.session_state.viz_figs["Boxplot of Metabolites"] = fig_box
            st.session_state.boxplot_fig = fig_box
            st.session_state.boxplot_stats = stats_table
            st.pyplot(fig_box)

            st.subheader("Statistical Results")
            st.dataframe(stats_table, width='stretch')

            st.subheader("Export")
            ec1, ec2, ec3 = st.columns(3)
            export_fmt = ec1.selectbox("Format", ["PNG", "PDF", "SVG", "JPEG", "TIFF"], key="boxplot_export_fmt")
            export_dpi = ec2.selectbox("Resolution (DPI)", [150, 300, 600], index=1,
                                        disabled=export_fmt in ("PDF", "SVG"), key="boxplot_export_dpi")
            mime_map = {"PNG": "image/png", "PDF": "application/pdf", "SVG": "image/svg+xml",
                        "JPEG": "image/jpeg", "TIFF": "image/tiff"}
            ext_map = {"PNG": "png", "PDF": "pdf", "SVG": "svg", "JPEG": "jpg", "TIFF": "tiff"}
            fmt_key = "jpeg" if export_fmt == "JPEG" else export_fmt.lower()
            file_bytes = boxplot_module.export_figure(fig_box, fmt=fmt_key, dpi=export_dpi)
            with ec3:
                st.write("")
                st.write("")
                st.download_button(
                    f"Download Boxplot.{ext_map[export_fmt]}", file_bytes,
                    file_name=f"Boxplot.{ext_map[export_fmt]}", mime=mime_map[export_fmt]
                )

st.sidebar.title("MetaboAI Pro")
st.sidebar.markdown(
    "**LC-MS based Metabolomics & Lipidomics Analysis**\n\n"
    "Workflow: QC → ISTD/IQR/Log2 Normalization → Statistics (t-test/Wilcoxon/ANOVA) "
    "→ PCA → Volcano → Biomarker Discovery → Heatmap (every step has its own downloads)"
)
if st.session_state.processing_notes:
    st.sidebar.subheader("Processing Log")
    for note in st.session_state.processing_notes:
        st.sidebar.markdown(f"- {note}")
