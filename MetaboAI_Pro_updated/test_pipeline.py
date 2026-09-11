import sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from modules import utils, qc, normalization, imputation_module, stats_analysis, pca_module, volcano, biomarker, heatmap_module

# =====================================================================
# PART A: TARGETED dataset — exercises the ISTD -> Log2 branch only
# =====================================================================
peak_raw_t = pd.read_csv(f"{BASE}/sample_peak_area_matrix_targeted.csv")
meta_raw_t = pd.read_csv(f"{BASE}/sample_metadata_targeted.csv")
peak_df_t = utils.validate_peak_matrix(peak_raw_t)
meta_t = utils.validate_metadata(meta_raw_t, peak_df_t.columns)
qc_cols_t, sample_cols_t = utils.split_qc_and_samples(peak_df_t, meta_t)
assert "ISTD_D4-Alanine" in peak_df_t.index, "Targeted demo must include an ISTD row"
assert set(meta_t["Group"].unique()) >= {"Control", "Mild", "Moderate", "Severe"}
print("A1. Targeted demo loaded OK:", peak_df_t.shape, "groups:", meta_t["Group"].unique().tolist())

working_t = peak_df_t[sample_cols_t].copy()
missingness_t = imputation_module.compute_missingness(working_t)
n_before_t = working_t.shape[0]
cleaned_t = imputation_module.filter_by_missingness(working_t, missingness_t, max_pct_missing=50)
n_missing_t = imputation_module.count_missing(cleaned_t)
imputed_t = imputation_module.impute_half_minimum(cleaned_t) if n_missing_t > 0 else cleaned_t
assert imputation_module.count_missing(imputed_t, treat_zero_as_missing=False) == 0
print(f"A1b. Targeted cleaning+imputation OK: {imputed_t.shape[0]}/{n_before_t} features retained, "
      f"{n_missing_t} values imputed")

istd_norm = normalization.istd_normalize(imputed_t, "ISTD_D4-Alanine")
log2_t, const_t = normalization.log2_transform(istd_norm)
assert set(log2_t.columns) == set(sample_cols_t)
assert log2_t.isna().sum().sum() == 0, "log2_transform left NaN values (zero-handling bug regressed)"
print("A2. ISTD -> Log2 pipeline OK:", log2_t.shape, "NaNs:", log2_t.isna().sum().sum())

# =====================================================================
# PART B: UNTARGETED dataset — exercises the Log2 -> Median-IQR branch,
# and is used for the rest of the pipeline (stats/PCA/volcano/heatmap/report)
# =====================================================================
peak_raw = pd.read_csv(f"{BASE}/sample_peak_area_matrix_untargeted.csv")
meta_raw = pd.read_csv(f"{BASE}/sample_metadata_untargeted.csv")
peak_df = utils.validate_peak_matrix(peak_raw)
meta = utils.validate_metadata(meta_raw, peak_df.columns)
qc_cols, sample_cols = utils.split_qc_and_samples(peak_df, meta)
assert "ISTD_D4-Alanine" not in peak_df.index, "Untargeted demo must NOT include an ISTD row"
assert set(meta["Group"].unique()) >= {"Control", "Mild", "Moderate", "Severe"}
print("B1. Untargeted demo loaded OK:", peak_df.shape, "groups:", meta["Group"].unique().tolist())

# 2. QC — visualization uses QC-ONLY data (no biological samples mixed in).
# Only the correlation matrix is retained (PCA/dendrogram/distance-heatmap removed per request).
cv_table = qc.calculate_cv(peak_df[qc_cols])
fig_cv_dist = qc.cv_distribution_plot(cv_table)
fig_cv_hist = qc.cv_histogram(cv_table)

qc_log, _ = normalization.log2_transform(peak_df[qc_cols])
assert qc_log.shape[1] == len(qc_cols)
fig_corr, corr_df = qc.sample_correlation_matrix(qc_log)
assert set(corr_df.columns) == set(qc_cols)
print("B2. QC-only plots OK (correlation matrix only):", cv_table.shape, cv_table["Quality"].value_counts().to_dict())

keep = cv_table.index[cv_table["Quality"] == "Acceptable"]
working_df = peak_df[sample_cols].copy()
working_df = working_df.loc[working_df.index.intersection(keep)]
assert set(working_df.columns) == set(sample_cols)
print("B3. After CV filter + QC exclusion:", working_df.shape)

# 3b. Data Cleaning & Missing Value Imputation (new step, runs before normalization)
from modules import imputation_module
missingness_table = imputation_module.compute_missingness(working_df)
n_before = working_df.shape[0]
cleaned_df = imputation_module.filter_by_missingness(working_df, missingness_table, max_pct_missing=50)
print(f"B3a. Missingness filter OK: {cleaned_df.shape[0]}/{n_before} features retained "
      f"(categories: {missingness_table['Quality'].value_counts().to_dict()})")

n_missing_before = imputation_module.count_missing(cleaned_df)
assert n_missing_before > 0, "Expected some missing values in the filtered demo data"

for method in imputation_module.METHOD_INFO:
    imputed = imputation_module.impute(cleaned_df, method)
    assert imputation_module.count_missing(imputed, treat_zero_as_missing=False) == 0, f"{method} left NaN"
    assert (imputed.values < 0).sum() == 0, f"{method} produced negative peak areas"
print(f"B3b. All 6 imputation methods OK, no NaN/negative values ({n_missing_before} values imputed)")

# Use Half-Minimum (fast, deterministic) for the rest of the pipeline
working_df = imputation_module.impute_half_minimum(cleaned_df)
assert imputation_module.count_missing(working_df, treat_zero_as_missing=False) == 0
print("B3c. Working dataset fully imputed, proceeding to normalization:", working_df.shape)

# Untargeted path: Median-IQR normalization (on raw abundance) -> shift + Log2
# (Median-IQR produces negative values for points below the median; log2 needs a
# positivity shift first -- shift_and_log2_transform handles this.)
iqr_norm_feat = normalization.iqr_normalize(working_df, axis="feature")
iqr_norm_sample = normalization.iqr_normalize(working_df, axis="sample")
iqr_norm_batch = normalization.iqr_normalize(working_df, axis="batch", batch_map=meta["Batch"])
print("B4. Median-IQR robust scaling OK (feature/sample/batch), pre-log2:",
      iqr_norm_feat.shape, iqr_norm_sample.shape, iqr_norm_batch.shape)
assert (iqr_norm_feat.values < 0).any(), "Expected negative values from centering normalization"

log2_df, shift_used = normalization.shift_and_log2_transform(iqr_norm_feat)
assert np.isfinite(log2_df.values).all(), "shift_and_log2_transform produced NaN/Inf"
print(f"B5. Shift+Log2 (untargeted path) OK, shift={shift_used:.4g}", log2_df.shape,
      "NaNs:", log2_df.isna().sum().sum())
assert set(log2_df.columns) == set(sample_cols)

fig_dist = normalization.distribution_plots(iqr_norm_feat, log2_df)
print("B6. Distribution plots OK")

# 4. Statistics — two-group (Control vs Severe), computed purely from log2 data
a_samples = meta.index[meta["Group"] == "Control"].tolist()
b_samples = meta.index[meta["Group"] == "Severe"].tolist()
a_samples = [s for s in a_samples if s in log2_df.columns]
b_samples = [s for s in b_samples if s in log2_df.columns]
result = stats_analysis.complete_statistical_table(log2_df, a_samples, b_samples)
assert "Mean_Log2_GroupA" in result.columns and "CI_Lower_Log2FC" in result.columns
print("B7. Two-group stats (Control vs Severe) OK:", result.shape, "Significant:", result["Significant"].sum())
print(result.head(3))

# 4b. ANOVA across the real 4 groups (Control/Mild/Moderate/Severe) — no fake relabeling needed
group_map = meta.loc[sample_cols, "Group"]
group_map = group_map[group_map.index.isin(log2_df.columns)]
anova_table, posthoc_results = stats_analysis.anova_test(log2_df[group_map.index], group_map, posthoc="tukey")
print("B8. ANOVA (4 real groups) OK:", anova_table.shape, "Sig(FDR<0.25):", (anova_table["FDR"] < 0.25).sum())
if posthoc_results:
    k = list(posthoc_results.keys())[0]
    print("    posthoc example:\n", posthoc_results[k])

anova_table2, posthoc2 = stats_analysis.anova_test(log2_df[group_map.index], group_map, posthoc="dunnett")
print("B8b. ANOVA dunnett OK")
anova_table3, posthoc3 = stats_analysis.anova_test(log2_df[group_map.index], group_map, posthoc="pairwise")
print("B8c. ANOVA pairwise OK")

# 5. PCA — biological samples only, with customization
pca, scores_df, cols = pca_module.run_pca(log2_df, n_components=5)
assert set(scores_df.index) == set(sample_cols)
fig_score = pca_module.pca_score_plot(pca, scores_df, meta, palette="Set2",
                                        marker_map={"Control": "o", "Mild": "s", "Moderate": "^", "Severe": "D"},
                                        show_ellipse=True)
fig_loading, top_loadings = pca_module.pca_loading_plot(pca, log2_df.index, top_n=20)
fig_var = pca_module.pca_variance_plot(pca)
print("B9. PCA OK (4 groups, QC-free, customizable):", scores_df.shape, top_loadings.shape)

# 6. Volcano
fig_volc_p, annotated_p = volcano.volcano_plot(result, y_metric="pvalue", sig_cutoff=0.05, top_label_n=10)
fig_volc_f, annotated_f = volcano.volcano_plot(result, y_metric="fdr", sig_cutoff=0.25, top_label_n=10)
print("B10. Volcano OK:", annotated_p["Direction"].value_counts().to_dict())

# New publication-grade features: custom thresholds, highlight, theme, legend positions
fig_volc_custom, ann_custom = volcano.volcano_plot(
    result, fc_threshold=0.5, sig_cutoff=0.01, palette="Colorblind-safe (Okabe-Ito)",
    point_shape="Triangle", legend_position="Bottom", theme="Nature",
    highlight_names=list(result.index[:2]), show_stats_box=True,
)
print("B10a. Volcano custom thresholds/palette/shape/legend-position/theme/highlight OK")

up_tbl = volcano.get_direction_table(annotated_p, "Up")
down_tbl = volcano.get_direction_table(annotated_p, "Down")
settings_json = volcano.export_settings_json({"fc_threshold": 1.0, "sig_cutoff": 0.05})
assert len(settings_json) > 0
print("B10a2. Direction tables + settings JSON export OK:", len(up_tbl), len(down_tbl))

for fmt in ["png", "pdf", "svg", "eps", "jpeg", "tiff"]:
    b = volcano.export_figure(fig_volc_p, fmt=fmt, dpi=300)
    assert len(b) > 0
print("B10b. Volcano export (png/pdf/svg/eps/jpeg/tiff) OK")

# 7. Biomarker discovery
bio_combined = biomarker.discover_biomarkers(result, criterion="combined")
bio_p = biomarker.discover_biomarkers(result, criterion="pvalue")
bio_fdr = biomarker.discover_biomarkers(result, criterion="fdr")
print("B11. Biomarker discovery OK:", len(bio_combined), len(bio_p), len(bio_fdr))

# 8. Heatmap — all 4 clustering modes + color customization + export
sig_feats = result[result["p-value"] < 0.05].index
for cr, cc, label in [(True, True, "both"), (True, False, "rows-only"),
                      (False, True, "cols-only"), (False, False, "none")]:
    fig_heat, z_ordered, notes = heatmap_module.clustered_heatmap(
        log2_df, sig_feats, meta=meta, cluster_rows=cr, cluster_cols=cc,
        distance="euclidean", linkage_method="ward"
    )
    print(f"B12. Heatmap clustering={label} OK:", z_ordered.shape)

for fmt in ["png", "pdf", "svg", "jpeg", "tiff"]:
    b = heatmap_module.export_figure(fig_heat, fmt=fmt, dpi=300)
    assert len(b) > 0
print("B13. Heatmap export (png/pdf/svg/jpeg/tiff) OK")

# 8a. Regression test: PDF export must actually embed a high-resolution raster
# (previously ignored dpi for "vector" formats, producing blurry/smeared cells)
pdf_bytes_lowdpi = heatmap_module.export_figure(fig_heat, fmt="pdf", dpi=72)
pdf_bytes_hidpi = heatmap_module.export_figure(fig_heat, fmt="pdf", dpi=300)
assert len(pdf_bytes_hidpi) > len(pdf_bytes_lowdpi), (
    "300 DPI PDF should embed more raster data than 72 DPI -- if sizes are equal, "
    "dpi is being ignored for PDF export again (the blur bug has regressed)."
)
print(f"B13a. PDF actually respects DPI (72dpi={len(pdf_bytes_lowdpi)}B vs "
      f"300dpi={len(pdf_bytes_hidpi)}B) OK")

# 8b. Regression test: large feature-count heatmap must not crash or produce an
# unbounded figure size (previously OOM-crashed at ~500 rows exported as 600 DPI TIFF)
np.random.seed(0)
n_rows_stress, n_cols_stress = 800, log2_df.shape[1]
stress_data = pd.DataFrame(np.random.randn(n_rows_stress, n_cols_stress),
                            index=[f"StressFeature_{i:04d}" for i in range(n_rows_stress)],
                            columns=log2_df.columns)
fig_stress, z_stress, notes_stress = heatmap_module.clustered_heatmap(
    stress_data, stress_data.index, meta=meta, cluster_rows=True, cluster_cols=True
)
w_in, h_in = fig_stress.get_size_inches()
assert h_in <= heatmap_module.MAX_HEATMAP_HEIGHT_IN + 5, f"Height not capped: {h_in}in"
assert len(notes_stress) > 0, "Expected a size-cap/label-hiding note for an 800-row heatmap"
for fmt in ["png", "tiff", "jpeg"]:
    b = heatmap_module.export_figure(fig_stress, fmt=fmt, dpi=600)
    assert len(b) > 0
print(f"B13b. Large-panel heatmap (800 features) OK: figsize={w_in:.1f}x{h_in:.1f}in, "
      f"notes={len(notes_stress)}, 600dpi export succeeded (would have crashed before this fix)")

# 9. Excel export
xlsx_bytes = utils.to_download_bytes_xlsx({
    "Log2_Normalized": log2_df, "Statistics": result, "Differential_Metabolites": bio_combined, "ANOVA": anova_table
})
with open(f"{BASE}/test_output_Statistics.xlsx", "wb") as f:
    f.write(xlsx_bytes)
print("B14. Excel export OK:", len(xlsx_bytes), "bytes")

# 16. Boxplot of Metabolites (new module)
from modules import boxplot_module
box_mets = log2_df.index[:4].tolist()
box_stats_2g = boxplot_module.compute_stats_for_metabolites(log2_df, meta, box_mets, ["Control", "Severe"])
box_stats_4g = boxplot_module.compute_stats_for_metabolites(
    log2_df, meta, box_mets, meta["Group"].unique().tolist()[:4]
)
fig_box = boxplot_module.boxplot_metabolites(log2_df, meta, box_mets, ["Control", "Severe"],
                                              stats_table=box_stats_2g, show_points=True, show_mean=True)
for fmt in ["png", "pdf", "svg", "jpeg", "tiff"]:
    b = boxplot_module.export_figure(fig_box, fmt=fmt, dpi=300)
    assert len(b) > 0
print("B16. Boxplot module OK:", box_stats_2g.shape, box_stats_4g.shape, "all export formats OK")

# =====================================================================
# PART C: MULTI-DATASET MODE — clean/QC/normalize 4 method/mode datasets
# independently, then combine into one unified matrix
# =====================================================================
from modules import dataset_manager

meta_multi_raw = pd.read_csv(f"{BASE}/sample_metadata_multimethod.csv")
demo_types = ["Untargeted Metabolomics", "Targeted Metabolomics",
              "Untargeted Lipidomics", "Targeted Lipidomics"]
datasets = {}
first_peak_df_multi = None
for dtype in demo_types:
    prefix = dtype.lower().replace(" ", "_")
    peak_raw = pd.read_csv(f"{BASE}/sample_peak_area_matrix_multimethod_{prefix}.csv")
    peak_df = utils.validate_peak_matrix(peak_raw)
    if first_peak_df_multi is None:
        first_peak_df_multi = peak_df
    entry = dataset_manager.make_dataset_entry(dtype, dtype)
    entry["raw_df"] = peak_df
    datasets[dtype] = entry

meta_multi = utils.validate_metadata(meta_multi_raw, first_peak_df_multi.columns)
warnings = dataset_manager.validate_shared_samples(datasets, meta_multi)
assert not warnings, f"Unexpected sample mismatch across multi-method datasets: {warnings}"
qc_cols_multi, sample_cols_multi = utils.split_qc_and_samples(first_peak_df_multi, meta_multi)
print(f"C1. Multi-dataset demo loaded OK: {len(datasets)} datasets "
      f"({', '.join(demo_types)}) sharing {len(sample_cols_multi)} biological samples")
for dtype in demo_types:
    is_targeted = "Targeted" in dtype
    assert ("ISTD_D4-Alanine" in datasets[dtype]["raw_df"].index) == is_targeted, \
        f"{dtype}: ISTD row presence should match Targeted/Untargeted"

for ds_id, ds in datasets.items():
    is_targeted = "Targeted" in ds["data_type"]
    ds["qc_cols"] = qc_cols_multi
    ds["sample_cols"] = sample_cols_multi
    source_df = ds["raw_df"][sample_cols_multi]

    # Clean + impute
    miss_table = imputation_module.compute_missingness(source_df)
    cleaned = imputation_module.filter_by_missingness(source_df, miss_table, max_pct_missing=50)
    ds["cleaned_data"] = cleaned
    n_missing = imputation_module.count_missing(cleaned)
    imputed = imputation_module.impute_half_minimum(cleaned) if n_missing > 0 else cleaned
    assert imputation_module.count_missing(imputed, treat_zero_as_missing=False) == 0
    ds["imputed_data"] = imputed

    # QC (CV-based filtering) + QC-only log2 matrix for the combined correlation matrix
    cv_table = qc.calculate_cv(ds["raw_df"][qc_cols_multi])
    ds["cv_table"] = cv_table
    qc_log_ds, _ = normalization.log2_transform(ds["raw_df"][qc_cols_multi])
    ds["qc_log_data"] = qc_log_ds
    keep = cv_table.index[cv_table["Quality"] == "Acceptable"]
    qc_filtered = imputed.loc[imputed.index.intersection(keep)]

    # Normalization branches by analysis type, same as the app's render_normalization_ui
    if is_targeted:
        istd_norm = normalization.istd_normalize(qc_filtered, "ISTD_D4-Alanine")
        ds["istd_normalized"] = istd_norm
        log2_ds, _ = normalization.log2_transform(istd_norm)
    else:
        iqr_norm = normalization.iqr_normalize(qc_filtered, axis="feature")
        ds["iqr_normalized"] = iqr_norm
        log2_ds, _ = normalization.shift_and_log2_transform(iqr_norm)
    assert np.isfinite(log2_ds.values).all(), f"{ds_id}: NaN/Inf after per-dataset normalization"
    ds["log2_data"] = log2_ds

print("C2. Per-dataset cleaning + QC + normalization (branched by analysis type) OK for all "
      f"{len(demo_types)} datasets ({[datasets[d]['log2_data'].shape[0] for d in demo_types]} "
      "features respectively)")

# --- Combined Peak Area (post-imputation), Tab 2's "Combine Datasets" step ---
combined_imputed, imputed_counts = dataset_manager.combine_imputed_datasets(datasets, sample_cols_multi)
assert combined_imputed.shape[0] == sum(imputed_counts.values())
assert all("::" in idx for idx in combined_imputed.index)
imputation_summary = dataset_manager.imputation_summary_table(datasets)
assert (imputation_summary["Status"] == "Imputed").sum() == len(demo_types)
print(f"C3. Combined post-imputation peak area matrix OK: {combined_imputed.shape}, "
      f"counts={imputed_counts}")

# --- Combined QC: CV distribution / feature counts / sample correlation matrix ---
combined_cv = dataset_manager.combine_cv_tables(datasets)
assert all("::" in idx for idx in combined_cv.index)
assert combined_cv.shape[0] == sum(len(ds["cv_table"]) for ds in datasets.values())
fig_cv_dist_combined = qc.cv_distribution_plot(combined_cv)
fig_cv_hist_combined = qc.cv_histogram(combined_cv)
combined_qc_log = dataset_manager.combine_qc_log_data(datasets, qc_cols_multi)
assert set(combined_qc_log.columns) == set(qc_cols_multi)
fig_corr_combined, corr_df_combined = qc.sample_correlation_matrix(combined_qc_log)
assert set(corr_df_combined.columns) == set(qc_cols_multi)
for fig in (fig_cv_dist_combined, fig_cv_hist_combined, fig_corr_combined):
    b = dataset_manager.fig_to_png_bytes(fig)
    assert len(b) > 0
print(f"C4. Combined QC (CV distribution, feature counts, sample correlation) OK: "
      f"{combined_cv.shape[0]} combined features, {combined_qc_log.shape[0]} combined QC rows")

# --- Combined Normalization: both pre-Log2 and Log2 tables ---
combined_prelog2, prelog2_counts = dataset_manager.combine_normalized_datasets(
    datasets, sample_cols_multi, use_log2=False
)
assert combined_prelog2.shape[0] == sum(prelog2_counts.values())
assert all("::" in idx for idx in combined_prelog2.index)
combined_log2, log2_counts = dataset_manager.combine_normalized_datasets(
    datasets, sample_cols_multi, use_log2=True
)
# combine_log2_datasets() must remain a working alias
combined_log2_alias, log2_counts_alias = dataset_manager.combine_log2_datasets(datasets, sample_cols_multi)
assert combined_log2.equals(combined_log2_alias) and log2_counts == log2_counts_alias
expected_total = sum(log2_counts.values())
assert combined_log2.shape[0] == expected_total
assert combined_log2.shape[1] == len(sample_cols_multi)
assert all("::" in idx for idx in combined_log2.index), "Combined features must be dataset-prefixed"
assert np.isfinite(combined_log2.values).all()
print(f"C5. Combined normalized matrices OK — pre-Log2: {combined_prelog2.shape} "
      f"({prelog2_counts}), Log2: {combined_log2.shape} ({log2_counts})")

combined = combined_log2

# Confirm the combined matrix works with the SAME downstream stats/PCA/heatmap functions
a_samples_multi = meta_multi.index[meta_multi["Group"] == "Control"].tolist()
b_samples_multi = meta_multi.index[meta_multi["Group"] == "Severe"].tolist()
a_samples_multi = [s for s in a_samples_multi if s in combined.columns]
b_samples_multi = [s for s in b_samples_multi if s in combined.columns]
result_multi = stats_analysis.complete_statistical_table(combined, a_samples_multi, b_samples_multi)
assert result_multi.shape[0] == combined.shape[0]
print(f"C6. Statistics on combined multi-dataset matrix OK: {result_multi.shape}, "
      f"{result_multi['Significant'].sum()} significant")

pca_multi, scores_multi, _ = pca_module.run_pca(combined, n_components=5)
print("C7. PCA on combined multi-dataset matrix OK:", scores_multi.shape)

qc_comparison = dataset_manager.qc_comparison_table(datasets)
assert len(qc_comparison) == len(demo_types)
print("C8. Cross-dataset QC comparison table OK:\n", qc_comparison.to_string(index=False))

print("\nALL PIPELINE STEPS PASSED (targeted + untargeted, 4-group demo datasets; "
      "4-analysis-type multi-dataset combine workflow; PDF report removed).")
