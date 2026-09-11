# MetaboAI Pro

**Automated LC-MS Metabolomics & Lipidomics Statistical Analysis and Reporting Platform**

Converts raw LC-MS peak area datasets into statistically validated, biologically
interpretable, publication-ready results. Supports untargeted/targeted metabolomics
and untargeted/targeted lipidomics workflows.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt` deliberately pins **nothing** — just package names, no versions.
Earlier rounds tried exact pins matched to a specific Python version, but each
package's version was chosen independently by searching "does X support Python
3.14/3.12" — with no way to verify the *combination* resolves without conflict
(streamlit, scikit-learn, etc. each carry their own transitive version
requirements). An unpinned file hands that resolution problem to pip, which is
built for exactly this — it'll pick a mutually-compatible set for whatever Python
Streamlit Cloud actually runs, rather than being locked to a snapshot no one has
actually installed together. `runtime.txt` requests Python 3.12 (Cloud's own
default) but isn't load-bearing if you'd rather manage the version from Cloud's
"Advanced settings" instead.

If you need a fully reproducible environment later (e.g. for CI), run
`pip freeze > requirements.lock.txt` after a successful install and use that
file instead — but start unpinned.

The app opens with a 12-tab workflow. Check "Use built-in demo dataset" on the first
tab to explore the full pipeline immediately — two demo options are offered:
- **Standard**: 150 metabolites, 16 study samples across Control/Disease groups, 6 QC replicates.
- **Rich Clinical Demo**: 200 metabolites, 36 biological samples across 4 diagnosis
  groups (Healthy Control, Prediabetic, Type 2 Diabetes, Metabolic Syndrome, 9 each)
  + 6 QC replicates = 42 total. Metadata includes Diagnosis, Age, Gender, Treatment,
  Ethnicity, and Body Weight, plus a metabolite-level Method + Pathway row-annotation
  file — built to exercise the **dynamic grouping variable** support described below.

**Any categorical metadata column can drive grouping/coloring**, not just a column
literally named "Group". PCA ("Color/group samples by:"), Statistics ("Grouping
variable:"), Heatmap ("Filter samples by:"), and Boxplot ("Grouping variable:") each
expose a selector populated from whichever categorical columns your metadata
actually has (auto-detected — non-numeric columns, or numeric columns with ≤15
distinct values among biological samples, so continuous covariates like Age or Body
Weight are correctly excluded from the grouping-variable list even though they're
still visible in the metadata table, and still usable as a Heatmap annotation track
— see below). Volcano Plot and Biomarker Discovery inherit whichever variable
Statistics last used and say so in a caption. Heatmap's default sample-filter
variable follows Statistics' choice too, but can be overridden independently. This
works with only `Group`/`IsQC`/`Batch` present (the Standard demo, and any dataset
using the older simpler schema) exactly as before — `Group` is always offered first
when present — richer metadata just adds more options, it isn't required.

**Heatmap: multiple stacked row & column annotation tracks**, matching how
published figures typically layer several sample/feature attributes at once rather
than showing just one. "Column annotation tracks:" is a multiselect over every
metadata column (any number, in any combination) — each renders as its own stacked
strip directly above the heatmap, auto-classified continuous (numeric, many
distinct values — e.g. Age, Body Weight — rendered as a color gradient with its own
small labeled colorbar) or categorical (e.g. Diagnosis, Gender, Treatment,
Ethnicity — rendered as discrete color blocks with a swatch legend). Upload an
optional metabolite/feature row-annotation table in Tab 1 (columns: `Metabolite`,
then any annotation columns — e.g. `Method`, `Pathway`) to unlock "Row annotation
tracks:", the same multi-track multiselect for the row (feature) side — useful for
seeing whether significant metabolites cluster by analytical method, biological
pathway, or any other feature-level grouping you provide; each row track gets its
own rotated label above the strip, same as column tracks get a label beside theirs.
In multi-dataset mode, row-annotation matching tries the full `Dataset::Feature`
name first, then falls back to just the part after `::`, so a plain (non-prefixed)
annotation file still matches correctly against combined data. Every track gets its
own legend block, stacked in one column to the right without overlap (the figure
grows taller automatically if you pick enough tracks that they need more room than
the heatmap provides). Features not covered by the row-annotation file, or samples
missing a value for a selected column, show as "(unannotated)"/"(missing)" in
neutral gray rather than causing an error. A soft on-figure note appears past 6
tracks on either axis suggesting you trim for readability, but there's no hard cap.
A compact "🎨 Annotation Colors" expander appears once any track is selected —
tracks lay out in a 2-6 column grid (name + small swatches on one row per track;
continuous tracks get a colormap dropdown instead of swatches), so customizing
several tracks doesn't take over the screen. Edits are staged, not live — nothing
changes on the heatmap until you click **Apply Colors** (avoids re-rendering the
whole figure on every single swatch click), and **Reset Colors** clears back to
the automatic palette, both pickers and the rendered heatmap together. "Font
family"/"Base font size" controls scale the title, tick labels, legends, and track
labels together from one setting. The title itself is just "Heatmap of
Metabolites", dynamically suffixed with the feature count and whichever
significance cutoff is selected, e.g. "Heatmap of Metabolites (41 features,
FDR ≤ 0.25)".

## Workflow (tabs, left to right)

**Multi-dataset mode**: Tab 1 offers a choice between a single dataset and multiple
datasets (e.g. combining several assay types — Untargeted Metabolomics, Targeted
Metabolomics, Untargeted Lipidomics, Targeted Lipidomics — run on the same biological
samples). Choosing "Multiple datasets" splits into two clearly separate paths:
**Demo Data** (loads the 4 built-in analysis-type datasets) and **Real Data**
(upload your own — a dedicated, unmissable metadata upload sits right below the
per-dataset file uploaders, since every dataset needs it to identify QC vs.
biological samples and group assignments).

- **Dataset dropdown, not nested tabs**: In Tabs 2, 3, and 5 (Cleaning, QC,
  Normalization), a status dashboard at the top shows every dataset's progress across
  all three stages (⚪ not started / 🟡 in progress / ✅ done) at a glance, and a single
  dropdown below it lets you pick which dataset to configure — each dataset is
  cleaned, QC'd, and normalized **fully independently** (different methods/modes have
  different intensity scales and technical characteristics), never mixed with
  another until you explicitly ask.
- **🚀 Quick Start**: once datasets are loaded, a single button in Tab 1 runs the
  recommended default pipeline (missingness filter → imputation → QC CV review →
  ISTD/Median-IQR normalization → Log2) on **every** dataset **independently** — it
  does not combine them.
- **Combined tabs are the required downstream path.** Multi-dataset mode has two
  dedicated combine tabs, each gated the same way:
  - **Tab 4 — Combined QC**: pools every dataset's QC diagnostics (CV distribution,
    feature counts by quality, sample correlation, peak-area distribution) once
    they've been reviewed per-dataset in Tab 3.
  - **Tab 6 — Combined Normalized Data**: has a **"🔗 Generate Combined Normalized
    Data"** button that stays **disabled** until *every* loaded dataset has completed
    its own independent normalization (a status table shows exactly what's still
    pending). Once generated, this combined table becomes the **single required
    input** for every downstream tab (PCA, Statistics, Volcano, Biomarker Discovery,
    Heatmap, Boxplot) — those tabs explicitly warn rather than
    silently falling back to raw data or a single dataset if you haven't generated
    it yet. Metabolite names are prefixed by dataset label (e.g.
    `Targeted Metabolomics::Glucose`, since the same compound name can be a
    genuinely different measurement across methods/ionization modes) so you can
    always tell which dataset a row came from. Both the pre-Log2 and Log2-transformed
    combined tables are downloadable as CSV. Only normalized data can ever be
    combined this way — raw/unnormalized datasets are never mixed together.
  This all works identically regardless of how many datasets you loaded (2, 3, 4, or
  more) — nothing is hard-coded to a specific count.

A built-in multi-method demo (4 datasets — one per analysis type, 24 shared
biological samples) is available for a one-click walkthrough: Tab 1 → Multiple
datasets → Demo Data → "Load Demo Datasets", then "🚀 Auto-Process All Datasets",
then Tab 6 → "Generate Combined Normalized Data" to unlock downstream analysis.

PCA, Volcano Plot, Heatmap, and Boxplot update **in real time** — figures redraw
automatically as you change any customization control, with no "Generate" button to
click. Every plot (QC, Combined QC, PCA) has a **format (PNG/TIFF/SVG/PDF) and DPI
(150/300/600) selector** next to its download button, for publication-ready exports.


1. **Data Upload** — Upload a peak area matrix (CSV/XLSX, rows = metabolites,
   columns = samples) and a metadata table (`Sample, Group, IsQC, Batch`). Two
   built-in demo datasets are included and **selected automatically** based on the
   analysis type you pick:
   - **Targeted** Metabolomics/Lipidomics → 60 features incl. an ISTD row, 24
     biological samples across 4 groups (Control/Mild/Moderate/Severe), 6 QC
     replicates.
   - **Untargeted** Metabolomics/Lipidomics → 300 features, no ISTD row (untargeted
     assays have no single spiked standard), same 4-group/QC design.
   Both demos include enough groups to exercise ANOVA (≥3 groups) directly, not just
   the two-group t-test. For multi-dataset mode, see above — a separate built-in
   4-dataset demo (one dataset per analysis type: Untargeted Metabolomics, Targeted
   Metabolomics, Untargeted Lipidomics, Targeted Lipidomics) is available.
2. **Data Cleaning & Missing Value Imputation** (optional, runs on biological samples
   before QC) — Missing values (NaN or exact zero, both common LC-MS conventions for
   "not detected") are handled in two steps:
   - **Filter by missingness**: a per-metabolite missingness table with an adjustable
     threshold slider, following standard guidance (<20% keep · 20-50% keep with
     careful imputation · 50-70% usually remove · >70-80% remove unless essential).
   - **Impute remaining missing values** using one of six methods: Half-Minimum
     (LOD/2, the most common LC-MS convention), Mean, Median, K-Nearest Neighbors,
     Random Forest (MissForest-style, via scikit-learn's IterativeImputer +
     RandomForestRegressor — no official `missForest` package exists outside R), BPCA
     (approximated via IterativeImputer + BayesianRidge — a reasonable regression-based
     approximation, not the exact original algorithm), and QRILC (approximated —
     draws left-censored values from a truncated distribution reflecting the
     below-detection-limit assumption; the exact quantile-regression algorithm from
     R's `imputeLCMD` package isn't available in Python). All methods clip to
     non-negative output, since peak areas can't be negative.
   If skipped, QC validation and normalization fall back to the raw biological data.
   The complete post-imputation table is shown in full (not just a preview) and is
   individually downloadable per dataset in multi-dataset mode.
3. **QC Validation** — Coefficient of Variation (CV) across QC replicates, CV
   distribution, feature counts by CV quality, a **QC sample correlation matrix**, and
   a **peak-area distribution histogram** (log10 scale) — **CV/correlation are
   computed on QC replicates only** (biological samples are never mixed into those
   diagnostics), while the peak-area histogram covers all values. Every plot has a
   format (PNG/TIFF/SVG/PDF) and DPI (150/300/600) selector next to its download
   button. Optionally filter out features with CV > 20% — **unchecked by default,
   so all features are kept unless you explicitly opt in**; this filtering applies
   on top of whatever came out of Tab 2 (cleaned/imputed if you ran it, raw
   otherwise). Confirming this step **excludes all QC samples** from every tab
   downstream. In multi-dataset mode, each dataset gets its own QC section here
   (via the dataset dropdown) — see Tab 4 for cross-dataset comparisons.
4. **Combined QC** (multi-dataset mode only) — pools every dataset's CV table and
   QC-only log2 data (dataset-prefixed) to compute the same analyses across all
   datasets together: QC CV Distribution, Feature Counts by CV Quality, QC Sample
   Correlation Matrix, and a combined peak-area histogram, plus a side-by-side
   cross-dataset comparison table. Every plot has the same format/DPI download
   controls as Tab 3, and the underlying data is downloadable as CSV.
5. **Normalization** — branches automatically by analysis type selected in Tab 1:
   - **Targeted Metabolomics/Lipidomics**: ISTD normalization (endogenous ÷ ISTD peak
     area) → Log2 transform.
   - **Untargeted Metabolomics/Lipidomics**: Median-IQR normalization (robust
     scaling, `(x − median)/IQR`, computed per metabolite across samples, applied to
     the cleaned/QC-validated peak areas) → Log2 transform. Since Median-IQR
     normalized values are frequently negative (any point below the median), the data
     is automatically shifted by a constant just large enough to make the global
     minimum slightly positive before logging — this preserves every value's relative
     position while making log2 well-defined.
   An "Advanced: override" checkbox exposes both paths if needed. Only biological
   samples ever reach this tab (QC already excluded in Tab 3). In multi-dataset mode,
   each dataset is normalized independently here (via the dataset dropdown) — see
   Tab 6 to combine them.
6. **Combined Normalized Data** (multi-dataset mode only) — the **required final
   step** before any downstream analysis: a **"🔗 Generate Combined Normalized Data"**
   button, disabled until every dataset has completed Tab 5, that combines all of
   them into one unified table (metabolite names prefixed by dataset label). Produces
   both a pre-Log2 and a Log2-transformed combined table, both shown in full and
   downloadable as CSV — the Log2 table is what feeds every tab from PCA onward.
7. **PCA** — Choose any subset of groups to include (2, 3, 4, or more), score plot
   (biological samples only — QC has its own correlation matrix in Tab 3), customizable color
   palette (5 presets) and per-group marker style, optional 95% confidence ellipses,
   loading plot (top contributing metabolites), and variance-explained plot. Every
   plot has a format/DPI download control; the sample scores (with group labels),
   % variance explained per component, the loadings table, and the exact input data
   matrix are all downloadable as CSV.
8. **Statistics** — Two-group comparison (Student's t-test or Wilcoxon rank-sum,
   choosing any 2 of the available groups) or one-way ANOVA (choose any 3+ groups)
   with Tukey HSD / Dunnett / pairwise post-hoc tests.
   **Every statistic — mean abundance, linear/log2 fold change, p-value, FDR, and 95%
   confidence interval — is computed from the log2-transformed, normalized data.** Raw
   peak areas are stored for traceability only and never feed into inference. Every
   downloadable result — the two-group table, the ANOVA table, and post-hoc results
   — has the comparison (e.g. `Control_vs_Severe`) or groups compared baked into the
   filename, so results from different comparisons never get confused with each other.
9. **Volcano Plot** — publication-grade, journal-oriented customization:
   - **Thresholds**: user-defined fold-change and significance cutoffs, optional
     secondary FDR requirement, p-value or FDR on the y-axis, log2FC or linear FC on
     the x-axis, selectable threshold line style
   - **Colors**: 5 palettes including colorblind-safe options, individual
     up/down/NS color overrides, adjustable transparency, point border color/width
   - **Point style**: size, shape (circle/triangle/square/diamond/cross)
   - **Labels**: top-N / all-significant / manually-selected-by-name modes, font
     size, bold/italic, color override, automatic overlap-avoidance (label repelling)
   - **Legend**: right/left/top/bottom/hidden placement, full or short label format
   - **Axes & title**: custom titles/subtitles, font size, bold, custom axis limits
     and tick spacing, decimal precision, alignment, hide option
   - **Gridlines & threshold lines**: on/off/major/minor, custom color/style/width
   - **Highlight specific metabolites** (e.g. known markers) with a distinct
     color/size/shape
   - **Background**: white/transparent/gray/custom (transparent useful for
     Illustrator workflows)
   - **Figure size presets**: single column, double column, presentation, poster,
     or custom inches
   - **Publication theme presets**: stylistic approximations of Nature/Cell/Cancer
     Research/Clinical Cancer Research/PNAS (not exact journal specifications)
   - **On-figure statistics box**: total/up/down counts and cutoffs
   - **Export**: figure as PNG/PDF/SVG/EPS/JPEG/TIFF at 300/600/1200 DPI; data as
     Upregulated/Downregulated/Complete CSVs; figure settings as JSON for
     reproducibility
   - Not included (would need a different architecture): true interactive
     hover/click (would require Plotly/Bokeh instead of static matplotlib),
     HMDB/KEGG/pathway-based labeling or filtering (no compound-database
     integration in this app), VIP-based filtering (needs a PLS-DA model), and
     freehand in-app annotation (arrows/shapes) — labeling and highlighting work
     by compound name instead.
10. **Biomarker Discovery** — Filter significant metabolites (from the same two
    selected groups in Statistics) by p-value, FDR, or combined (p<0.05 AND
    FDR<0.25) criteria. Downloadable with the comparison name baked into the
    filename (e.g. `Control_vs_Severe_Biomarkers.csv`).
11. **Heatmap** — Choose any subset of groups to include (2, 3, 4, or more) before
   rendering. Optionally set an exact width/height (inches) for the heatmap panel —
   dendrograms, annotation bar, legend, colorbar, and labels all rescale
   automatically to match. Fully customizable clustered heatmap of significant
   metabolites:
   - Independent row/column clustering (both, rows only, columns only, or none)
   - Euclidean/Ward or correlation/average distance metrics
   - Group annotation bar with legend
   - Predefined palettes or a custom 3-color gradient builder, with reverse toggle
   - Adjustable Z-score color range and optional discrete color breakpoints
   - Horizontal colorbar below the plot, labeled with Z-score
   - Export to PNG, PDF, SVG, JPEG, or TIFF (150/300/600 DPI)
   - **Downloadable Z-score table**: the exact row-scaled Z-score values used to
     render the heatmap (same clustered row/column order) as CSV
   - **Large-panel safety limits**: with hundreds of significant features, the panel
     height/width auto-caps at 30in/24in (uncapped scaling previously produced
     100+ inch figures that could crash on export), and row/column labels
     auto-hide above 150/120 items since they'd be unreadable anyway — a preview
     metric and warning appear before you generate, and any auto-adjustments are
     reported after. TIFF export uses lossless LZW compression (~100x smaller
     files) and DPI auto-reduces if a request would demand an excessive raw pixel
     buffer, preventing out-of-memory crashes during export.
12. **Boxplot of Metabolites** — Select one or more metabolites and any subset of
   groups (2, 3, 4, or more) to compare. Statistics (Welch's t-test for 2 groups,
   one-way ANOVA for 3+) are computed on log2-transformed data and shown directly on
   each panel — note the FDR here is corrected only across the metabolites you've
   selected for display, not the full feature panel (use the Statistics tab for a
   panel-wide FDR). Customizable: figure width/height, font size/family, per-group
   box colors, individual data points, mean/median overlay, panels-per-row, and
   export to PNG/PDF/SVG/JPEG/TIFF. Every step in this app (Cleaning, QC, Combined
   QC, Normalization, Combined Normalized Data, PCA, Statistics, Volcano, Biomarker
   Discovery, Heatmap, Boxplot) has its own downloads — there's no separate
   all-in-one export tab.

## Project structure

```
MetaboAI_Pro/
├── app.py                     # Streamlit UI (12-tab workflow)
├── modules/
│   ├── utils.py                # Data loading, validation, zero replacement
│   ├── qc.py                   # CV calculation, QC distribution/histogram/correlation matrix
│   ├── imputation_module.py    # Data cleaning & missing value imputation (6 methods)
│   ├── dataset_manager.py      # Multi-dataset (method/mode) combination logic
│   ├── normalization.py        # ISTD, IQR (feature/sample/batch), Log2 transform
│   ├── stats_analysis.py       # t-test/Wilcoxon, ANOVA + post-hoc, BH-FDR
│   ├── pca_module.py           # PCA score/loading/variance plots with ellipses
│   ├── volcano.py              # p-value and FDR based volcano plots
│   ├── biomarker.py            # Statistical filtering for biomarker discovery
│   ├── heatmap_module.py       # Clustered heatmap (Euclidean/Ward, correlation)
│   └── boxplot_module.py       # Per-metabolite boxplot comparisons across groups
├── make_sample_data.py        # Synthetic demo dataset generator (targeted, untargeted,
│                               #   and a 4-dataset multi-method demo covering all four
│                               #   analysis types)
├── sample_peak_area_matrix_targeted.csv
├── sample_metadata_targeted.csv
├── sample_peak_area_matrix_untargeted.csv
├── sample_metadata_untargeted.csv
├── sample_metadata_multimethod.csv
├── sample_peak_area_matrix_multimethod_untargeted_metabolomics.csv
├── sample_peak_area_matrix_multimethod_targeted_metabolomics.csv
├── sample_peak_area_matrix_multimethod_untargeted_lipidomics.csv
├── sample_peak_area_matrix_multimethod_targeted_lipidomics.csv
├── test_pipeline.py           # Headless smoke test of the full pipeline
└── requirements.txt
```

## Input format notes

- **Peak area matrix**: first column is the metabolite/lipid identifier; every other
  column is a sample (including QC samples). Values must be numeric peak areas.
- **Metadata table**: one row per sample. `Sample` must match a column name in the
  peak area matrix. `IsQC` marks QC replicates (True/False, 1/0, "QC", etc.).
  `Batch` is optional and only used for batch-specific IQR normalization.
- To designate an internal standard for ISTD normalization, include it as a row in
  the peak area matrix and select it by name in the Normalization tab.

## Notes on statistical conventions

- Significance in the two-group table and biomarker module defaults to
  **p < 0.05 AND FDR < 0.25**, matching common exploratory metabolomics practice —
  adjust cutoffs in the UI as needed for your study.
- Two-group results include a 95% confidence interval (`CI_Lower_Log2FC`,
  `CI_Upper_Log2FC`) on the log2 fold change, computed via Welch's t formula.
- ANOVA post-hoc tests are only run for features with FDR < 0.25 to keep runtime
  reasonable on large feature sets.
- Log2 transformation replaces zeros/missing values with half the minimum positive
  value observed (per feature) before transforming, avoiding -Inf.
- QC samples are used exclusively for the QC Validation tab's reproducibility
  diagnostics and are excluded from every other tab (normalization, statistics, PCA,
  volcano, biomarker discovery, heatmap) once QC is confirmed.

## Testing

Run `python test_pipeline.py` for a headless smoke test that exercises every module
against the synthetic demo dataset (useful in environments without a display/Streamlit
server).
