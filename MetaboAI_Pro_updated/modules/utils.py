"""
utils.py - Core data loading, validation, and helper utilities for MetaboAI Pro.

Expected input formats
-----------------------
Peak area matrix (CSV/XLSX):
    Metabolite, Sample1, Sample2, QC1, QC2, ...
    (rows = metabolites/features, columns = samples)

Metadata table (CSV/XLSX):
    Sample, Group, IsQC, Batch
    - Sample   : must match a column name in the peak area matrix
    - Group    : experimental group / condition label
    - IsQC     : True/False (or 1/0) flag for QC samples
    - Batch    : optional batch identifier (used for batch-specific normalization)
"""

import io
import numpy as np
import pandas as pd


REQUIRED_METADATA_COLS = ["Sample", "Group"]


class DataValidationError(Exception):
    pass


def load_table(file_or_buffer, filename_hint: str = "") -> pd.DataFrame:
    """
    Load a CSV or Excel file into a DataFrame, auto-detecting format and, for CSV,
    auto-detecting text encoding. Many lab instrument/software exports (Excel "CSV"
    saves, older Windows tools) use Windows-1252/Latin-1, not UTF-8 -- e.g. '±' (as
    in 'mean ± SD'), 'µ' (micro), or curly quotes are classic culprits that raise
    UnicodeDecodeError under pandas' UTF-8 default. Falls back through common
    encodings in order; Latin-1 can decode any byte sequence, so this never raises
    UnicodeDecodeError itself (a genuinely corrupt/binary file will instead fail
    validate_peak_matrix/validate_metadata with a clearer structural error).
    """
    name = filename_hint.lower() if filename_hint else getattr(file_or_buffer, "name", "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file_or_buffer)

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            if hasattr(file_or_buffer, "seek"):
                file_or_buffer.seek(0)
            return pd.read_csv(file_or_buffer, encoding=encoding)
        except UnicodeDecodeError:
            continue
    # Unreachable in practice (latin-1 accepts every byte value 0-255), but keeps
    # the function's contract honest if that ever changes.
    if hasattr(file_or_buffer, "seek"):
        file_or_buffer.seek(0)
    return pd.read_csv(file_or_buffer)


def validate_peak_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw peak area matrix. First column must be metabolite identifiers."""
    if df.shape[1] < 2:
        raise DataValidationError("Peak area matrix must have a metabolite column plus at least one sample column.")
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Metabolite"})
    df["Metabolite"] = df["Metabolite"].astype(str)
    if df["Metabolite"].duplicated().any():
        dupes = df["Metabolite"][df["Metabolite"].duplicated()].unique().tolist()
        raise DataValidationError(f"Duplicate metabolite names found: {dupes[:5]}...")
    sample_cols = df.columns[1:]
    non_numeric = [c for c in sample_cols if not pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]
    for c in sample_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("Metabolite")


def validate_metadata(meta: pd.DataFrame, sample_cols) -> pd.DataFrame:
    """Validate metadata table and align it to the sample columns present in the peak matrix."""
    missing = [c for c in REQUIRED_METADATA_COLS if c not in meta.columns]
    if missing:
        raise DataValidationError(f"Metadata missing required column(s): {missing}")
    if "IsQC" not in meta.columns:
        meta["IsQC"] = False
    else:
        meta["IsQC"] = meta["IsQC"].astype(str).str.lower().isin(["true", "1", "yes", "qc"])
    if "Batch" not in meta.columns:
        meta["Batch"] = "1"
    meta = meta.set_index("Sample")
    missing_samples = [s for s in sample_cols if s not in meta.index]
    if missing_samples:
        raise DataValidationError(f"Samples present in peak matrix but missing from metadata: {missing_samples}")
    return meta.loc[list(sample_cols)]


def split_qc_and_samples(peak_df: pd.DataFrame, meta: pd.DataFrame):
    """Return (qc_columns, sample_columns) based on metadata IsQC flag."""
    qc_cols = meta.index[meta["IsQC"]].tolist()
    sample_cols = meta.index[~meta["IsQC"]].tolist()
    return qc_cols, sample_cols


def get_categorical_metadata_columns(meta: pd.DataFrame, sample_cols=None, max_unique: int = 15) -> list:
    """
    Which metadata columns are usable as a grouping/coloring variable (for PCA,
    Statistics, Heatmap column annotation, Boxplot) -- not just the hardcoded
    'Group' column. A column qualifies if, among biological samples only (QC rows'
    placeholder values like 'QC'/NaN are excluded from this check), it's non-numeric
    (object/category dtype -- covers text labels like Diagnosis, Gender, Treatment,
    Ethnicity) or numeric with few enough distinct values to plausibly be a
    group/category rather than a continuous measurement (excludes Age, Body Weight,
    and similar continuous covariates, which aren't meaningful t-test/ANOVA/boxplot
    groups). 'Group' is always included first if present, for a stable default.
    """
    if sample_cols is None:
        sample_cols = meta.index.tolist()
    bio_meta = meta.loc[meta.index.intersection(sample_cols)]
    candidates = []
    for col in meta.columns:
        if col in ("IsQC",):
            continue
        series = bio_meta[col].dropna()
        if series.empty:
            continue
        n_unique = series.nunique()
        if n_unique < 2 or n_unique > len(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            if n_unique <= max_unique:
                candidates.append(col)
        else:
            candidates.append(col)
    # stable, predictable ordering: Group first (if present), then the rest as-authored
    ordered = [c for c in ("Group",) if c in candidates]
    ordered += [c for c in candidates if c not in ordered]
    return ordered


def zero_replacement(df: pd.DataFrame, method: str = "min_fraction", fraction: float = 0.5) -> pd.DataFrame:
    """
    Replace zeros / missing values prior to log transform.
    method:
      - 'min_fraction': replace with fraction * (smallest non-zero value in that feature's row)
      - 'global_min': replace with fraction * (smallest non-zero value across whole matrix)
    """
    out = df.copy()
    if method == "min_fraction":
        for idx in out.index:
            row = out.loc[idx]
            nonzero = row[(row > 0) & row.notna()]
            fill_val = nonzero.min() * fraction if len(nonzero) else np.nan
            out.loc[idx] = row.where((row > 0) & row.notna(), fill_val)
    else:  # global_min
        nonzero = out.values[(out.values > 0) & (~np.isnan(out.values))]
        fill_val = nonzero.min() * fraction if nonzero.size else np.nan
        out = out.where((out > 0) & out.notna(), fill_val)
    return out


def to_download_bytes_csv(df: pd.DataFrame) -> bytes:
    """
    Serialize a DataFrame to CSV bytes for st.download_button, with a UTF-8 BOM
    (utf-8-sig). Metabolite/lipid names routinely contain Greek letters (α, β, Δ),
    symbols (±, µ, °), or accented Latin characters -- plain 'utf-8' without a BOM
    is valid and round-trips fine in Python/pandas, but Excel (still the most common
    tool users re-open these exports in) assumes the system locale encoding for a
    plain UTF-8 CSV and renders those characters as mojibake unless a BOM marks it
    explicitly as UTF-8. The BOM is a no-op for pandas/any UTF-8-aware reader.
    """
    return df.to_csv().encode("utf-8-sig")


def to_download_bytes_xlsx(sheets: dict) -> bytes:
    """sheets: dict of {sheet_name: DataFrame}"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, d in sheets.items():
            safe_name = name[:31]
            d.to_excel(writer, sheet_name=safe_name)
    buf.seek(0)
    return buf.read()
