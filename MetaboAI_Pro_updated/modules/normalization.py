"""
normalization.py - ISTD normalization, IQR normalization, and Log2 transformation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# 1. ISTD normalization
# ---------------------------------------------------------------------------
def istd_normalize(peak_df: pd.DataFrame, istd_name: str) -> pd.DataFrame:
    """
    Normalized Peak Area = Endogenous Metabolite Peak Area / Internal Standard (ISTD) Peak Area.

    Every row except the ISTD itself is treated as an endogenous metabolite and divided,
    sample-by-sample, by that sample's ISTD peak area. istd_name must be a row (feature)
    present in peak_df.
    """
    if istd_name not in peak_df.index:
        raise ValueError(f"Internal standard '{istd_name}' not found among features.")
    istd_row = peak_df.loc[istd_name]
    normalized = peak_df.drop(index=istd_name).div(istd_row.replace(0, np.nan), axis=1)
    return normalized


# ---------------------------------------------------------------------------
# 2. IQR normalization
# ---------------------------------------------------------------------------
def iqr_normalize(df: pd.DataFrame, axis: str = "feature", batch_map: pd.Series = None) -> pd.DataFrame:
    """
    Median-IQR normalization (a.k.a. robust scaling):
        X_norm = (X - Median(X)) / IQR(X)

    Recommended usage: apply this AFTER log2 transformation, not before. Robust-scaled
    values are frequently negative (anything below the median), so taking log2 of the
    OUTPUT of this function will produce NaNs for roughly half the data. If you need
    both steps, always log2 first, then robust-scale the log2 values.

    axis:
      - 'feature' : normalize each metabolite (row) across samples — the standard choice,
                    matches X_ij = (X_ij - Median(X_j)) / IQR(X_j) computed per metabolite j
                    across all samples i.
      - 'sample'  : normalize each sample (column) across features
      - 'batch'   : normalize each feature within each batch separately (batch_map required:
                    Series indexed by sample name -> batch label)
    """
    if axis == "feature":
        median = df.median(axis=1)
        q1 = df.quantile(0.25, axis=1)
        q3 = df.quantile(0.75, axis=1)
        iqr = (q3 - q1).replace(0, np.nan)
        return df.sub(median, axis=0).div(iqr, axis=0)

    elif axis == "sample":
        median = df.median(axis=0)
        q1 = df.quantile(0.25, axis=0)
        q3 = df.quantile(0.75, axis=0)
        iqr = (q3 - q1).replace(0, np.nan)
        return df.sub(median, axis=1).div(iqr, axis=1)

    elif axis == "batch":
        if batch_map is None:
            raise ValueError("batch_map (sample -> batch label) is required for batch-specific normalization.")
        out = df.copy()
        for batch in batch_map.unique():
            cols = batch_map.index[batch_map == batch].tolist()
            cols = [c for c in cols if c in df.columns]
            if not cols:
                continue
            sub = df[cols]
            median = sub.median(axis=1)
            q1 = sub.quantile(0.25, axis=1)
            q3 = sub.quantile(0.75, axis=1)
            iqr = (q3 - q1).replace(0, np.nan)
            out[cols] = sub.sub(median, axis=0).div(iqr, axis=0)
        return out

    else:
        raise ValueError("axis must be one of 'feature', 'sample', 'batch'")


# ---------------------------------------------------------------------------
# 3. Log2 transformation
# ---------------------------------------------------------------------------
def log2_transform(df: pd.DataFrame, constant: float = None, auto_zero_replace: bool = True):
    """
    Apply log2(x + constant). If constant is None, it is automatically set to a small
    fraction of the smallest positive value in the dataset (minimum value adjustment).
    This handles exact zeros (the common "not detected" convention) automatically:
    log2(0 + constant) is well-defined and represents "at or below the minimum
    detected level" — no separate zero-replacement step is needed once a sensible
    constant is chosen. Genuine NaN (true missing data) still propagates as NaN;
    handle real missing values via the Data Cleaning & Imputation step before this.
    Returns (transformed_df, constant_used).
    """
    work = df.copy()
    if auto_zero_replace:
        positive_vals = work.values[(work.values > 0) & (~np.isnan(work.values))]
        min_pos = positive_vals.min() if positive_vals.size else 1.0
        if constant is None:
            constant = min_pos * 0.5 if min_pos > 0 else 1.0
    else:
        if constant is None:
            constant = 1.0

    transformed = np.log2(work + constant)
    return transformed, constant


def shift_and_log2_transform(df: pd.DataFrame, shift: float = None):
    """
    Shift data to be strictly positive (if needed), then log2 transform.

    Use this — instead of log2_transform() — for data that has already passed through
    a CENTERING normalization such as the classic Median-IQR formula,
    (X - Median) / IQR, which produces negative values for any point below the median.
    log2 is undefined for those without an offset; log2_transform()'s zero-replacement
    logic would incorrectly treat all negative values as missing data and clobber them
    to a single small constant. This function instead shifts the entire dataset by a
    constant just large enough to make the global minimum slightly positive, preserving
    every value's relative position, then logs the shifted data.

    If the data is already all-positive, shift defaults to 0 (equivalent to a plain
    log2 with no zero-handling — use log2_transform() instead if you need automatic
    zero replacement for genuinely-zero raw values).

    Returns (transformed_df, shift_used).
    """
    finite_vals = df.values[np.isfinite(df.values)]
    min_val = finite_vals.min() if finite_vals.size else 0.0
    if shift is None:
        if min_val <= 0:
            shift = abs(min_val) + max(abs(min_val) * 0.01, 1e-3)
        else:
            shift = 0.0
    shifted = df + shift
    transformed = np.log2(shifted)
    return transformed, shift


def distribution_plots(before: pd.DataFrame, after: pd.DataFrame, sample_id: str = None):
    """
    Generate before/after density and box plots, each on ITS OWN appropriately-scaled
    panel. Raw peak areas (before) typically span several orders of magnitude while
    log2-transformed values (after) span a much smaller range (~0-30); overlaying them
    on a single shared axis makes one distribution collapse to an invisible sliver.
    Using separate panels (with a log x-axis for the raw-scale density, since raw
    abundances are all positive and right-skewed) keeps both distributions legible.
    """
    b = before.values.flatten()
    b = b[~np.isnan(b) & (b > 0)]
    a = after.values.flatten()
    a = a[~np.isnan(a)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Before: density (log-x, since raw peak areas are positive and span orders of magnitude)
    if b.size:
        axes[0, 0].hist(b, bins=60, color="#C44E52", alpha=0.8)
        axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("Before: Density (raw peak area, log scale)")
    axes[0, 0].set_ylabel("Count")

    # Before: box plot (own y-axis, raw scale)
    if b.size:
        try:
            axes[0, 1].boxplot([b], tick_labels=["Before"])
        except TypeError:
            axes[0, 1].boxplot([b], labels=["Before"])
        axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Before: Box Plot (raw peak area, log scale)")

    # After: density (linear x, already log2 scale)
    if a.size:
        axes[1, 0].hist(a, bins=60, color="#55A868", alpha=0.8)
    axes[1, 0].set_title("After: Density (log2-transformed)")
    axes[1, 0].set_xlabel("log2(peak area)")
    axes[1, 0].set_ylabel("Count")

    # After: box plot (own y-axis, log2 scale)
    if a.size:
        try:
            axes[1, 1].boxplot([a], tick_labels=["After"])
        except TypeError:
            axes[1, 1].boxplot([a], labels=["After"])
    axes[1, 1].set_title("After: Box Plot (log2-transformed)")

    fig.tight_layout()
    return fig
