"""
normalization.py
ISTD normalization, Log2 transformation, and IQR normalization.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. ISTD normalization
# ---------------------------------------------------------------------------
def istd_normalize(peak_df: pd.DataFrame, istd_name: str) -> pd.DataFrame:
    """
    ISTD normalization:

        Normalized Peak Area = Endogenous Peak Area / ISTD Peak Area

    The ISTD is removed from the output.
    """

    if istd_name not in peak_df.index:
        raise ValueError(
            f"Internal standard '{istd_name}' not found among features."
        )

    istd_row = peak_df.loc[istd_name]

    # Avoid division by zero
    istd_row = istd_row.replace(0, np.nan)

    normalized = (
        peak_df
        .drop(index=istd_name)
        .div(istd_row, axis=1)
    )

    return normalized


# ---------------------------------------------------------------------------
# 2. Log2 transformation
# ---------------------------------------------------------------------------
def log2_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a direct log2(x) transformation.

        X_log2 = log2(X)

    No constant, pseudocount, or zero replacement is applied.

    Values <= 0 are converted to NaN because log2(x) is undefined
    for zero and negative values.

    NaN values remain NaN.
    """

    work = df.copy()

    # Values <= 0 cannot be log2 transformed
    work = work.mask(work <= 0)

    transformed = np.log2(work)

    return transformed


# ---------------------------------------------------------------------------
# 3. IQR normalization
# ---------------------------------------------------------------------------
def iqr_normalize(
    df: pd.DataFrame,
    axis: str = "feature",
    batch_map: pd.Series = None
) -> pd.DataFrame:
    """
    Median-IQR normalization:

        X_norm = (X - Median) / IQR

    Recommended workflow:

        Raw Peak Area
            ↓
        ISTD normalization
            ↓
        log2(x)
            ↓
        IQR normalization

    axis:
        'feature' : normalize each metabolite across samples
        'sample'  : normalize each sample across metabolites
        'batch'   : normalize each metabolite within each batch
    """

    if axis == "feature":

        median = df.median(axis=1)
        q1 = df.quantile(0.25, axis=1)
        q3 = df.quantile(0.75, axis=1)

        iqr = (q3 - q1).replace(0, np.nan)

        normalized = (
            df
            .sub(median, axis=0)
            .div(iqr, axis=0)
        )

        return normalized

    elif axis == "sample":

        median = df.median(axis=0)
        q1 = df.quantile(0.25, axis=0)
        q3 = df.quantile(0.75, axis=0)

        iqr = (q3 - q1).replace(0, np.nan)

        normalized = (
            df
            .sub(median, axis=1)
            .div(iqr, axis=1)
        )

        return normalized

    elif axis == "batch":

        if batch_map is None:
            raise ValueError(
                "batch_map (sample -> batch label) is required "
                "for batch-specific normalization."
            )

        out = df.copy()

        for batch in batch_map.dropna().unique():

            cols = batch_map.index[
                batch_map == batch
            ].tolist()

            cols = [
                c for c in cols
                if c in df.columns
            ]

            if not cols:
                continue

            sub = df[cols]

            median = sub.median(axis=1)
            q1 = sub.quantile(0.25, axis=1)
            q3 = sub.quantile(0.75, axis=1)

            iqr = (q3 - q1).replace(0, np.nan)

            out[cols] = (
                sub
                .sub(median, axis=0)
                .div(iqr, axis=0)
            )

        return out

    else:

        raise ValueError(
            "axis must be one of 'feature', 'sample', 'batch'"
        )


# ---------------------------------------------------------------------------
# 4. Distribution plots
# ---------------------------------------------------------------------------
def distribution_plots(
    before: pd.DataFrame,
    after: pd.DataFrame,
    sample_id: str = None
):
    """
    Generate before/after distribution and box plots.

    'before' should normally be the ISTD-normalized peak area.

    'after' should normally be the log2-transformed data.
    """

    b = before.values.flatten()
    b = b[np.isfinite(b) & (b > 0)]

    a = after.values.flatten()
    a = a[np.isfinite(a)]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8)
    )

    # -----------------------------------------------------------------------
    # Before: raw/ISTD-normalized distribution
    # -----------------------------------------------------------------------
    if b.size:
        axes[0, 0].hist(
            b,
            bins=60,
            alpha=0.8
        )
        axes[0, 0].set_xscale("log")

    axes[0, 0].set_title(
        "Before: ISTD-normalized Peak Area"
    )
    axes[0, 0].set_ylabel("Count")

    # -----------------------------------------------------------------------
    # Before: box plot
    # -----------------------------------------------------------------------
    if b.size:
        try:
            axes[0, 1].boxplot(
                [b],
                tick_labels=["Before"]
            )
        except TypeError:
            axes[0, 1].boxplot(
                [b],
                labels=["Before"]
            )

        axes[0, 1].set_yscale("log")

    axes[0, 1].set_title(
        "Before: Box Plot"
    )

    # -----------------------------------------------------------------------
    # After: log2 distribution
    # -----------------------------------------------------------------------
    if a.size:
        axes[1, 0].hist(
            a,
            bins=60,
            alpha=0.8
        )

    axes[1, 0].set_title(
        "After: log2(x)"
    )
    axes[1, 0].set_xlabel(
        "log2(ISTD-normalized Peak Area)"
    )
    axes[1, 0].set_ylabel("Count")

    # -----------------------------------------------------------------------
    # After: box plot
    # -----------------------------------------------------------------------
    if a.size:
        try:
            axes[1, 1].boxplot(
                [a],
                tick_labels=["After"]
            )
        except TypeError:
            axes[1, 1].boxplot(
                [a],
                labels=["After"]
            )

    axes[1, 1].set_title(
        "After: log2(x) Box Plot"
    )

    fig.tight_layout()

    return fig
