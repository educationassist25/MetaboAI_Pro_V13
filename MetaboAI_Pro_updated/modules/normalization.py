"""
normalization.py - ISTD normalization, IQR normalization, and strict Log2 transformation.

Log2 transformation is strictly:

    log2(x)

No pseudocount, constant addition, zero replacement, shifting, or arbitrary offset
is applied. Values <= 0 cannot be log2 transformed and are converted to NaN.
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
    Internal Standard (ISTD) normalization.

    Normalized Peak Area =
        Endogenous Metabolite Peak Area / Internal Standard Peak Area

    The ISTD feature itself is removed from the returned dataframe.

    If an ISTD value is zero, division is undefined and the corresponding
    normalized values become NaN.
    """
    if istd_name not in peak_df.index:
        raise ValueError(
            f"Internal standard '{istd_name}' not found among features."
        )

    istd_row = peak_df.loc[istd_name]

    # Avoid division by zero.
    denominator = istd_row.replace(0, np.nan)

    normalized = (
        peak_df
        .drop(index=istd_name)
        .div(denominator, axis=1)
    )

    return normalized


# ---------------------------------------------------------------------------
# 2. IQR normalization
# ---------------------------------------------------------------------------
def iqr_normalize(
    df: pd.DataFrame,
    axis: str = "feature",
    batch_map: pd.Series = None
) -> pd.DataFrame:
    """
    Median-IQR normalization (robust scaling):

        X_norm = (X - Median(X)) / IQR(X)

    Parameters
    ----------
    df : pd.DataFrame
        Numeric feature x sample dataframe.

    axis : str
        'feature':
            Normalize each feature across samples.

        'sample':
            Normalize each sample across features.

        'batch':
            Normalize each feature independently within each batch.
            Requires batch_map.

    batch_map : pd.Series, optional
        Series indexed by sample name containing batch labels.

    Returns
    -------
    pd.DataFrame
        IQR-normalized dataframe.

    Notes
    -----
    This is a centering/scaling transformation and can produce negative
    values. Therefore, the result should NOT be passed directly to the
    strict log2 transformation unless all values are positive.
    """

    if axis == "feature":

        median = df.median(axis=1)
        q1 = df.quantile(0.25, axis=1)
        q3 = df.quantile(0.75, axis=1)

        iqr = (q3 - q1).replace(0, np.nan)

        return (
            df
            .sub(median, axis=0)
            .div(iqr, axis=0)
        )

    elif axis == "sample":

        median = df.median(axis=0)
        q1 = df.quantile(0.25, axis=0)
        q3 = df.quantile(0.75, axis=0)

        iqr = (q3 - q1).replace(0, np.nan)

        return (
            df
            .sub(median, axis=1)
            .div(iqr, axis=1)
        )

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

            # Keep only columns that actually exist in the dataframe.
            cols = [c for c in cols if c in df.columns]

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
            "axis must be one of 'feature', 'sample', or 'batch'"
        )


# ---------------------------------------------------------------------------
# 3. STRICT Log2 transformation
# ---------------------------------------------------------------------------
def log2_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strict Log2 transformation.

        transformed = log2(x)

    IMPORTANT
    ---------
    No pseudocount is added.

    No constant is added.

    No zero replacement is performed.

    No shifting is performed.

    No arbitrary offset is applied.

    Values <= 0 cannot be log2 transformed and are converted to NaN.

    NaN values remain NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric dataframe.

    Returns
    -------
    pd.DataFrame
        Strict log2-transformed dataframe.
    """

    work = df.copy()

    # Convert values <= 0 to NaN.
    work = work.where(work > 0, np.nan)

    # Strict mathematical log2(x).
    transformed = np.log2(work)

    return transformed


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

    Before:
        Raw peak-area values are displayed using a logarithmic x/y scale
        where appropriate.

    After:
        Strict log2-transformed values are displayed on a linear scale.

    Values that are <= 0 before transformation are excluded from the
    transformed distribution because they become NaN under strict log2.
    """

    # ------------------------------------------------------------------
    # Before data
    # ------------------------------------------------------------------
    b = before.values.flatten()

    b = b[
        np.isfinite(b) &
        (b > 0)
    ]

    # ------------------------------------------------------------------
    # After data
    # ------------------------------------------------------------------
    a = after.values.flatten()

    a = a[
        np.isfinite(a)
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8)
    )

    # ------------------------------------------------------------------
    # Before: Distribution
    # ------------------------------------------------------------------
    if b.size:
        axes[0, 0].hist(
            b,
            bins=60,
            alpha=0.8
        )

        axes[0, 0].set_xscale("log")

    axes[0, 0].set_title(
        "Before: Raw Peak Area Distribution"
    )

    axes[0, 0].set_xlabel(
        "Peak Area (log scale)"
    )

    axes[0, 0].set_ylabel(
        "Count"
    )

    # ------------------------------------------------------------------
    # Before: Box plot
    # ------------------------------------------------------------------
    if b.size:

        try:
            axes[0, 1].boxplot(
                [b],
                tick_labels=["Before"]
            )
        except TypeError:
            # Compatibility with older matplotlib versions.
            axes[0, 1].boxplot(
                [b],
                labels=["Before"]
            )

        axes[0, 1].set_yscale("log")

    axes[0, 1].set_title(
        "Before: Box Plot"
    )

    axes[0, 1].set_ylabel(
        "Peak Area (log scale)"
    )

    # ------------------------------------------------------------------
    # After: Strict Log2 distribution
    # ------------------------------------------------------------------
    if a.size:

        axes[1, 0].hist(
            a,
            bins=60,
            alpha=0.8
        )

    axes[1, 0].set_title(
        "After: Strict Log2 Distribution"
    )

    axes[1, 0].set_xlabel(
        "log2(Peak Area)"
    )

    axes[1, 0].set_ylabel(
        "Count"
    )

    # ------------------------------------------------------------------
    # After: Box plot
    # ------------------------------------------------------------------
    if a.size:

        try:
            axes[1, 1].boxplot(
                [a],
                tick_labels=["After"]
            )
        except TypeError:
            # Compatibility with older matplotlib versions.
            axes[1, 1].boxplot(
                [a],
                labels=["After"]
            )

    axes[1, 1].set_title(
        "After: Strict Log2 Box Plot"
    )

    axes[1, 1].set_ylabel(
        "log2(Peak Area)"
    )

    # ------------------------------------------------------------------
    # Optional sample identifier
    # ------------------------------------------------------------------
    if sample_id:
        fig.suptitle(
            f"Normalization / Log2 Transformation: {sample_id}",
            fontsize=12
        )

    fig.tight_layout()

    return fig
