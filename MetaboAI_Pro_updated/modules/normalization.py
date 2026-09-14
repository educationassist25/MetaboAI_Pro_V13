"""
normalization.py
------------------------------------------------------------
ISTD normalization, direct log2 transformation,
and Median-IQR normalization.

Recommended workflow:

    Raw Peak Area
          ↓
    ISTD normalization
          ↓
    log2(x)
          ↓
    IQR normalization
          ↓
    Statistical analysis

IMPORTANT:
    log2 transformation is strictly:

        log2(x)

    No pseudocount
    No constant
    No shifting
------------------------------------------------------------
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =====================================================================
# 1. ISTD NORMALIZATION
# =====================================================================

def istd_normalize(
    peak_df: pd.DataFrame,
    istd_name: str
) -> pd.DataFrame:
    """
    Normalize each endogenous metabolite by its internal standard (ISTD).

    Formula:

        Normalized Peak Area =
            Endogenous Peak Area / ISTD Peak Area

    Parameters
    ----------
    peak_df : pandas.DataFrame
        DataFrame with metabolites/features as rows
        and samples as columns.

    istd_name : str
        Name of the ISTD row.

    Returns
    -------
    pandas.DataFrame
        ISTD-normalized dataframe.

    Notes
    -----
    The ISTD itself is removed from the output.
    """

    if not isinstance(peak_df, pd.DataFrame):
        raise TypeError(
            "peak_df must be a pandas DataFrame."
        )

    if istd_name not in peak_df.index:
        raise ValueError(
            f"Internal standard '{istd_name}' "
            f"was not found among the dataframe rows."
        )

    # Get ISTD peak area
    istd_row = peak_df.loc[istd_name].copy()

    # Convert to numeric
    istd_row = pd.to_numeric(
        istd_row,
        errors="coerce"
    )

    # Zero ISTD values cannot be used for division
    istd_row = istd_row.replace(
        0,
        np.nan
    )

    # Remove ISTD from endogenous metabolite dataframe
    endogenous = peak_df.drop(
        index=istd_name
    ).copy()

    # Make sure all values are numeric
    endogenous = endogenous.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # ISTD normalization
    normalized = endogenous.div(
        istd_row,
        axis=1
    )

    return normalized


# =====================================================================
# 2. DIRECT LOG2 TRANSFORMATION
# =====================================================================

def log2_transform(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Perform a direct log2 transformation.

    Formula:

        X_log2 = log2(X)

    IMPORTANT
    ---------
    This function does NOT:

        - add a constant
        - add a pseudocount
        - replace zeros
        - shift negative values

    Values <= 0 are converted to NaN because:

        log2(0)       = undefined
        log2(negative) = undefined

    Positive values are transformed directly.

    Parameters
    ----------
    df : pandas.DataFrame
        Numeric dataframe.

    Returns
    -------
    pandas.DataFrame
        Direct log2-transformed dataframe.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    # Make a copy so the original dataframe is not changed
    work = df.copy()

    # Convert all values to numeric
    work = work.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Values <= 0 cannot be log2 transformed
    work = work.mask(
        work <= 0
    )

    # DIRECT log2 transformation
    transformed = np.log2(
        work
    )

    return transformed


# =====================================================================
# 3. IQR NORMALIZATION
# =====================================================================

def iqr_normalize(
    df: pd.DataFrame,
    axis: str = "feature",
    batch_map: pd.Series = None
) -> pd.DataFrame:
    """
    Median-IQR normalization.

    Formula:

        X_normalized =
            (X - Median) / IQR

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    axis : str
        'feature'
            Normalize each metabolite across samples.

        'sample'
            Normalize each sample across metabolites.

        'batch'
            Normalize each metabolite separately within each batch.

    batch_map : pandas.Series
        Required when axis='batch'.

        Index  = sample names
        Values = batch labels

    Returns
    -------
    pandas.DataFrame
        IQR-normalized dataframe.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    # ---------------------------------------------------------------
    # Feature-wise IQR normalization
    # ---------------------------------------------------------------

    if axis == "feature":

        median = df.median(
            axis=1,
            skipna=True
        )

        q1 = df.quantile(
            0.25,
            axis=1
        )

        q3 = df.quantile(
            0.75,
            axis=1
        )

        iqr = q3 - q1

        # Avoid division by zero
        iqr = iqr.replace(
            0,
            np.nan
        )

        normalized = (
            df
            .sub(
                median,
                axis=0
            )
            .div(
                iqr,
                axis=0
            )
        )

        return normalized

    # ---------------------------------------------------------------
    # Sample-wise IQR normalization
    # ---------------------------------------------------------------

    elif axis == "sample":

        median = df.median(
            axis=0,
            skipna=True
        )

        q1 = df.quantile(
            0.25,
            axis=0
        )

        q3 = df.quantile(
            0.75,
            axis=0
        )

        iqr = q3 - q1

        iqr = iqr.replace(
            0,
            np.nan
        )

        normalized = (
            df
            .sub(
                median,
                axis=1
            )
            .div(
                iqr,
                axis=1
            )
        )

        return normalized

    # ---------------------------------------------------------------
    # Batch-specific IQR normalization
    # ---------------------------------------------------------------

    elif axis == "batch":

        if batch_map is None:
            raise ValueError(
                "batch_map is required when "
                "axis='batch'."
            )

        if not isinstance(
            batch_map,
            pd.Series
        ):
            raise TypeError(
                "batch_map must be a pandas Series."
            )

        output = df.copy()

        for batch in batch_map.dropna().unique():

            # Samples belonging to this batch
            cols = batch_map.index[
                batch_map == batch
            ].tolist()

            # Keep only columns actually present
            cols = [
                col
                for col in cols
                if col in df.columns
            ]

            if not cols:
                continue

            sub = df[
                cols
            ].copy()

            median = sub.median(
                axis=1,
                skipna=True
            )

            q1 = sub.quantile(
                0.25,
                axis=1
            )

            q3 = sub.quantile(
                0.75,
                axis=1
            )

            iqr = q3 - q1

            iqr = iqr.replace(
                0,
                np.nan
            )

            output[
                cols
            ] = (
                sub
                .sub(
                    median,
                    axis=0
                )
                .div(
                    iqr,
                    axis=0
                )
            )

        return output

    else:

        raise ValueError(
            "axis must be one of: "
            "'feature', 'sample', 'batch'."
        )


# =====================================================================
# 4. COMPLETE NORMALIZATION PIPELINE
# =====================================================================

def normalize_metabolomics(
    peak_df: pd.DataFrame,
    istd_name: str,
    perform_iqr: bool = True,
    iqr_axis: str = "feature"
):
    """
    Complete metabolomics normalization workflow.

    Workflow:

        Raw Peak Area
             ↓
        ISTD normalization
             ↓
        Direct log2(x)
             ↓
        IQR normalization

    Parameters
    ----------
    peak_df : pandas.DataFrame
        Raw peak-area dataframe.

    istd_name : str
        Internal standard row name.

    perform_iqr : bool
        If True, perform IQR normalization after log2.

    iqr_axis : str
        IQR normalization axis.

    Returns
    -------
    dict
        Contains:

        'istd_normalized'
        'log2'
        'iqr_normalized'
    """

    # ---------------------------------------------------------------
    # Step 1: ISTD normalization
    # ---------------------------------------------------------------

    istd_df = istd_normalize(
        peak_df,
        istd_name
    )

    # ---------------------------------------------------------------
    # Step 2: DIRECT log2(x)
    # ---------------------------------------------------------------

    log2_df = log2_transform(
        istd_df
    )

    # ---------------------------------------------------------------
    # Step 3: IQR normalization
    # ---------------------------------------------------------------

    if perform_iqr:

        iqr_df = iqr_normalize(
            log2_df,
            axis=iqr_axis
        )

    else:

        iqr_df = None

    return {
        "istd_normalized": istd_df,
        "log2": log2_df,
        "iqr_normalized": iqr_df
    }


# =====================================================================
# 5. DISTRIBUTION PLOTS
# =====================================================================

def distribution_plots(
    before: pd.DataFrame,
    after: pd.DataFrame,
    sample_id: str = None
):
    """
    Generate before/after distribution plots.

    BEFORE:
        ISTD-normalized peak area

    AFTER:
        Direct log2-transformed data
    """

    # Flatten before data
    b = before.to_numpy(
        dtype=float
    ).flatten()

    # Keep positive finite values
    b = b[
        np.isfinite(b) &
        (b > 0)
    ]

    # Flatten after data
    a = after.to_numpy(
        dtype=float
    ).flatten()

    # Keep finite values
    a = a[
        np.isfinite(a)
    ]

    # Create figure
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8)
    )

    # ===============================================================
    # BEFORE: Distribution
    # ===============================================================

    if b.size > 0:

        axes[0, 0].hist(
            b,
            bins=60,
            alpha=0.8
        )

        axes[0, 0].set_xscale(
            "log"
        )

    axes[0, 0].set_title(
        "Before: ISTD-normalized Peak Area"
    )

    axes[0, 0].set_ylabel(
        "Count"
    )

    # ===============================================================
    # BEFORE: Boxplot
    # ===============================================================

    if b.size > 0:

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

        axes[0, 1].set_yscale(
            "log"
        )

    axes[0, 1].set_title(
        "Before: Box Plot"
    )

    # ===============================================================
    # AFTER: log2 distribution
    # ===============================================================

    if a.size > 0:

        axes[1, 0].hist(
            a,
            bins=60,
            alpha=0.8
        )

    axes[1, 0].set_title(
        "After: Direct log2(x)"
    )

    axes[1, 0].set_xlabel(
        "log2(ISTD-normalized Peak Area)"
    )

    axes[1, 0].set_ylabel(
        "Count"
    )

    # ===============================================================
    # AFTER: Boxplot
    # ===============================================================

    if a.size > 0:

        try:

            axes[1, 1].boxplot(
                [a],
                tick_labels=["log2(x)"]
            )

        except TypeError:

            axes[1, 1].boxplot(
                [a],
                labels=["log2(x)"]
            )

    axes[1, 1].set_title(
        "After: Direct log2(x)"
    )

    # ===============================================================
    # Optional figure title
    # ===============================================================

    if sample_id is not None:

        fig.suptitle(
            str(sample_id),
            fontsize=14
        )

    fig.tight_layout()

    return fig


# =====================================================================
# 6. VALIDATION FUNCTION
# =====================================================================

def validate_log2(
    df: pd.DataFrame,
    transformed_df: pd.DataFrame
):
    """
    Validate that transformed values are exactly log2(x).

    This is useful for checking the normalization pipeline.
    """

    original = df.copy()

    original = original.apply(
        pd.to_numeric,
        errors="coerce"
    )

    expected = original.mask(
        original <= 0
    )

    expected = np.log2(
        expected
    )

    difference = (
        transformed_df - expected
    )

    max_difference = np.nanmax(
        np.abs(
            difference.to_numpy()
        )
    )

    print(
        "Maximum absolute difference:",
        max_difference
    )

    if max_difference < 1e-10:

        print(
            "✓ Validation passed: "
            "transformed values are log2(x)."
        )

    else:

        print(
            "✗ Validation failed: "
            "transformed values differ from log2(x)."
        )

    return difference
