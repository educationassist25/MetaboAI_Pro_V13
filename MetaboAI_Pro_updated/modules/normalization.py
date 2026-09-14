"""
normalization.py
------------------------------------------------------------
Metabolomics normalization utilities

Workflow:

    Raw Peak Area
          ↓
    ISTD normalization
          ↓
    Direct log2 transformation
          ↓
    Median-IQR normalization
          ↓
    Statistical analysis

IMPORTANT
---------
The log2 transformation is strictly:

    log2(x)

No:
    - pseudocount
    - constant addition
    - zero replacement
    - shifting
    - arbitrary offset

Values <= 0 cannot be log2 transformed and are therefore
converted to NaN.

ISTD normalization:

    Endogenous Peak Area / ISTD Peak Area

Median-IQR normalization:

    (X - Median) / IQR

------------------------------------------------------------
"""

import numpy as np
import pandas as pd

import matplotlib

# Prevent GUI/backend problems in Streamlit/server environments
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
    Normalize each endogenous metabolite by its internal standard.

    Formula
    -------
    Normalized Peak Area =
        Endogenous Peak Area / ISTD Peak Area

    Parameters
    ----------
    peak_df : pandas.DataFrame
        DataFrame with metabolites/features as rows
        and samples as columns.

    istd_name : str
        Exact name of the ISTD row.

    Returns
    -------
    pandas.DataFrame
        ISTD-normalized dataframe.

    Notes
    -----
    The ISTD itself is removed from the output.

    ISTD values equal to zero are converted to NaN because
    division by zero is not valid.
    """

    # ---------------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------------

    if not isinstance(peak_df, pd.DataFrame):
        raise TypeError(
            "peak_df must be a pandas DataFrame."
        )

    if istd_name not in peak_df.index:
        raise ValueError(
            f"Internal standard '{istd_name}' "
            f"was not found among the dataframe rows."
        )

    # ---------------------------------------------------------------
    # Extract ISTD row
    # ---------------------------------------------------------------

    istd_row = peak_df.loc[istd_name].copy()

    # Convert ISTD values to numeric
    istd_row = pd.to_numeric(
        istd_row,
        errors="coerce"
    )

    # Zero ISTD values cannot be used as denominators
    istd_row = istd_row.replace(
        0,
        np.nan
    )

    # ---------------------------------------------------------------
    # Remove ISTD from endogenous metabolites
    # ---------------------------------------------------------------

    endogenous = peak_df.drop(
        index=istd_name
    ).copy()

    # Convert endogenous values to numeric
    endogenous = endogenous.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # ---------------------------------------------------------------
    # ISTD normalization
    # ---------------------------------------------------------------

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

    Formula
    -------
        X_log2 = log2(X)

    IMPORTANT
    ---------
    This function performs ONLY:

        log2(x)

    It does NOT:

        - add a constant
        - add a pseudocount
        - replace zero with a small number
        - shift the data
        - subtract the minimum
        - perform any additional normalization

    Values <= 0 are converted to NaN because:

        log2(0)        = undefined
        log2(negative) = undefined

    Parameters
    ----------
    df : pandas.DataFrame
        Numeric dataframe.

    Returns
    -------
    pandas.DataFrame
        Direct log2-transformed dataframe.

    IMPORTANT FOR APP.PY
    --------------------
    This function returns ONE DataFrame.

    Correct:

        qc_log = normalization.log2_transform(
            peak_df[qc_cols]
        )

    Incorrect:

        qc_log, _ = normalization.log2_transform(
            peak_df[qc_cols]
        )
    """

    # ---------------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------------

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    # ---------------------------------------------------------------
    # Make a copy
    # ---------------------------------------------------------------

    work = df.copy()

    # ---------------------------------------------------------------
    # Convert all values to numeric
    # ---------------------------------------------------------------

    work = work.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # ---------------------------------------------------------------
    # Values <= 0 cannot be log2 transformed
    #
    # They become NaN.
    # ---------------------------------------------------------------

    work = work.mask(
        work <= 0
    )

    # ---------------------------------------------------------------
    # DIRECT log2 transformation
    # ---------------------------------------------------------------

    transformed = np.log2(
        work
    )

    # ---------------------------------------------------------------
    # Preserve original index and columns
    # ---------------------------------------------------------------

    transformed.index = df.index
    transformed.columns = df.columns

    return transformed


# =====================================================================
# 3. MEDIAN-IQR NORMALIZATION
# =====================================================================

def iqr_normalize(
    df: pd.DataFrame,
    axis: str = "feature",
    batch_map: pd.Series = None
) -> pd.DataFrame:
    """
    Median-IQR normalization.

    Formula
    -------
        X_normalized =
            (X - Median) / IQR

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    axis : str
        Available options:

        'feature'
            Normalize each metabolite across samples.

        'sample'
            Normalize each sample across metabolites.

        'batch'
            Normalize each metabolite separately within
            each batch.

    batch_map : pandas.Series, optional
        Required when axis='batch'.

        Index:
            sample names

        Values:
            batch labels

    Returns
    -------
    pandas.DataFrame
        Median-IQR normalized dataframe.

    Notes
    -----
    If IQR = 0, the normalized values are set to NaN for
    that feature/sample because division by zero is undefined.
    """

    # ---------------------------------------------------------------
    # Validate dataframe
    # ---------------------------------------------------------------

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    # Work on numeric data
    work = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # ===============================================================
    # FEATURE-WISE NORMALIZATION
    # ===============================================================

    if axis == "feature":

        # Median across samples for each metabolite
        median = work.median(
            axis=1,
            skipna=True
        )

        # First quartile
        q1 = work.quantile(
            0.25,
            axis=1
        )

        # Third quartile
        q3 = work.quantile(
            0.75,
            axis=1
        )

        # IQR
        iqr = q3 - q1

        # Avoid division by zero
        iqr = iqr.replace(
            0,
            np.nan
        )

        # Median-IQR normalization
        normalized = (
            work
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

    # ===============================================================
    # SAMPLE-WISE NORMALIZATION
    # ===============================================================

    elif axis == "sample":

        # Median across metabolites for each sample
        median = work.median(
            axis=0,
            skipna=True
        )

        # First quartile
        q1 = work.quantile(
            0.25,
            axis=0
        )

        # Third quartile
        q3 = work.quantile(
            0.75,
            axis=0
        )

        # IQR
        iqr = q3 - q1

        # Avoid division by zero
        iqr = iqr.replace(
            0,
            np.nan
        )

        # Median-IQR normalization
        normalized = (
            work
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

    # ===============================================================
    # BATCH-SPECIFIC NORMALIZATION
    # ===============================================================

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

        # Make sure batch_map contains sample names
        missing_samples = [
            sample
            for sample in batch_map.index
            if sample not in work.columns
        ]

        # Copy original dataframe
        output = work.copy()

        # -----------------------------------------------------------
        # Process each batch independently
        # -----------------------------------------------------------

        for batch in batch_map.dropna().unique():

            # Samples belonging to this batch
            cols = batch_map.index[
                batch_map == batch
            ].tolist()

            # Keep only columns present in dataframe
            cols = [
                col
                for col in cols
                if col in work.columns
            ]

            # Skip empty batches
            if not cols:
                continue

            # Data for current batch
            sub = work[
                cols
            ].copy()

            # Median for each feature
            median = sub.median(
                axis=1,
                skipna=True
            )

            # Q1
            q1 = sub.quantile(
                0.25,
                axis=1
            )

            # Q3
            q3 = sub.quantile(
                0.75,
                axis=1
            )

            # IQR
            iqr = q3 - q1

            # Avoid division by zero
            iqr = iqr.replace(
                0,
                np.nan
            )

            # Normalize batch
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

    # ===============================================================
    # INVALID AXIS
    # ===============================================================

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

    Workflow
    --------
        Raw Peak Area
             ↓
        ISTD normalization
             ↓
        Direct log2(x)
             ↓
        Median-IQR normalization

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

        Options:

            'feature'
            'sample'
            'batch'

    Returns
    -------
    dict

        {
            "istd_normalized": DataFrame,
            "log2": DataFrame,
            "iqr_normalized": DataFrame or None
        }

    Example
    -------
        result = normalize_metabolomics(
            peak_df,
            istd_name="ISTD"
        )

        istd = result["istd_normalized"]
        log2_data = result["log2"]
        iqr_data = result["iqr_normalized"]
    """

    # ===============================================================
    # STEP 1
    # ISTD NORMALIZATION
    # ===============================================================

    istd_df = istd_normalize(
        peak_df,
        istd_name
    )

    # ===============================================================
    # STEP 2
    # DIRECT LOG2 TRANSFORMATION
    # ===============================================================

    log2_df = log2_transform(
        istd_df
    )

    # ===============================================================
    # STEP 3
    # MEDIAN-IQR NORMALIZATION
    # ===============================================================

    if perform_iqr:

        iqr_df = iqr_normalize(
            log2_df,
            axis=iqr_axis
        )

    else:

        iqr_df = None

    # ===============================================================
    # RETURN RESULTS
    # ===============================================================

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

    BEFORE
    ------
        ISTD-normalized peak area

    AFTER
    -----
        Direct log2-transformed data

    Parameters
    ----------
    before : pandas.DataFrame
        ISTD-normalized data.

    after : pandas.DataFrame
        log2-transformed data.

    sample_id : str, optional
        Optional figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    # ===============================================================
    # BEFORE DATA
    # ===============================================================

    b = before.to_numpy(
        dtype=float
    ).flatten()

    # Keep finite positive values
    b = b[
        np.isfinite(b) &
        (b > 0)
    ]

    # ===============================================================
    # AFTER DATA
    # ===============================================================

    a = after.to_numpy(
        dtype=float
    ).flatten()

    # Keep finite values
    a = a[
        np.isfinite(a)
    ]

    # ===============================================================
    # CREATE FIGURE
    # ===============================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8)
    )

    # ===============================================================
    # BEFORE: DISTRIBUTION
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

    axes[0, 0].set_xlabel(
        "ISTD-normalized Peak Area"
    )

    axes[0, 0].set_ylabel(
        "Count"
    )

    # ===============================================================
    # BEFORE: BOXPLOT
    # ===============================================================

    if b.size > 0:

        try:

            axes[0, 1].boxplot(
                [b],
                tick_labels=["Before"]
            )

        except TypeError:

            # Compatibility with older matplotlib
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
    # AFTER: LOG2 DISTRIBUTION
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
    # AFTER: BOXPLOT
    # ===============================================================

    if a.size > 0:

        try:

            axes[1, 1].boxplot(
                [a],
                tick_labels=["log2(x)"]
            )

        except TypeError:

            # Compatibility with older matplotlib
            axes[1, 1].boxplot(
                [a],
                labels=["log2(x)"]
            )

    axes[1, 1].set_title(
        "After: Direct log2(x)"
    )

    # ===============================================================
    # OPTIONAL FIGURE TITLE
    # ===============================================================

    if sample_id is not None:

        fig.suptitle(
            str(sample_id),
            fontsize=14
        )

    # ===============================================================
    # LAYOUT
    # ===============================================================

    fig.tight_layout()

    return fig


# =====================================================================
# 6. VALIDATE LOG2 TRANSFORMATION
# =====================================================================

def validate_log2(
    df: pd.DataFrame,
    transformed_df: pd.DataFrame
):
    """
    Validate that transformed values are exactly log2(x).

    Parameters
    ----------
    df : pandas.DataFrame
        Original input dataframe.

    transformed_df : pandas.DataFrame
        Output from log2_transform().

    Returns
    -------
    pandas.DataFrame
        Difference between calculated and expected values.

    Notes
    -----
    The expected transformation is strictly:

        log2(x)

    No pseudocount or constant is used.
    """

    # ===============================================================
    # ORIGINAL DATA
    # ===============================================================

    original = df.copy()

    original = original.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # ===============================================================
    # EXPECTED LOG2
    # ===============================================================

    expected = original.mask(
        original <= 0
    )

    expected = np.log2(
        expected
    )

    # ===============================================================
    # DIFFERENCE
    # ===============================================================

    difference = (
        transformed_df - expected
    )

    # ===============================================================
    # MAXIMUM ABSOLUTE DIFFERENCE
    # ===============================================================

    difference_values = difference.to_numpy(
        dtype=float
    )

    if np.isfinite(
        difference_values
    ).any():

        max_difference = np.nanmax(
            np.abs(
                difference_values
            )
        )

    else:

        max_difference = np.nan

    print(
        "Maximum absolute difference:",
        max_difference
    )

    # ===============================================================
    # VALIDATION RESULT
    # ===============================================================

    if np.isnan(max_difference):

        print(
            "⚠ Validation could not be completed: "
            "no finite transformed values were found."
        )

    elif max_difference < 1e-10:

        print(
            "✓ Validation passed: "
            "transformed values are exactly log2(x)."
        )

    else:

        print(
            "✗ Validation failed: "
            "transformed values differ from log2(x)."
        )

    return difference


# =====================================================================
# 7. SIMPLE PIPELINE TEST
# =====================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Testing normalization.py")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Example raw peak-area data
    # ---------------------------------------------------------------

    test_data = pd.DataFrame(
        {
            "Sample_1": [
                10000,
                2000,
                5000,
                100
            ],
            "Sample_2": [
                12000,
                2500,
                5500,
                120
            ],
            "Sample_3": [
                9000,
                1800,
                4500,
                90
            ],
        },
        index=[
            "Metabolite_A",
            "Metabolite_B",
            "Metabolite_C",
            "ISTD"
        ]
    )

    print("\nRaw Peak Area:")
    print(test_data)

    # ---------------------------------------------------------------
    # ISTD normalization
    # ---------------------------------------------------------------

    istd = istd_normalize(
        test_data,
        "ISTD"
    )

    print("\nISTD-normalized:")
    print(istd)

    # ---------------------------------------------------------------
    # Direct log2
    # ---------------------------------------------------------------

    log2_data = log2_transform(
        istd
    )

    print("\nDirect log2:")
    print(log2_data)

    # ---------------------------------------------------------------
    # Validate log2
    # ---------------------------------------------------------------

    print("\nLog2 validation:")

    validate_log2(
        istd,
        log2_data
    )

    # ---------------------------------------------------------------
    # IQR normalization
    # ---------------------------------------------------------------

    iqr = iqr_normalize(
        log2_data,
        axis="feature"
    )

    print("\nMedian-IQR normalized:")
    print(iqr)

    # ---------------------------------------------------------------
    # Complete pipeline
    # ---------------------------------------------------------------

    result = normalize_metabolomics(
        test_data,
        istd_name="ISTD",
        perform_iqr=True,
        iqr_axis="feature"
    )

    print("\nComplete pipeline:")
    print(
        result["istd_normalized"]
    )

    print("\nLog2:")
    print(
        result["log2"]
    )

    print("\nIQR normalized:")
    print(
        result["iqr_normalized"]
    )

    print("\n" + "=" * 70)
    print("Test completed successfully.")
    print("=" * 70)
