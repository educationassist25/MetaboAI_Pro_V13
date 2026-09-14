```python
"""
stats_analysis.py
============================================================

MetaboAI Pro
Automated LC-MS Metabolomics Statistical Analysis

Core statistical analysis module.

SUPPORTED ANALYSES
------------------
1. Two-group Welch t-test
2. Two-group Mann-Whitney U test
3. One-way ANOVA
4. Tukey HSD post-hoc analysis
5. Pairwise Welch post-hoc analysis
6. Control-vs-group Welch comparisons
7. Benjamini-Hochberg FDR correction
8. Welch confidence intervals
9. Log2 fold-change
10. Linear fold-change

IMPORTANT DATA WORKFLOW
-----------------------

The input data must already be processed upstream:

    Raw Peak Area
          |
          v
    ISTD normalization
          |
          v
    log2 transformation
          |
          v
    IQR normalization
          |
          v
    Statistical analysis

THIS MODULE DOES NOT PERFORM LOG2 TRANSFORMATION.

Fold-change convention
----------------------

    Group A = reference/control
    Group B = treatment/experimental

    Log2FC = Mean(Group B) - Mean(Group A)

    Linear FC = 2 ** Log2FC

Therefore:

    Log2FC > 0
        Group B is higher

    Log2FC < 0
        Group B is lower

Default significance:

    p < 0.05
    FDR < 0.25

============================================================
"""

import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations


# ============================================================
# OPTIONAL TUKEY SUPPORT
# ============================================================

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    HAS_TUKEY = True

except ImportError:
    HAS_TUKEY = False


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_PVALUE_THRESHOLD = 0.05
DEFAULT_FDR_THRESHOLD = 0.25
DEFAULT_CONFIDENCE = 0.95


# ============================================================
# 1. FDR CORRECTION
# ============================================================

def _fdr(pvals):
    """
    Benjamini-Hochberg FDR correction.

    NaN values remain NaN.
    """

    pvals = np.asarray(
        pvals,
        dtype=float
    )

    fdr = np.full(
        pvals.shape,
        np.nan,
        dtype=float
    )

    valid = np.isfinite(pvals)

    if not np.any(valid):
        return fdr

    fdr[valid] = multipletests(
        pvals[valid],
        method="fdr_bh"
    )[1]

    return fdr


# ============================================================
# 2. WELCH CONFIDENCE INTERVAL
# ============================================================

def _welch_ci(
    x,
    y,
    confidence=DEFAULT_CONFIDENCE
):
    """
    Welch confidence interval for:

        Mean(y) - Mean(x)

    This corresponds to:

        Group B - Group A
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    nx = len(x)
    ny = len(y)

    if nx < 2 or ny < 2:
        return np.nan, np.nan

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    difference = mean_y - mean_x

    var_x = np.var(
        x,
        ddof=1
    )

    var_y = np.var(
        y,
        ddof=1
    )

    variance_term = (
        var_x / nx
        +
        var_y / ny
    )

    if variance_term <= 0:
        return difference, difference

    se = np.sqrt(
        variance_term
    )

    denominator = (
        (var_x / nx) ** 2 / (nx - 1)
        +
        (var_y / ny) ** 2 / (ny - 1)
    )

    if denominator > 0:

        df = (
            variance_term ** 2
            /
            denominator
        )

    else:

        df = nx + ny - 2

    alpha = 1.0 - confidence

    t_critical = stats.t.ppf(
        1.0 - alpha / 2.0,
        df
    )

    margin = t_critical * se

    return (
        difference - margin,
        difference + margin
    )


# ============================================================
# 3. VALIDATE TWO GROUPS
# ============================================================

def _validate_two_groups(
    data_log2,
    group_a_samples,
    group_b_samples
):
    """
    Validate two-group inputs.
    """

    if not isinstance(
        data_log2,
        pd.DataFrame
    ):
        raise TypeError(
            "data_log2 must be a pandas DataFrame."
        )

    if data_log2.empty:
        raise ValueError(
            "data_log2 is empty."
        )

    group_a_samples = list(
        group_a_samples
    )

    group_b_samples = list(
        group_b_samples
    )

    if len(group_a_samples) == 0:
        raise ValueError(
            "Group A contains no samples."
        )

    if len(group_b_samples) == 0:
        raise ValueError(
            "Group B contains no samples."
        )

    missing_a = [
        sample
        for sample in group_a_samples
        if sample not in data_log2.columns
    ]

    missing_b = [
        sample
        for sample in group_b_samples
        if sample not in data_log2.columns
    ]

    if missing_a:
        raise ValueError(
            f"Group A samples not found in data: {missing_a}"
        )

    if missing_b:
        raise ValueError(
            f"Group B samples not found in data: {missing_b}"
        )

    if len(group_a_samples) < 2:
        raise ValueError(
            "Group A requires at least 2 samples."
        )

    if len(group_b_samples) < 2:
        raise ValueError(
            "Group B requires at least 2 samples."
        )

    overlap = set(
        group_a_samples
    ).intersection(
        group_b_samples
    )

    if overlap:
        raise ValueError(
            "A sample cannot belong to both groups: "
            f"{sorted(overlap)}"
        )

    return (
        group_a_samples,
        group_b_samples
    )


# ============================================================
# 4. TWO-GROUP TEST
# ============================================================

def two_group_test(
    data_log2: pd.DataFrame,
    group_a_samples,
    group_b_samples,
    method="ttest",
    pvalue_threshold=DEFAULT_PVALUE_THRESHOLD,
    fdr_threshold=DEFAULT_FDR_THRESHOLD,
    confidence=DEFAULT_CONFIDENCE
):
    """
    Perform a two-group statistical comparison.

    Group A = reference/control
    Group B = treatment/experimental

    Log2FC:

        Mean(Group B) - Mean(Group A)

    Linear FC:

        2 ** Log2FC
    """

    (
        group_a_samples,
        group_b_samples
    ) = _validate_two_groups(
        data_log2,
        group_a_samples,
        group_b_samples
    )

    method = str(
        method
    ).strip().lower()

    group_a = data_log2[
        group_a_samples
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    group_b = data_log2[
        group_b_samples
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    pvals = []
    ci_lower = []
    ci_upper = []

    # ========================================================
    # FEATURE-BY-FEATURE TESTING
    # ========================================================

    for feature in data_log2.index:

        x = (
            group_a
            .loc[feature]
            .dropna()
            .astype(float)
            .values
        )

        y = (
            group_b
            .loc[feature]
            .dropna()
            .astype(float)
            .values
        )

        if len(x) < 2 or len(y) < 2:

            pvals.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)

            continue

        # ----------------------------------------------------
        # Welch t-test
        # ----------------------------------------------------

        if method in (
            "ttest",
            "welch",
            "welch_ttest",
            "welch t-test",
            "t-test",
            "t_test"
        ):

            try:

                _, p = stats.ttest_ind(
                    x,
                    y,
                    equal_var=False,
                    nan_policy="omit"
                )

            except Exception:

                p = np.nan

        # ----------------------------------------------------
        # Mann-Whitney U
        # ----------------------------------------------------

        elif method in (
            "wilcoxon",
            "mannwhitney",
            "mann-whitney",
            "mann_whitney",
            "mannwhitneyu",
            "mw"
        ):

            try:

                _, p = stats.mannwhitneyu(
                    x,
                    y,
                    alternative="two-sided"
                )

            except Exception:

                p = np.nan

        else:

            raise ValueError(
                "Unsupported statistical method: "
                f"{method}. "
                "Use 'ttest'/'welch' or "
                "'wilcoxon'/'mannwhitney'."
            )

        pvals.append(p)

        lo, hi = _welch_ci(
            x,
            y,
            confidence=confidence
        )

        ci_lower.append(lo)
        ci_upper.append(hi)

    # ========================================================
    # GROUP MEANS
    # ========================================================

    mean_a = group_a.mean(
        axis=1
    )

    mean_b = group_b.mean(
        axis=1
    )

    # ========================================================
    # LOG2FC
    # ========================================================

    log2fc = (
        mean_b
        -
        mean_a
    )

    # ========================================================
    # LINEAR FC
    # ========================================================

    linear_fc = np.power(
        2.0,
        log2fc
    )

    # ========================================================
    # FDR
    # ========================================================

    fdr = _fdr(
        pvals
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = pd.DataFrame(
        {
            "Mean_Log2_GroupA": mean_a,
            "Mean_Log2_GroupB": mean_b,
            "Linear_FC": linear_fc,
            "Log2FC": log2fc,
            "CI_Lower_Log2FC": ci_lower,
            "CI_Upper_Log2FC": ci_upper,
            "p-value": pvals,
            "FDR": fdr
        },
        index=data_log2.index
    )

    # ========================================================
    # SIGNIFICANCE
    # ========================================================

    result["Significant"] = (
        result["p-value"].lt(
            pvalue_threshold
        )
        &
        result["FDR"].lt(
            fdr_threshold
        )
    )

    # ========================================================
    # DIRECTION
    # ========================================================

    result["Direction"] = np.select(
        [
            result["Log2FC"] > 0,
            result["Log2FC"] < 0
        ],
        [
            "Higher in Group B",
            "Lower in Group B"
        ],
        default="No change"
    )

    # ========================================================
    # SORT
    # ========================================================

    result = result.sort_values(
        by="p-value",
        na_position="last"
    )

    return result


# ============================================================
# 5. ONE-WAY ANOVA
# ============================================================

def anova_test(
    data_log2: pd.DataFrame,
    group_map: pd.Series,
    posthoc="tukey",
    fdr_threshold=DEFAULT_FDR_THRESHOLD
):
    """
    One-way ANOVA across three or more groups.
    """

    if not isinstance(
        data_log2,
        pd.DataFrame
    ):
        raise TypeError(
            "data_log2 must be a pandas DataFrame."
        )

    if data_log2.empty:
        raise ValueError(
            "data_log2 is empty."
        )

    if not isinstance(
        group_map,
        pd.Series
    ):
        group_map = pd.Series(
            group_map
        )

    missing_samples = [
        sample
        for sample in group_map.index
        if sample not in data_log2.columns
    ]

    if missing_samples:
        raise ValueError(
            "Samples in group_map not found in data: "
            f"{missing_samples}"
        )

    common_samples = [
        sample
        for sample in group_map.index
        if sample in data_log2.columns
    ]

    group_map = group_map.loc[
        common_samples
    ].dropna()

    groups = (
        group_map
        .unique()
        .tolist()
    )

    if len(groups) < 3:
        raise ValueError(
            "ANOVA requires at least 3 groups."
        )

    f_stats = []
    pvals = []

    # ========================================================
    # ANOVA FEATURE LOOP
    # ========================================================

    for feature in data_log2.index:

        samples_by_group = []

        for group in groups:

            samples = group_map.index[
                group_map == group
            ]

            values = (
                data_log2
                .loc[
                    feature,
                    samples
                ]
                .dropna()
                .astype(float)
                .values
            )

            if len(values) >= 2:
                samples_by_group.append(
                    values
                )

        if len(samples_by_group) < 3:

            f_stats.append(np.nan)
            pvals.append(np.nan)

            continue

        try:

            f_stat, p = stats.f_oneway(
                *samples_by_group
            )

        except Exception:

            f_stat = np.nan
            p = np.nan

        f_stats.append(f_stat)
        pvals.append(p)

    # ========================================================
    # FDR
    # ========================================================

    fdr = _fdr(
        pvals
    )

    anova_table = pd.DataFrame(
        {
            "F-statistic": f_stats,
            "ANOVA p-value": pvals,
            "FDR": fdr
        },
        index=data_log2.index
    )

    anova_table = anova_table.sort_values(
        by="ANOVA p-value",
        na_position="last"
    )

    significant_features = (
        anova_table
        .loc[
            anova_table["FDR"]
            <
            fdr_threshold
        ]
        .index
        .tolist()
    )

    posthoc_results = {}

    posthoc_method = str(
        posthoc
    ).strip().lower()

    # ========================================================
    # POST-HOC
    # ========================================================

    for feature in significant_features:

        feature_values = data_log2.loc[
            feature,
            group_map.index
        ]

        sub_df = pd.DataFrame(
            {
                "value": pd.to_numeric(
                    feature_values,
                    errors="coerce"
                ),
                "group": group_map
            }
        ).dropna()

        # ----------------------------------------------------
        # Tukey
        # ----------------------------------------------------

        if (
            posthoc_method == "tukey"
            and HAS_TUKEY
        ):

            try:

                tukey = pairwise_tukeyhsd(
                    endog=sub_df["value"].values,
                    groups=sub_df["group"].values,
                    alpha=0.05
                )

                ph = pd.DataFrame(
                    tukey._results_table.data[1:],
                    columns=tukey._results_table.data[0]
                )

                ph = ph.rename(
                    columns={
                        "meandiff": "Mean_Difference",
                        "p-adj": "p-value",
                        "lower": "CI_Lower",
                        "upper": "CI_Upper",
                        "reject": "Significant"
                    }
                )

                if "Mean_Difference" in ph.columns:

                    ph["Log2FC"] = pd.to_numeric(
                        ph["Mean_Difference"],
                        errors="coerce"
                    )

                    ph["Linear_FC"] = np.power(
                        2.0,
                        ph["Log2FC"]
                    )

            except Exception as exc:

                ph = pd.DataFrame(
                    {
                        "Error": [
                            str(exc)
                        ]
                    }
                )

        elif (
            posthoc_method == "tukey"
            and not HAS_TUKEY
        ):

            ph = pd.DataFrame(
                {
                    "Error": [
                        "Tukey HSD is unavailable."
                    ]
                }
            )

        # ----------------------------------------------------
        # Control-vs-all
        # ----------------------------------------------------

        elif posthoc_method in (
            "dunnett",
            "control",
            "control_vs_all",
            "control-vs-all"
        ):

            control = groups[0]

            rows = []

            control_values = (
                sub_df
                .loc[
                    sub_df["group"] == control,
                    "value"
                ]
                .values
            )

            for group in groups[1:]:

                group_values = (
                    sub_df
                    .loc[
                        sub_df["group"] == group,
                        "value"
                    ]
                    .values
                )

                if (
                    len(control_values) < 2
                    or
                    len(group_values) < 2
                ):
                    continue

                try:

                    _, p = stats.ttest_ind(
                        group_values,
                        control_values,
                        equal_var=False
                    )

                except Exception:

                    p = np.nan

                log2fc = (
                    np.mean(group_values)
                    -
                    np.mean(control_values)
                )

                linear_fc = np.power(
                    2.0,
                    log2fc
                )

                lo, hi = _welch_ci(
                    control_values,
                    group_values
                )

                rows.append(
                    {
                        "control": control,
                        "group": group,
                        "Mean_Log2_Control": np.mean(
                            control_values
                        ),
                        "Mean_Log2_Group": np.mean(
                            group_values
                        ),
                        "Log2FC": log2fc,
                        "Linear_FC": linear_fc,
                        "CI_Lower_Log2FC": lo,
                        "CI_Upper_Log2FC": hi,
                        "p-value": p
                    }
                )

            ph = pd.DataFrame(rows)

            if not ph.empty:

                ph["FDR"] = _fdr(
                    ph["p-value"].values
                )

                ph["Significant"] = (
                    ph["FDR"]
                    <
                    fdr_threshold
                )

        # ----------------------------------------------------
        # Pairwise Welch
        # ----------------------------------------------------

        elif posthoc_method in (
            "pairwise",
            "welch",
            "pairwise_welch"
        ):

            rows = []

            for group1, group2 in combinations(
                groups,
                2
            ):

                values1 = (
                    sub_df
                    .loc[
                        sub_df["group"] == group1,
                        "value"
                    ]
                    .values
                )

                values2 = (
                    sub_df
                    .loc[
                        sub_df["group"] == group2,
                        "value"
                    ]
                    .values
                )

                if (
                    len(values1) < 2
                    or
                    len(values2) < 2
                ):
                    continue

                try:

                    _, p = stats.ttest_ind(
                        values1,
                        values2,
                        equal_var=False
                    )

                except Exception:

                    p = np.nan

                log2fc = (
                    np.mean(values2)
                    -
                    np.mean(values1)
                )

                linear_fc = np.power(
                    2.0,
                    log2fc
                )

                lo, hi = _welch_ci(
                    values1,
                    values2
                )

                rows.append(
                    {
                        "group1": group1,
                        "group2": group2,
                        "Mean_Log2_Group1": np.mean(
                            values1
                        ),
                        "Mean_Log2_Group2": np.mean(
                            values2
                        ),
                        "Log2FC": log2fc,
                        "Linear_FC": linear_fc,
                        "CI_Lower_Log2FC": lo,
                        "CI_Upper_Log2FC": hi,
                        "p-value": p
                    }
                )

            ph = pd.DataFrame(rows)

            if not ph.empty:

                ph["FDR"] = _fdr(
                    ph["p-value"].values
                )

                ph["Significant"] = (
                    ph["FDR"]
                    <
                    fdr_threshold
                )

        else:

            raise ValueError(
                f"Unsupported posthoc method: {posthoc}. "
                "Use 'tukey', 'pairwise', or 'dunnett'."
            )

        posthoc_results[feature] = ph

    return (
        anova_table,
        posthoc_results
    )


# ============================================================
# 6. COMPLETE TWO-GROUP TABLE
# ============================================================

def complete_statistical_table(
    data_log2,
    group_a_samples,
    group_b_samples
):
    """
    Compatibility function for generating the standard
    two-group statistical table.
    """

    return two_group_test(
        data_log2=data_log2,
        group_a_samples=group_a_samples,
        group_b_samples=group_b_samples,
        method="ttest"
    )


# ============================================================
# 7. DIRECT COMPATIBILITY WRAPPERS
# ============================================================

def compare_two_groups(
    data_log2,
    group_a_samples,
    group_b_samples,
    method="ttest",
    **kwargs
):
    """
    Compatibility wrapper.

    Equivalent to:

        two_group_test()
    """

    return two_group_test(
        data_log2=data_log2,
        group_a_samples=group_a_samples,
        group_b_samples=group_b_samples,
        method=method,
        **kwargs
    )


def compare_groups(
    data_log2,
    group_a_samples=None,
    group_b_samples=None,
    method="ttest",
    **kwargs
):
    """
    Compatibility wrapper for older application code.

    Supports both:

        compare_groups(
            data,
            group_a_samples,
            group_b_samples
        )

    and keyword-based calls.
    """

    return two_group_test(
        data_log2=data_log2,
        group_a_samples=group_a_samples,
        group_b_samples=group_b_samples,
        method=method,
        **kwargs
    )


def two_group_stats(
    data_log2,
    group_a_samples,
    group_b_samples,
    method="ttest",
    **kwargs
):
    """
    Compatibility alias for two_group_test().
    """

    return two_group_test(
        data_log2=data_log2,
        group_a_samples=group_a_samples,
        group_b_samples=group_b_samples,
        method=method,
        **kwargs
    )


def perform_two_group_test(
    data_log2,
    group_a_samples,
    group_b_samples,
    method="ttest",
    **kwargs
):
    """
    Compatibility alias for two_group_test().
    """

    return two_group_test(
        data_log2=data_log2,
        group_a_samples=group_a_samples,
        group_b_samples=group_b_samples,
        method=method,
        **kwargs
    )


def run_two_group_test(
    data_log2,
    group_a_samples,
    group_b_samples,
    method="ttest",
    **kwargs
):
    """
    Compatibility alias for two_group_test().
    """

    return two_group_test(
        data_log2=data_log2,
        group_a_samples=group_a_samples,
        group_b_samples=group_b_samples,
        method=method,
        **kwargs
    )


# ============================================================
# 8. ANOVA COMPATIBILITY WRAPPERS
# ============================================================

def perform_anova(
    data_log2,
    group_map,
    posthoc="tukey",
    **kwargs
):
    """
    Compatibility alias for anova_test().
    """

    return anova_test(
        data_log2=data_log2,
        group_map=group_map,
        posthoc=posthoc,
        **kwargs
    )


def run_anova(
    data_log2,
    group_map,
    posthoc="tukey",
    **kwargs
):
    """
    Compatibility alias for anova_test().
    """

    return anova_test(
        data_log2=data_log2,
        group_map=group_map,
        posthoc=posthoc,
        **kwargs
    )


# ============================================================
# 9. GENERIC STATISTICAL ANALYSIS WRAPPER
# ============================================================

def statistical_analysis(
    data_log2,
    group_a_samples=None,
    group_b_samples=None,
    group_map=None,
    analysis_type="two-group",
    method="ttest",
    posthoc="tukey",
    **kwargs
):
    """
    General compatibility wrapper.

    Parameters
    ----------
    analysis_type : str

        Two-group options:

            "two-group"
            "two_group"
            "two"
            "ttest"

        ANOVA options:

            "anova"
            "one-way-anova"
            "one_way_anova"

    This wrapper allows older MetaboAI application code to
    call a single statistical_analysis() function.
    """

    analysis = str(
        analysis_type
    ).strip().lower()

    # --------------------------------------------------------
    # TWO-GROUP
    # --------------------------------------------------------

    if analysis in (
        "two-group",
        "two_group",
        "two",
        "ttest",
        "welch"
    ):

        if (
            group_a_samples is None
            or
            group_b_samples is None
        ):
            raise ValueError(
                "Two-group analysis requires "
                "group_a_samples and group_b_samples."
            )

        return two_group_test(
            data_log2=data_log2,
            group_a_samples=group_a_samples,
            group_b_samples=group_b_samples,
            method=method,
            **kwargs
        )

    # --------------------------------------------------------
    # ANOVA
    # --------------------------------------------------------

    if analysis in (
        "anova",
        "one-way-anova",
        "one_way_anova",
        "oneway"
    ):

        if group_map is None:
            raise ValueError(
                "ANOVA requires group_map."
            )

        return anova_test(
            data_log2=data_log2,
            group_map=group_map,
            posthoc=posthoc,
            **kwargs
        )

    raise ValueError(
        f"Unsupported analysis_type: {analysis_type}"
    )


# ============================================================
# 10. FOLD-CHANGE COMPATIBILITY FUNCTIONS
# ============================================================

def calculate_log2fc(
    data_log2,
    group_a_samples,
    group_b_samples
):
    """
    Calculate:

        Log2FC = Mean(Group B) - Mean(Group A)

    Returns
    -------
    pandas.Series
    """

    group_a = data_log2[
        group_a_samples
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    group_b = data_log2[
        group_b_samples
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    return (
        group_b.mean(axis=1)
        -
        group_a.mean(axis=1)
    )


def calculate_fold_change(
    data_log2,
    group_a_samples,
    group_b_samples
):
    """
    Calculate linear fold-change:

        FC = 2 ** Log2FC

    Direction:

        Group B / Group A
    """

    log2fc = calculate_log2fc(
        data_log2,
        group_a_samples,
        group_b_samples
    )

    return np.power(
        2.0,
        log2fc
    )


def calculate_linear_fc(
    data_log2,
    group_a_samples,
    group_b_samples
):
    """
    Compatibility alias for calculate_fold_change().
    """

    return calculate_fold_change(
        data_log2=data_log2,
        group_a_samples=group_a_samples,
        group_b_samples=group_b_samples
    )


# ============================================================
# 11. RESULT FILTERING HELPER
# ============================================================

def filter_significant(
    result,
    pvalue_threshold=DEFAULT_PVALUE_THRESHOLD,
    fdr_threshold=DEFAULT_FDR_THRESHOLD
):
    """
    Filter a two-group statistical result table.

    Requires:

        p-value < pvalue_threshold

    AND

        FDR < fdr_threshold
    """

    if not isinstance(
        result,
        pd.DataFrame
    ):
        raise TypeError(
            "result must be a pandas DataFrame."
        )

    required = {
        "p-value",
        "FDR"
    }

    missing = required.difference(
        result.columns
    )

    if missing:
        raise ValueError(
            f"Result is missing columns: {sorted(missing)}"
        )

    mask = (
        result["p-value"]
        <
        pvalue_threshold
    ) & (
        result["FDR"]
        <
        fdr_threshold
    )

    return result.loc[
        mask
    ].copy()


# ============================================================
# 12. MODULE EXPORTS
# ============================================================

__all__ = [
    # Core functions
    "_fdr",
    "_welch_ci",
    "two_group_test",
    "anova_test",
    "complete_statistical_table",

    # Compatibility wrappers
    "compare_two_groups",
    "compare_groups",
    "two_group_stats",
    "perform_two_group_test",
    "run_two_group_test",
    "perform_anova",
    "run_anova",
    "statistical_analysis",

    # Fold-change helpers
    "calculate_log2fc",
    "calculate_fold_change",
    "calculate_linear_fc",

    # Filtering
    "filter_significant",

    # Capability flag
    "HAS_TUKEY"
]


# ============================================================
# 13. BASIC SELF-TEST
# ============================================================

if __name__ == "__main__":

    np.random.seed(42)

    test_data = pd.DataFrame(
        {
            "Control_1": [10.0, 8.0, 12.0],
            "Control_2": [10.2, 8.2, 11.8],
            "Control_3": [9.8, 7.9, 12.1],

            "Treatment_1": [11.0, 8.8, 12.0],
            "Treatment_2": [11.2, 9.0, 12.2],
            "Treatment_3": [10.8, 8.9, 11.9]
        },
        index=[
            "Metabolite_A",
            "Metabolite_B",
            "Metabolite_C"
        ]
    )

    result = two_group_test(
        data_log2=test_data,
        group_a_samples=[
            "Control_1",
            "Control_2",
            "Control_3"
        ],
        group_b_samples=[
            "Treatment_1",
            "Treatment_2",
            "Treatment_3"
        ]
    )

    print("\nTwo-group analysis")
    print("=" * 70)
    print(result)

    print("\nLog2FC")
    print("=" * 70)

    print(
        calculate_log2fc(
            test_data,
            [
                "Control_1",
                "Control_2",
                "Control_3"
            ],
            [
                "Treatment_1",
                "Treatment_2",
                "Treatment_3"
            ]
        )
    )

    print("\nLinear FC")
    print("=" * 70)

    print(
        calculate_fold_change(
            test_data,
            [
                "Control_1",
                "Control_2",
                "Control_3"
            ],
            [
                "Treatment_1",
                "Treatment_2",
                "Treatment_3"
            ]
        )
    )

    print("\nModule loaded successfully.")
```

```
