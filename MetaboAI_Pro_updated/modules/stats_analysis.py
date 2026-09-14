"""
stats_analysis.py
Two-group and multi-group (ANOVA) statistical comparison module.

Statistical rules:
    1. Log transformation is STRICT log2(x).
       - No pseudocount
       - No constant addition
       - No shifting
    2. Fold change is calculated from log2-transformed data:
           Log2FC = Mean(Log2 Group B) - Mean(Log2 Group A)
           Linear_FC = 2 ** Log2FC
    3. For the standard two-group comparison:
           Group A = Untreated
           Group B = IR Day 2
    4. Output statistical table contains ONLY:
           p-value
           FDR
           Linear_FC
           Log2FC
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    HAS_TUKEY = True
except ImportError:
    HAS_TUKEY = False


# ---------------------------------------------------------------------
# FDR
# ---------------------------------------------------------------------

def _fdr(pvals):
    """
    Benjamini-Hochberg FDR correction.

    NaN p-values remain NaN.
    """
    pvals = np.asarray(pvals, dtype=float)

    mask = np.isfinite(pvals)

    fdr = np.full(pvals.shape, np.nan, dtype=float)

    if mask.sum() > 0:
        fdr[mask] = multipletests(
            pvals[mask],
            method="fdr_bh"
        )[1]

    return fdr


# ---------------------------------------------------------------------
# STRICT LOG2
# ---------------------------------------------------------------------

def strict_log2(data):
    """
    Apply STRICT log2(x).

    IMPORTANT:
        No pseudocount is added.
        No constant is added.
        No shifting is performed.

    Values <= 0 are converted to NaN because log2(x) is undefined
    for zero or negative values.

    Parameters
    ----------
    data : pandas DataFrame or Series

    Returns
    -------
    pandas DataFrame or Series
        Strict log2-transformed data.
    """

    result = data.copy()

    # Strict log2 only.
    # Do NOT add pseudocounts, constants, or shifts.
    result = result.where(result > 0)

    return np.log2(result)


# ---------------------------------------------------------------------
# TWO-GROUP TEST
# ---------------------------------------------------------------------

def two_group_test(
    data_log2: pd.DataFrame,
    group_a_samples,
    group_b_samples,
    method: str = "ttest"
):
    """
    Perform a two-group statistical comparison.

    Data must already be STRICT log2-transformed.

    Group A:
        Untreated

    Group B:
        IR Day 2

    Fold-change definitions:

        Log2FC =
            Mean(Log2 Group B) - Mean(Log2 Group A)

        Linear_FC =
            2 ** Log2FC

    Statistical test:
        ttest:
            Welch's independent two-sample t-test

        wilcoxon:
            Mann-Whitney U test

    Returned table contains ONLY:

        p-value
        FDR
        Linear_FC
        Log2FC

    Parameters
    ----------
    data_log2 : DataFrame
        Strict log2-transformed normalized data.
        Rows = features
        Columns = samples

    group_a_samples : list
        Samples belonging to Group A (Untreated).

    group_b_samples : list
        Samples belonging to Group B (IR Day 2).

    method : str
        "ttest" or "wilcoxon"

    Returns
    -------
    DataFrame
    """

    if method not in {"ttest", "wilcoxon"}:
        raise ValueError(
            "method must be either 'ttest' or 'wilcoxon'."
        )

    # Validate sample names
    missing_a = [
        sample for sample in group_a_samples
        if sample not in data_log2.columns
    ]

    missing_b = [
        sample for sample in group_b_samples
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

    a2 = data_log2[group_a_samples]
    b2 = data_log2[group_b_samples]

    pvals = []

    # -------------------------------------------------------------
    # Statistical test feature-by-feature
    # -------------------------------------------------------------

    for feature in data_log2.index:

        x = pd.to_numeric(
            a2.loc[feature],
            errors="coerce"
        ).dropna().values

        y = pd.to_numeric(
            b2.loc[feature],
            errors="coerce"
        ).dropna().values

        # Need at least two observations in each group
        if len(x) < 2 or len(y) < 2:
            pvals.append(np.nan)
            continue

        try:

            if method == "ttest":

                # Welch's t-test
                _, p = stats.ttest_ind(
                    x,
                    y,
                    equal_var=False,
                    nan_policy="omit"
                )

            else:

                # Mann-Whitney U / Wilcoxon rank-sum
                _, p = stats.mannwhitneyu(
                    x,
                    y,
                    alternative="two-sided"
                )

            pvals.append(float(p))

        except Exception:
            pvals.append(np.nan)

    # -------------------------------------------------------------
    # Means calculated ONLY from strict log2 data
    # -------------------------------------------------------------

    mean_a = a2.mean(axis=1, skipna=True)
    mean_b = b2.mean(axis=1, skipna=True)

    # -------------------------------------------------------------
    # Log2 Fold Change
    #
    # Group A = Untreated
    # Group B = IR Day 2
    #
    # Log2FC = B - A
    # -------------------------------------------------------------

    log2fc = mean_b - mean_a

    # -------------------------------------------------------------
    # Linear Fold Change
    #
    # Linear_FC = 2 ** Log2FC
    #
    # > 1  = higher in Group B
    # < 1  = lower in Group B
    # = 1  = equal
    # -------------------------------------------------------------

    linear_fc = np.power(2.0, log2fc)

    # -------------------------------------------------------------
    # FDR
    # -------------------------------------------------------------

    fdr = _fdr(pvals)

    # -------------------------------------------------------------
    # FINAL OUTPUT
    #
    # ONLY these four columns are returned.
    # -------------------------------------------------------------

    result = pd.DataFrame(
        {
            "p-value": pvals,
            "FDR": fdr,
            "Linear_FC": linear_fc,
            "Log2FC": log2fc,
        },
        index=data_log2.index
    )

    # Sort by p-value
    result = result.sort_values(
        by="p-value",
        na_position="last"
    )

    return result


# ---------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------

def anova_test(
    data_log2: pd.DataFrame,
    group_map: pd.Series,
    posthoc: str = "tukey"
):
    """
    One-way ANOVA across >=3 groups.

    Data must already be STRICT log2-transformed.

    Parameters
    ----------
    data_log2 : DataFrame
        Strict log2-transformed normalized data.

    group_map : Series
        Series indexed by sample name with group labels.

    posthoc : str
        "tukey", "dunnett", or "pairwise"

    Returns
    -------
    anova_table : DataFrame

    posthoc_results : dict
    """

    if not isinstance(group_map, pd.Series):
        raise TypeError(
            "group_map must be a pandas Series."
        )

    # Keep only samples that actually exist in data
    valid_samples = [
        sample
        for sample in group_map.index
        if sample in data_log2.columns
    ]

    if not valid_samples:
        raise ValueError(
            "No samples from group_map were found in data_log2."
        )

    group_map = group_map.loc[valid_samples]

    # Remove missing group labels
    group_map = group_map.dropna()

    groups = group_map.unique().tolist()

    if len(groups) < 3:
        raise ValueError(
            "ANOVA module requires 3 or more groups."
        )

    # -------------------------------------------------------------
    # Validate group sizes
    # -------------------------------------------------------------

    invalid_groups = []

    for group in groups:

        n = int((group_map == group).sum())

        if n < 2:
            invalid_groups.append(
                f"{group} (n={n})"
            )

    if invalid_groups:
        raise ValueError(
            "Every ANOVA group must contain at least 2 samples. "
            f"Invalid groups: {', '.join(invalid_groups)}"
        )

    # -------------------------------------------------------------
    # ANOVA feature-by-feature
    # -------------------------------------------------------------

    f_stats = []
    pvals = []

    for feature in data_log2.index:

        samples_by_group = []

        valid_feature = True

        for group in groups:

            samples = data_log2.loc[
                feature,
                group_map.index[group_map == group]
            ]

            samples = pd.to_numeric(
                samples,
                errors="coerce"
            ).dropna().values

            if len(samples) < 2:
                valid_feature = False
                break

            samples_by_group.append(samples)

        if not valid_feature:
            f_stats.append(np.nan)
            pvals.append(np.nan)
            continue

        try:

            f, p = stats.f_oneway(
                *samples_by_group
            )

            f_stats.append(float(f))
            pvals.append(float(p))

        except Exception:

            f_stats.append(np.nan)
            pvals.append(np.nan)

    # -------------------------------------------------------------
    # FDR across ANOVA features
    # -------------------------------------------------------------

    fdr = _fdr(pvals)

    anova_table = pd.DataFrame(
        {
            "F-statistic": f_stats,
            "ANOVA p-value": pvals,
            "FDR": fdr,
        },
        index=data_log2.index
    )

    anova_table = anova_table.sort_values(
        by="ANOVA p-value",
        na_position="last"
    )

    # -------------------------------------------------------------
    # Post-hoc
    # -------------------------------------------------------------

    sig_features = anova_table[
        anova_table["FDR"] < 0.25
    ].index.tolist()

    posthoc_results = {}

    for feature in sig_features:

        vals = data_log2.loc[feature]

        sub_df = pd.DataFrame(
            {
                "value": vals,
                "group": group_map
            }
        )

        sub_df["value"] = pd.to_numeric(
            sub_df["value"],
            errors="coerce"
        )

        sub_df = sub_df.dropna(
            subset=["value", "group"]
        )

        # ---------------------------------------------------------
        # Tukey HSD
        # ---------------------------------------------------------

        if posthoc == "tukey" and HAS_TUKEY:

            try:

                res = pairwise_tukeyhsd(
                    endog=sub_df["value"],
                    groups=sub_df["group"]
                )

                ph = pd.DataFrame(
                    data=res._results_table.data[1:],
                    columns=res._results_table.data[0]
                )

                # Standardize Tukey column names
                rename_map = {
                    "group1": "Group1",
                    "group2": "Group2",
                    "meandiff": "Mean_Difference",
                    "p-adj": "p-value",
                    "lower": "CI_Lower",
                    "upper": "CI_Upper",
                    "reject": "Reject",
                }

                ph = ph.rename(
                    columns=rename_map
                )

            except Exception:

                ph = pd.DataFrame()

        # ---------------------------------------------------------
        # Dunnett-style comparisons
        #
        # NOTE:
        # scipy/statsmodels availability varies by version.
        # This fallback performs Welch comparisons against the
        # first group and BH correction.
        # ---------------------------------------------------------

        elif posthoc == "dunnett":

            control = groups[0]

            rows = []

            ctrl_vals = sub_df.loc[
                sub_df["group"] == control,
                "value"
            ].values

            for group in groups:

                if group == control:
                    continue

                group_vals = sub_df.loc[
                    sub_df["group"] == group,
                    "value"
                ].values

                if (
                    len(group_vals) >= 2
                    and len(ctrl_vals) >= 2
                ):

                    try:

                        _, p = stats.ttest_ind(
                            group_vals,
                            ctrl_vals,
                            equal_var=False
                        )

                        rows.append(
                            {
                                "control": control,
                                "group": group,
                                "p-value": float(p),
                            }
                        )

                    except Exception:
                        continue

            ph = pd.DataFrame(rows)

            if not ph.empty:
                ph["FDR"] = _fdr(
                    ph["p-value"].values
                )

        # ---------------------------------------------------------
        # Pairwise Welch comparisons
        # ---------------------------------------------------------

        else:

            rows = []

            for group1, group2 in combinations(groups, 2):

                v1 = sub_df.loc[
                    sub_df["group"] == group1,
                    "value"
                ].values

                v2 = sub_df.loc[
                    sub_df["group"] == group2,
                    "value"
                ].values

                if len(v1) < 2 or len(v2) < 2:
                    continue

                try:

                    _, p = stats.ttest_ind(
                        v1,
                        v2,
                        equal_var=False
                    )

                    rows.append(
                        {
                            "group1": group1,
                            "group2": group2,
                            "p-value": float(p),
                        }
                    )

                except Exception:
                    continue

            ph = pd.DataFrame(rows)

            if not ph.empty:

                ph["FDR"] = _fdr(
                    ph["p-value"].values
                )

        posthoc_results[feature] = ph

    return anova_table, posthoc_results


# ---------------------------------------------------------------------
# COMPLETE TWO-GROUP TABLE
# ---------------------------------------------------------------------

def complete_statistical_table(
    data_log2: pd.DataFrame,
    group_a_samples,
    group_b_samples
):
    """
    Complete two-group statistical analysis.

    Group A should normally be:
        Untreated

    Group B should normally be:
        IR Day 2

    Returns ONLY:

        p-value
        FDR
        Linear_FC
        Log2FC

    No mean columns.
    No confidence interval columns.
    No Significant column.
    """

    return two_group_test(
        data_log2=data_log2,
        group_a_samples=group_a_samples,
        group_b_samples=group_b_samples,
        method="ttest"
    )
