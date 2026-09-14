```python
"""
stats_analysis.py
------------------------------------------------------------
Two-group and multi-group (ANOVA) statistical comparison.

IMPORTANT:
    Input data must already be:
        1. ISTD normalized
        2. log2 transformed
        3. IQR normalized, if IQR normalization is part of
           the upstream workflow

Statistics are calculated ONLY from the log2-scale data.

Two-group fold-change convention:

    Group A = reference/control
    Group B = treatment/experimental

    Log2FC = Mean(Group B) - Mean(Group A)

    Linear FC = 2 ** Log2FC

Therefore:

    Log2FC > 0  --> higher in Group B
    Log2FC < 0  --> lower in Group B

Example:

    Untreated = Group A
    IR_Day_2  = Group B

    ATP:

        Log2FC = Mean(IR_Day_2) - Mean(Untreated)
               = 1.08113

        Linear FC = 2 ** 1.08113
                  = 2.11569

------------------------------------------------------------
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


# ============================================================
# 1. FDR CORRECTION
# ============================================================

def _fdr(pvals):
    """
    Benjamini-Hochberg FDR correction.

    NaN p-values are preserved as NaN.
    """
    pvals = np.asarray(pvals, dtype=float)

    mask = ~np.isnan(pvals)

    fdr = np.full(pvals.shape, np.nan)

    if mask.sum() > 0:
        fdr[mask] = multipletests(
            pvals[mask],
            method="fdr_bh"
        )[1]

    return fdr


# ============================================================
# 2. WELCH 95% CI
# ============================================================

def _welch_ci(
    x,
    y,
    confidence: float = 0.95
):
    """
    Welch 95% CI for:

        mean(y) - mean(x)

    This follows the same direction as the fold-change:

        Log2FC = Group B - Group A

    Parameters
    ----------
    x : Group A values
    y : Group B values

    Returns
    -------
    lower, upper
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx = len(x)
    ny = len(y)

    mean_diff = y.mean() - x.mean()

    var_x = x.var(ddof=1)
    var_y = y.var(ddof=1)

    se = np.sqrt(
        var_x / nx +
        var_y / ny
    )

    denominator = (
        (var_x / nx) ** 2 / (nx - 1)
        +
        (var_y / ny) ** 2 / (ny - 1)
    )

    numerator = (
        var_x / nx +
        var_y / ny
    ) ** 2

    if denominator > 0:
        df = numerator / denominator
    else:
        df = nx + ny - 2

    alpha = 1 - confidence

    t_critical = stats.t.ppf(
        1 - alpha / 2,
        df
    )

    lower = mean_diff - t_critical * se
    upper = mean_diff + t_critical * se

    return lower, upper


# ============================================================
# 3. TWO-GROUP STATISTICAL TEST
# ============================================================

def two_group_test(
    data_log2: pd.DataFrame,
    group_a_samples,
    group_b_samples,
    method: str = "ttest"
):
    """
    Perform two-group statistical comparison.

    --------------------------------------------------------
    GROUP DIRECTION
    --------------------------------------------------------

    Group A = reference/control
    Group B = experimental/treatment

    Log2FC:

        Mean(Group B) - Mean(Group A)

    Linear FC:

        2 ** Log2FC

    Therefore:

        FC > 1
            Group B is higher than Group A

        FC < 1
            Group B is lower than Group A

    --------------------------------------------------------
    INPUT
    --------------------------------------------------------

    data_log2:
        Features x samples DataFrame.

        Data must already be log2 transformed.

    group_a_samples:
        Reference/control sample names.

    group_b_samples:
        Experimental/treatment sample names.

    method:
        "ttest"
            Welch's independent t-test

        "wilcoxon"
            Mann-Whitney U test

    --------------------------------------------------------
    OUTPUT
    --------------------------------------------------------

    DataFrame containing:

        Mean_Log2_GroupA
        Mean_Log2_GroupB
        Linear_FC
        Log2FC
        CI_Lower_Log2FC
        CI_Upper_Log2FC
        p-value
        FDR
        Significant

    --------------------------------------------------------
    IMPORTANT
    --------------------------------------------------------

    No additional log2 transformation is performed here.

    The data are assumed to already be on the log2 scale.
    """

    # --------------------------------------------------------
    # Validate sample names
    # --------------------------------------------------------

    missing_a = [
        s for s in group_a_samples
        if s not in data_log2.columns
    ]

    missing_b = [
        s for s in group_b_samples
        if s not in data_log2.columns
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

    # --------------------------------------------------------
    # Select groups
    # --------------------------------------------------------

    a2 = data_log2[group_a_samples]
    b2 = data_log2[group_b_samples]

    pvals = []
    ci_lower = []
    ci_upper = []

    # --------------------------------------------------------
    # Statistical testing feature-by-feature
    # --------------------------------------------------------

    for feat in data_log2.index:

        x = (
            a2.loc[feat]
            .dropna()
            .astype(float)
            .values
        )

        y = (
            b2.loc[feat]
            .dropna()
            .astype(float)
            .values
        )

        # ----------------------------------------------------
        # Minimum sample requirement
        # ----------------------------------------------------

        if len(x) < 2 or len(y) < 2:

            pvals.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)

            continue

        # ----------------------------------------------------
        # Statistical test
        # ----------------------------------------------------

        if method.lower() == "ttest":

            # Welch's t-test
            _, p = stats.ttest_ind(
                x,
                y,
                equal_var=False,
                nan_policy="omit"
            )

        elif method.lower() in [
            "wilcoxon",
            "mannwhitney",
            "mann-whitney"
        ]:

            # Mann-Whitney U test
            _, p = stats.mannwhitneyu(
                x,
                y,
                alternative="two-sided"
            )

        else:

            raise ValueError(
                "method must be 'ttest' or 'wilcoxon'"
            )

        pvals.append(p)

        # ----------------------------------------------------
        # CI follows Group B - Group A direction
        # ----------------------------------------------------

        lo, hi = _welch_ci(
            x,
            y
        )

        ci_lower.append(lo)
        ci_upper.append(hi)

    # ========================================================
    # GROUP MEANS
    # ========================================================

    mean_a = a2.mean(axis=1)
    mean_b = b2.mean(axis=1)

    # ========================================================
    # LOG2 FOLD CHANGE
    # ========================================================

    # IMPORTANT:
    #
    # Group B - Group A
    #
    # This is the desired direction.
    #
    # Example:
    #
    # IR Day 2 - Untreated
    #
    # Positive value means higher in IR Day 2.

    log2fc = mean_b - mean_a

    # ========================================================
    # LINEAR FOLD CHANGE
    # ========================================================

    # Since the data are already log2 transformed:
    #
    # Linear FC = 2 ^ Log2FC

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
    # RESULT TABLE
    # ========================================================

    result = pd.DataFrame(
        {
            "Mean_Log2_GroupA": mean_a,
            "Mean_Log2_GroupB": mean_b,

            # Group B / Group A
            "Linear_FC": linear_fc,

            # Group B - Group A
            "Log2FC": log2fc,

            "CI_Lower_Log2FC": ci_lower,
            "CI_Upper_Log2FC": ci_upper,

            "p-value": pvals,
            "FDR": fdr,
        },
        index=data_log2.index
    )

    # ========================================================
    # SIGNIFICANCE
    # ========================================================

    result["Significant"] = (
        (result["p-value"] < 0.05)
        &
        (result["FDR"] < 0.25)
    )

    # ========================================================
    # DIRECTION
    # ========================================================

    result["Direction"] = np.where(
        result["Log2FC"] > 0,
        "Higher in Group B",
        np.where(
            result["Log2FC"] < 0,
            "Lower in Group B",
            "No change"
        )
    )

    # ========================================================
    # SORT BY P-VALUE
    # ========================================================

    result = result.sort_values(
        "p-value",
        na_position="last"
    )

    return result


# ============================================================
# 4. ONE-WAY ANOVA
# ============================================================

def anova_test(
    data_log2: pd.DataFrame,
    group_map: pd.Series,
    posthoc: str = "tukey"
):
    """
    One-way ANOVA across >=3 groups.

    The ANOVA is performed on the existing log2-scale data.

    group_map:
        Series indexed by sample name containing group labels.

    posthoc:
        "tukey"
        "dunnett"
        "pairwise"

    Returns
    -------
    anova_table
    posthoc_results
    """

    # --------------------------------------------------------
    # Validate group map
    # --------------------------------------------------------

    missing_samples = [
        s for s in group_map.index
        if s not in data_log2.columns
    ]

    if missing_samples:
        raise ValueError(
            f"Samples in group_map not found in data: "
            f"{missing_samples}"
        )

    # Keep only samples available in data
    group_map = group_map.loc[
        group_map.index.intersection(
            data_log2.columns
        )
    ]

    groups = (
        group_map
        .dropna()
        .unique()
        .tolist()
    )

    if len(groups) < 3:
        raise ValueError(
            "ANOVA module requires 3 or more groups."
        )

    # ========================================================
    # ANOVA
    # ========================================================

    f_stats = []
    pvals = []

    for feat in data_log2.index:

        samples_by_group = []

        for g in groups:

            samples = group_map.index[
                group_map == g
            ]

            values = (
                data_log2
                .loc[feat, samples]
                .dropna()
                .astype(float)
                .values
            )

            if len(values) >= 2:
                samples_by_group.append(values)

        if len(samples_by_group) < 2:

            f_stats.append(np.nan)
            pvals.append(np.nan)

            continue

        f, p = stats.f_oneway(
            *samples_by_group
        )

        f_stats.append(f)
        pvals.append(p)

    # ========================================================
    # FDR
    # ========================================================

    fdr = _fdr(
        pvals
    )

    # ========================================================
    # ANOVA TABLE
    # ========================================================

    anova_table = pd.DataFrame(
        {
            "F-statistic": f_stats,
            "ANOVA p-value": pvals,
            "FDR": fdr,
        },
        index=data_log2.index
    )

    anova_table = anova_table.sort_values(
        "ANOVA p-value",
        na_position="last"
    )

    # ========================================================
    # SIGNIFICANT FEATURES
    # ========================================================

    sig_features = anova_table[
        anova_table["FDR"] < 0.25
    ].index.tolist()

    posthoc_results = {}

    # ========================================================
    # POST-HOC
    # ========================================================

    for feat in sig_features:

        vals = data_log2.loc[feat]

        sub_df = pd.DataFrame(
            {
                "value": vals,
                "group": group_map
            }
        ).dropna()

        # ----------------------------------------------------
        # Tukey
        # ----------------------------------------------------

        if (
            posthoc.lower() == "tukey"
            and HAS_TUKEY
        ):

            try:

                res = pairwise_tukeyhsd(
                    endog=sub_df["value"],
                    groups=sub_df["group"]
                )

                ph = pd.DataFrame(
                    data=res._results_table.data[1:],
                    columns=res._results_table.data[0]
                )

            except Exception:

                ph = pd.DataFrame()

        # ----------------------------------------------------
        # Dunnett approximation
        # ----------------------------------------------------

        elif posthoc.lower() == "dunnett":

            control = groups[0]

            rows = []

            ctrl_vals = (
                sub_df
                .loc[
                    sub_df["group"] == control,
                    "value"
                ]
                .values
            )

            for g in groups:

                if g == control:
                    continue

                gv = (
                    sub_df
                    .loc[
                        sub_df["group"] == g,
                        "value"
                    ]
                    .values
                )

                if (
                    len(gv) >= 2
                    and len(ctrl_vals) >= 2
                ):

                    _, p = stats.ttest_ind(
                        gv,
                        ctrl_vals,
                        equal_var=False
                    )

                    rows.append(
                        {
                            "control": control,
                            "group": g,
                            "p-value": p
                        }
                    )

            ph = pd.DataFrame(rows)

            if len(ph):

                ph["FDR"] = _fdr(
                    ph["p-value"].values
                )

        # ----------------------------------------------------
        # Pairwise Welch t-tests
        # ----------------------------------------------------

        else:

            rows = []

            for g1, g2 in combinations(
                groups,
                2
            ):

                v1 = (
                    sub_df
                    .loc[
                        sub_df["group"] == g1,
                        "value"
                    ]
                    .values
                )

                v2 = (
                    sub_df
                    .loc[
                        sub_df["group"] == g2,
                        "value"
                    ]
                    .values
                )

                if (
                    len(v1) >= 2
                    and len(v2) >= 2
                ):

                    _, p = stats.ttest_ind(
                        v1,
                        v2,
                        equal_var=False
                    )

                    # Direction:
                    #
                    # g2 - g1

                    log2fc = (
                        np.mean(v2)
                        -
                        np.mean(v1)
                    )

                    linear_fc = (
                        2 ** log2fc
                    )

                    rows.append(
                        {
                            "group1": g1,
                            "group2": g2,
                            "Mean_Log2_Group1": np.mean(v1),
                            "Mean_Log2_Group2": np.mean(v2),
                            "Log2FC": log2fc,
                            "Linear_FC": linear_fc,
                            "p-value": p
                        }
                    )

            ph = pd.DataFrame(rows)

            if len(ph):

                ph["FDR"] = _fdr(
                    ph["p-value"].values
                )

        posthoc_results[feat] = ph

    return (
        anova_table,
        posthoc_results
    )


# ============================================================
# 5. COMPLETE TWO-GROUP TABLE
# ============================================================

def complete_statistical_table(
    data_log2: pd.DataFrame,
    group_a_samples,
    group_b_samples
):
    """
    Generate the complete two-group statistical table.

    Group A = reference/control
    Group B = experimental/treatment

    Fold-change:

        Log2FC = Group B - Group A

        Linear FC = 2 ** Log2FC

    Statistical test:

        Welch's independent t-test

    Significance:

        p < 0.05
        AND
        FDR < 0.25
    """

    return two_group_test(
        data_log2=data_log2,
        group_a_samples=group_a_samples,
        group_b_samples=group_b_samples,
        method="ttest"
    )
```
