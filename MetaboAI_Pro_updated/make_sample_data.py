"""
Generate two synthetic demo datasets for MetaboAI Pro:
  1. TARGETED   - includes an ISTD row, smaller panel (typical of targeted assays)
  2. UNTARGETED - no ISTD row, larger feature set (typical of untargeted assays)

Both use the same 4-group design (Control / Mild / Moderate / Severe) with QC
replicates, so the demo also exercises ANOVA (>=3 groups) out of the box, not
just the two-group t-test.
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GROUPS = ["Control", "Mild", "Moderate", "Severe"]
N_PER_GROUP = 6
N_QC = 6


def _build_sample_metadata(rng):
    sample_names, group_labels, is_qc, batch = [], [], [], []
    for g in GROUPS:
        for i in range(N_PER_GROUP):
            sample_names.append(f"{g}_{i+1}")
            group_labels.append(g)
            is_qc.append(False)
            batch.append("1" if i < N_PER_GROUP // 2 else "2")
    for i in range(N_QC):
        sample_names.append(f"QC_{i+1}")
        group_labels.append("QC")
        is_qc.append(True)
        batch.append("1" if i < N_QC // 2 else "2")
    return sample_names, group_labels, is_qc, batch


def generate_dataset(n_metabolites: int, include_istd: bool, seed: int, out_prefix: str, write_meta: bool = True):
    rng = np.random.default_rng(seed)
    sample_names, group_labels, is_qc, batch = _build_sample_metadata(rng)
    n_samples = len(sample_names)

    metabolite_names = [f"Metabolite_{i+1:03d}" for i in range(n_metabolites)]
    if include_istd:
        metabolite_names[0] = "ISTD_D4-Alanine"

    base = rng.lognormal(mean=10, sigma=1.0, size=(n_metabolites, 1))
    data = base * rng.lognormal(mean=0, sigma=0.15, size=(n_metabolites, n_samples))

    group_idx = {g: [i for i, lab in enumerate(group_labels) if lab == g] for g in GROUPS}
    start_feat = 1 if include_istd else 0

    # ~15% of features: progressive trend across Control -> Mild -> Moderate -> Severe
    n_trend = max(3, int(0.15 * n_metabolites))
    trend_feats = rng.choice(range(start_feat, n_metabolites), size=n_trend, replace=False)
    stage_multipliers = {
        "Control": 1.0, "Mild": rng.uniform(1.3, 1.8), "Moderate": None, "Severe": None,
    }
    for f in trend_feats:
        direction = rng.choice([1, -1])  # increasing or decreasing severity trend
        step = rng.uniform(0.35, 0.7)
        for stage_i, g in enumerate(GROUPS):
            if g == "Control":
                continue
            fold = (1 + direction * step * stage_i)
            fold = max(fold, 0.15)  # keep positive
            data[f, group_idx[g]] *= fold

    # ~8% of features: only "Severe" differs sharply (acute marker pattern)
    remaining = [i for i in range(start_feat, n_metabolites) if i not in trend_feats]
    n_acute = max(2, int(0.08 * n_metabolites))
    acute_feats = rng.choice(remaining, size=min(n_acute, len(remaining)), replace=False)
    for f in acute_feats:
        fold = rng.choice([3.0, 3.5, 0.25, 0.3])
        data[f, group_idx["Severe"]] *= fold

    if include_istd:
        # ISTD row: fairly constant across all samples (small technical variation only)
        data[0, :] = rng.lognormal(mean=9, sigma=0.05, size=n_samples)

    # Inject realistic missingness (encoded as exact zero, the common LC-MS convention
    # for "not detected") into biological samples only -- QC replicates and the ISTD
    # row stay complete, matching real-world expectations of high QC reproducibility
    # and a reliably-detected internal standard. Mixture of missingness rates so the
    # demo exercises every category in the Data Cleaning & Imputation tab:
    #   ~70% of features: 0-15% missing (typical low-level "not detected" noise)
    #   ~20% of features: 20-50% missing (moderate -- "keep with careful imputation")
    #   ~10% of features: 55-85% missing (heavy -- demonstrates the removal guidance)
    bio_sample_idx = [i for i, is_q in enumerate(is_qc) if not is_q]
    for f in range(start_feat, n_metabolites):
        tier = rng.random()
        if tier < 0.70:
            miss_rate = rng.uniform(0.0, 0.15)
        elif tier < 0.90:
            miss_rate = rng.uniform(0.20, 0.50)
        else:
            miss_rate = rng.uniform(0.55, 0.85)
        n_miss = int(round(miss_rate * len(bio_sample_idx)))
        if n_miss > 0:
            miss_cols = rng.choice(bio_sample_idx, size=n_miss, replace=False)
            data[f, miss_cols] = 0.0

    peak_df = pd.DataFrame(data, index=metabolite_names, columns=sample_names)
    peak_df.insert(0, "Metabolite", metabolite_names)
    peak_path = f"{BASE_DIR}/sample_peak_area_matrix_{out_prefix}.csv"
    peak_df.to_csv(peak_path, index=False)

    meta_df = pd.DataFrame({
        "Sample": sample_names, "Group": group_labels, "IsQC": is_qc, "Batch": batch,
    })
    if write_meta:
        meta_path = f"{BASE_DIR}/sample_metadata_{out_prefix}.csv"
        meta_df.to_csv(meta_path, index=False)

    print(f"[{out_prefix}] {peak_df.shape[0]} features x {len(sample_names)} samples "
          f"({N_QC} QC, {len(sample_names) - N_QC} biological across {len(GROUPS)} groups) "
          f"-> {peak_path}")
    return peak_df, meta_df


# 1. Targeted demo: includes ISTD, smaller panel (typical of targeted assays)
generate_dataset(n_metabolites=60, include_istd=True, seed=42, out_prefix="targeted")

# 2. Untargeted demo: no ISTD, larger feature set (typical of untargeted assays)
generate_dataset(n_metabolites=300, include_istd=False, seed=43, out_prefix="untargeted")


# ---------------------------------------------------------------------------
# 3. Multi-method demo: the SAME biological samples run through all FOUR analysis
# types the app supports (Untargeted Metabolomics, Targeted Metabolomics,
# Untargeted Lipidomics, Targeted Lipidomics) — the common real-world design this
# app's multi-dataset "Combine Datasets" workflow is built for, where a study
# combines multiple assay types run on the same samples. Each dataset gets its own
# feature panel (targeted assays get a smaller panel + an ISTD row; untargeted
# assays get a larger panel with no single spiked standard) but all four share ONE
# metadata table, since it's the same underlying samples.
# ---------------------------------------------------------------------------
def generate_multi_method_demo():
    rng = np.random.default_rng(100)
    sample_names, group_labels, is_qc, batch = _build_sample_metadata(rng)

    meta_df = pd.DataFrame({"Sample": sample_names, "Group": group_labels, "IsQC": is_qc, "Batch": batch})
    meta_path = f"{BASE_DIR}/sample_metadata_multimethod.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"[multimethod] shared metadata -> {meta_path}")

    # (analysis type label, feature count, include ISTD row, RNG seed)
    method_configs = [
        ("Untargeted Metabolomics", 120, False, 201),
        ("Targeted Metabolomics", 50, True, 202),
        ("Untargeted Lipidomics", 90, False, 203),
        ("Targeted Lipidomics", 40, True, 204),
    ]
    for label, n_metabolites, include_istd, seed in method_configs:
        prefix = label.lower().replace(" ", "_")
        peak_df, _ = generate_dataset(
            n_metabolites=n_metabolites, include_istd=include_istd, seed=seed,
            out_prefix=f"multimethod_{prefix}", write_meta=False
        )
    print(f"[multimethod] {len(method_configs)} analysis-type datasets generated "
          f"({', '.join(c[0] for c in method_configs)}), "
          f"all sharing {len(sample_names)} samples ({len(GROUPS)} groups + QC).")


generate_multi_method_demo()

print("Demo datasets written.")
