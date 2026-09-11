"""
generate_rich_demo_data.py - A richer, standalone example dataset for MetaboAI Pro
(and for general exploration outside the app), distinct from make_sample_data.py's
simpler 4-group demos.

Produces three files:
  1. sample_peak_area_matrix_richdemo.csv  - 200 metabolites x 42 samples
  2. sample_metadata_richdemo.csv          - 42 samples x clinical/demographic variables
  3. metabolite_row_annotations_richdemo.csv - 200 metabolites x Method/Pathway

Design:
  - 4 Diagnosis groups (Healthy Control, Prediabetic, Type 2 Diabetes, Metabolic
    Syndrome) x 9 biological samples each = 36, + 6 QC replicates = 42 total.
    "Diagnosis" doubles as the "Group" column the app's stats/PCA/ANOVA logic
    reads, so this drops straight into MetaboAI Pro's single-dataset upload with
    no changes needed -- Age/Gender/Treatment/Ethnicity/Body Weight ride along as
    extra columns the app simply passes through.
  - Metabolites carry a Method (one of 4 analysis types) and Pathway annotation in
    a separate row-annotation file, for external pathway-level interpretation,
    grouping, or filtering -- the app doesn't currently ingest a row-annotation
    file, so this is supplementary reference data, not something app.py reads.
  - Abundance values are realistic-ish: a lognormal per-metabolite baseline, small
    per-sample technical noise, a shared per-pathway "activity" factor per sample
    (so metabolites in the same pathway co-vary -- meaningful for correlation
    analysis), diagnosis-driven fold changes on a differential subset (meaningful
    for ANOVA/heatmap/PCA separation), and a continuous Age/Body-Weight-linked
    effect on a small subset (meaningful for correlation against continuous
    covariates). Missingness is injected with the same tiered scheme as the other
    demos, so Data Cleaning & Imputation has something to do.
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 777

DIAGNOSES = ["Healthy Control", "Prediabetic", "Type 2 Diabetes", "Metabolic Syndrome"]
N_PER_DIAGNOSIS = 9
N_QC = 6
N_METABOLITES = 200

METHODS = ["Untargeted Metabolomics", "Targeted Metabolomics", "Untargeted Lipidomics", "Targeted Lipidomics"]
METHOD_WEIGHTS = [0.40, 0.20, 0.25, 0.15]  # sums to 1.0, mirrors the multi-method demo's proportions

PATHWAYS = [
    "Amino Acid Metabolism", "TCA Cycle", "Glycolysis", "Lipid Metabolism",
    "Fatty Acid Oxidation", "Bile Acid Metabolism", "Sphingolipid Metabolism",
    "Purine Metabolism", "Nucleotide Metabolism", "Steroid Hormone Biosynthesis",
]

GENDERS = ["Female", "Male"]
ETHNICITIES = ["Caucasian", "Hispanic/Latino", "African American", "Asian", "Other"]
TREATMENTS_BY_DIAGNOSIS = {
    "Healthy Control": ["Untreated"],
    "Prediabetic": ["Untreated", "Diet & Exercise"],
    "Type 2 Diabetes": ["Metformin", "Insulin", "Diet & Exercise"],
    "Metabolic Syndrome": ["Diet & Exercise", "Metformin"],
}


def _build_sample_metadata(rng):
    sample_names, diagnosis, is_qc, batch = [], [], [], []
    age, gender, treatment, ethnicity, body_weight = [], [], [], [], []

    for d_i, dx in enumerate(DIAGNOSES):
        # Disease groups skew a bit older and heavier than Healthy Control -- realistic,
        # and gives Age/Body Weight a genuine (if modest) relationship to Diagnosis for
        # anyone cross-tabulating rather than just eyeballing group means.
        age_center = 38 if dx == "Healthy Control" else rng.uniform(48, 58)
        weight_center = 68 if dx == "Healthy Control" else rng.uniform(82, 95)
        for i in range(N_PER_DIAGNOSIS):
            sample_names.append(f"{dx.replace(' ', '')}_{i+1}")
            diagnosis.append(dx)
            is_qc.append(False)
            batch.append("1" if i < N_PER_DIAGNOSIS // 2 else "2")
            age.append(int(np.clip(rng.normal(age_center, 8), 20, 85)))
            gender.append(rng.choice(GENDERS))
            treatment.append(rng.choice(TREATMENTS_BY_DIAGNOSIS[dx]))
            ethnicity.append(rng.choice(ETHNICITIES, p=[0.45, 0.20, 0.15, 0.15, 0.05]))
            body_weight.append(round(float(np.clip(rng.normal(weight_center, 12), 45, 140)), 1))

    for i in range(N_QC):
        sample_names.append(f"QC_{i+1}")
        diagnosis.append("QC")
        is_qc.append(True)
        batch.append("1" if i < N_QC // 2 else "2")
        age.append(pd.NA)
        gender.append("QC")
        treatment.append("QC")
        ethnicity.append("QC")
        body_weight.append(pd.NA)

    meta_df = pd.DataFrame({
        "Sample": sample_names,
        "Group": diagnosis,          # app-compatible alias of Diagnosis (drives PCA/ANOVA/stats grouping)
        "IsQC": is_qc,
        "Batch": batch,
        "Diagnosis": diagnosis,
        "Age": age,
        "Gender": gender,
        "Treatment": treatment,
        "Ethnicity": ethnicity,
        "Body Weight": body_weight,
    })
    return meta_df


def _build_row_annotations(rng):
    metabolite_names = [f"Metabolite_{i+1:03d}" for i in range(N_METABOLITES)]
    methods = rng.choice(METHODS, size=N_METABOLITES, p=METHOD_WEIGHTS)
    # Roughly even pathway spread with natural random variation, not a hard-forced quota
    pathways = rng.choice(PATHWAYS, size=N_METABOLITES)
    return pd.DataFrame({"Metabolite": metabolite_names, "Method": methods, "Pathway": pathways})


def generate_rich_demo():
    rng = np.random.default_rng(SEED)
    meta_df = _build_sample_metadata(rng)
    row_annot = _build_row_annotations(rng)
    sample_names = meta_df["Sample"].tolist()
    diagnosis = meta_df["Diagnosis"].tolist()
    is_qc = meta_df["IsQC"].tolist()
    age_lookup = dict(zip(meta_df["Sample"], meta_df["Age"]))
    weight_lookup = dict(zip(meta_df["Sample"], meta_df["Body Weight"]))
    metabolite_names = row_annot["Metabolite"].tolist()
    pathway_of = dict(zip(row_annot["Metabolite"], row_annot["Pathway"]))
    n_samples = len(sample_names)

    # 1. Per-metabolite baseline abundance + per-sample technical noise
    base = rng.lognormal(mean=10, sigma=1.2, size=(N_METABOLITES, 1))
    data = base * rng.lognormal(mean=0, sigma=0.15, size=(N_METABOLITES, n_samples))

    # 2. Shared per-pathway "activity" factor per BIOLOGICAL sample -> co-regulated
    #    metabolites within a pathway (meaningful signal for correlation analysis).
    #    QC replicates are technical replicates of the same pooled material, not
    #    different biological subjects, so they must NOT carry this variation --
    #    applying it to QC would inflate QC CV into unrealistic territory and make
    #    the QC tab's Acceptable/Variable split meaningless.
    bio_idx_all = [i for i, q in enumerate(is_qc) if not q]
    for pw in PATHWAYS:
        pw_feats = [i for i, m in enumerate(metabolite_names) if pathway_of[m] == pw]
        if not pw_feats:
            continue
        pathway_activity = np.ones(n_samples)
        pathway_activity[bio_idx_all] = rng.lognormal(mean=0, sigma=0.22, size=len(bio_idx_all))
        data[np.ix_(pw_feats, range(n_samples))] *= pathway_activity

    # 3. Diagnosis-driven fold changes on a differential subset (~25% of features),
    #    each direction/magnitude independent per non-control diagnosis group --
    #    Diagnosis is nominal (no inherent severity order), unlike the other demos'
    #    Control->Severe staging.
    diag_idx = {dx: [i for i, d in enumerate(diagnosis) if d == dx] for dx in DIAGNOSES}
    n_diff = max(5, int(0.25 * N_METABOLITES))
    diff_feats = rng.choice(N_METABOLITES, size=n_diff, replace=False)
    for f in diff_feats:
        for dx in DIAGNOSES:
            if dx == "Healthy Control":
                continue
            fold = rng.choice([rng.uniform(1.5, 2.8), rng.uniform(0.3, 0.65)])
            data[f, diag_idx[dx]] *= fold

    # 4. Continuous covariate effects on a small subset (~5% of features), linked to
    #    Age and/or Body Weight -- gives correlation analysis against continuous
    #    metadata something genuine to find, not just categorical group differences.
    bio_mask = [not q for q in is_qc]
    ages = np.array([age_lookup[s] if not pd.isna(age_lookup[s]) else 45 for s in sample_names], dtype=float)
    weights = np.array([weight_lookup[s] if not pd.isna(weight_lookup[s]) else 70 for s in sample_names], dtype=float)
    age_z = (ages - ages[bio_mask].mean()) / ages[bio_mask].std()
    weight_z = (weights - weights[bio_mask].mean()) / weights[bio_mask].std()
    n_covariate = max(3, int(0.05 * N_METABOLITES))
    covariate_feats = rng.choice([i for i in range(N_METABOLITES) if i not in diff_feats],
                                  size=n_covariate, replace=False)
    for f in covariate_feats:
        driver = age_z if rng.random() < 0.5 else weight_z
        coef = rng.uniform(0.15, 0.35) * rng.choice([1, -1])
        multiplier = np.ones(n_samples)
        multiplier[bio_idx_all] = np.exp(coef * driver[bio_idx_all])  # QC unaffected -- no real age/weight
        data[f, :] *= multiplier

    # 5. QC replicate stability is already covered by the same low-sigma technical
    #    noise applied to everyone in step 1 -- no ISTD row in this combined panel
    #    (it spans multiple methods, so a single spiked standard wouldn't apply to
    #    all of it; use Group/Diagnosis + Median-IQR normalization when analyzing).

    # 6. Realistic missingness (encoded as exact zero), biological samples only,
    #    same tiered scheme as the other demos so Cleaning & Imputation has real work:
    #    ~70% features 0-15% missing, ~20% features 20-50%, ~10% features 55-85%.
    bio_sample_idx = [i for i, q in enumerate(is_qc) if not q]
    for f in range(N_METABOLITES):
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
    peak_path = f"{BASE_DIR}/sample_peak_area_matrix_richdemo.csv"
    peak_df.to_csv(peak_path, index=False)

    meta_path = f"{BASE_DIR}/sample_metadata_richdemo.csv"
    meta_df.to_csv(meta_path, index=False)

    annot_path = f"{BASE_DIR}/metabolite_row_annotations_richdemo.csv"
    row_annot.to_csv(annot_path, index=False)

    print(f"[richdemo] {N_METABOLITES} metabolites x {n_samples} samples "
          f"({N_QC} QC, {n_samples - N_QC} biological across {len(DIAGNOSES)} diagnosis groups) "
          f"-> {peak_path}")
    print(f"[richdemo] metadata (Diagnosis, Age, Gender, Treatment, Ethnicity, Body Weight) -> {meta_path}")
    print(f"[richdemo] row annotations (Method, Pathway) -> {annot_path}")
    return peak_df, meta_df, row_annot


if __name__ == "__main__":
    generate_rich_demo()
    print("Rich demo dataset written.")
