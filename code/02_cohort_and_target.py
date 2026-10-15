from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "preprocessed_nhanes.csv"
OUTPUT_FILE = ROOT / "data" / "processed_cohort.csv"

df = pd.read_csv(INPUT_FILE)
n_raw = len(df)
print(f"Raw rows after preprocessing: {n_raw}")

# KDIGO-based target: CKD if eGFR < 60 or urine albumin-creatinine ratio >= 30.
e = df["egfr"]
a = df["albumin_creatinine_ratio"]
df["ckd_kdigo"] = np.where(
    (e < 60) | (a >= 30), 1,
    np.where((e >= 60) & (a < 30), 0, np.nan)
)

# Adult cohort only (age >= 20).
adults = df[df["age"] >= 20].copy()
n_children = n_raw - len(adults)
undetermined = adults["ckd_kdigo"].isna()
n_undetermined = int(undetermined.sum())
df_clean = adults[~undetermined].reset_index(drop=True)

print("\n=== PARTICIPANT FLOW ===")
print(f"Raw rows: {n_raw}")
print(f"Excluded age <20: {n_children}")
print(f"Adults with indeterminate target: {n_undetermined}")
print(f"Final adult cohort: {len(df_clean)}")
print(f"CKD positive: {int(df_clean['ckd_kdigo'].sum())}")
print(f"CKD prevalence: {df_clean['ckd_kdigo'].mean():.4f}")

# Compare newly defined target against any pre-existing label, if available.
if "ckd_stage" in df_clean.columns:
    print("\n=== KDIGO TARGET BY EXISTING CKD STAGE ===")
    print(pd.crosstab(df_clean["ckd_stage"], df_clean["ckd_kdigo"], dropna=False).to_string())
if "ckd_present" in df_clean.columns:
    print("\n=== KDIGO TARGET BY EXISTING CKD PRESENT LABEL ===")
    print(pd.crosstab(df_clean["ckd_present"], df_clean["ckd_kdigo"], dropna=False).to_string())

# Keep gender consistently encoded where possible.
if "gender" in df_clean.columns and df_clean["gender"].dtype == object:
    df_clean["gender"] = df_clean["gender"].map({
        "Female": 0, "Male": 1,
        "female": 0, "male": 1,
        "F": 0, "M": 1,
    })

# X/y are constructed here for diagnostic checks; the modeling script reconstructs them from CSV.
y = df_clean["ckd_kdigo"].astype(int)
X = df_clean.drop(
    columns=["participant_id", "ckd_stage", "ckd_present", "ckd_kdigo"],
    errors="ignore"
)

print("\n=== MISSINGNESS (TOP 20 FEATURES) ===")
print(X.isna().mean().sort_values(ascending=False).head(20).round(4).to_string())

# Univariate AUROC scan: diagnostic markers may be nearly deterministic because they define the target.
print("\n=== SINGLE-FEATURE AUROC SCAN (diagnostic leakage check) ===")
scan_rows = []
for col in X.columns:
    values = X[col]
    valid = values.notna()
    if valid.sum() < 20 or values[valid].nunique() < 2:
        continue
    try:
        auc = roc_auc_score(y[valid], values[valid])
        auc_flipped = roc_auc_score(y[valid], -values[valid])
        scan_rows.append({"feature": col, "n": int(valid.sum()), "AUROC_directional": max(auc, auc_flipped)})
    except (ValueError, TypeError):
        continue
if scan_rows:
    print(pd.DataFrame(scan_rows).sort_values("AUROC_directional", ascending=False).head(20).round(4).to_string(index=False))
else:
    print("No eligible numeric features found for the AUROC scan.")

# Sanity check: recompute eGFR from the existing eGFR column is not possible without its source formula;
# report target-defining marker distributions instead.
print("\n=== TARGET-DEFINING MARKER CHECK ===")
print(df_clean[["egfr", "albumin_creatinine_ratio", "ckd_kdigo"]].describe().round(3).to_string())

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"\nFinal cohort saved to: {OUTPUT_FILE}")
