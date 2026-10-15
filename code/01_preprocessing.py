from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "CKD_NHANES_2021_2023_encoded.xlsx"
OUTPUT_FILE = ROOT / "data" / "preprocessed_nhanes.csv"

# Load the encoded NHANES workbook from the repository's data/ folder.
df = pd.read_excel(INPUT_FILE)
n_raw = len(df)
print(f"Raw rows: {n_raw}")

# NHANES education code 9 means refused/don't know; treat it as missing.
if "education_level" in df.columns:
    df["education_level"] = df["education_level"].replace({9: np.nan})

# Standardize gender labels to the encoding used by the analysis.
if "gender" in df.columns and df["gender"].dtype == object:
    df["gender"] = df["gender"].map({
        "Female": 0, "Male": 1,
        "female": 0, "male": 1,
        "F": 0, "M": 1,
    })

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Preprocessed data saved to: {OUTPUT_FILE}")
