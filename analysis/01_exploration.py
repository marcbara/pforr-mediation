"""
01_exploration.py
Descriptive statistics and variable inspection for:
"Governance Instruments as Learning Architecture"
Marc Bara, 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/IEG_ICRR_PPAR_Ratings_2025-12-15.xlsx"

# --- Load ---
try:
    df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
except Exception:
    df = pd.read_csv(DATA_PATH.replace(".xlsx", "").replace("IEG_ICRR_PPAR_Ratings_2025-12-15", "ieg_ratings") + ".csv")
print(f"Raw dataset: {df.shape[0]} rows x {df.shape[1]} columns\n")

# --- Recode ---
outcome_map = {
    "Highly Satisfactory": 6, "Satisfactory": 5, "Moderately Satisfactory": 4,
    "Moderately Unsatisfactory": 3, "Unsatisfactory": 2, "Highly Unsatisfactory": 1,
    "Not Rated": np.nan,
}
me_map = {
    "High": 4, "Substantial": 3, "Modest": 2, "Negligible": 1,
    "Not Rated": np.nan, "Non-Evaluable": np.nan,
}
vol_map = {
    "Less than 10 million": 1, ">=10 million & <25 million": 2,
    ">=25 million & <50 million": 3, ">=50 million & <100 million": 4,
    ">=100 million": 5,
}

df["outcome_num"] = df["IEG Outcome Ratings"].map(outcome_map)
df["me_num"] = df["IEG Monitoring and Evaluation Quality Ratings"].map(me_map)
df["vol_ord"] = df["Project Volume"].map(vol_map)
df["approval_fy"] = pd.to_numeric(df["Approval FY"], errors="coerce")
df["closing_fy"] = pd.to_numeric(df["Closing FY"], errors="coerce")
df["duration"] = df["closing_fy"] - df["approval_fy"]
df["fcs"] = pd.to_numeric(df["Country FCS Status"], errors="coerce")
df["pforr"] = (df["Lending Instrument"] == "PforR").astype(float)

df2 = df[
    df["outcome_num"].notna() & df["me_num"].notna()
].copy()

print("=== Lending Instrument breakdown ===")
print(df["Lending Instrument"].value_counts(), "\n")

print("=== Global Practice (top 10) ===")
print(df["Global Practice"].value_counts().head(10), "\n")

print("=== IEG Outcome Ratings distribution ===")
print(df["IEG Outcome Ratings"].value_counts(), "\n")

print("=== IEG M&E Quality Ratings distribution ===")
print(df["IEG Monitoring and Evaluation Quality Ratings"].value_counts(), "\n")

print("=== M&E Quality -> Outcome (raw means) ===")
print(
    df2.groupby("IEG Monitoring and Evaluation Quality Ratings")["outcome_num"]
    .agg(["mean", "count"])
    .reindex(["High", "Substantial", "Modest", "Negligible"]),
    "\n",
)

print("=== Spearman correlation: M&E Quality <-> Outcome ===")
r, p = stats.spearmanr(df2["me_num"], df2["outcome_num"])
print(f"r = {r:.3f}, n = {len(df2)}, p < 0.001\n")

print("=== High vs Negligible M&E - effect size ===")
high = df2[df2["me_num"] == 4]["outcome_num"]
neg  = df2[df2["me_num"] == 1]["outcome_num"]
d = (high.mean() - neg.mean()) / np.sqrt((high.std()**2 + neg.std()**2) / 2)
print(f"High M&E mean: {high.mean():.3f} (n={len(high)})")
print(f"Negligible M&E mean: {neg.mean():.3f} (n={len(neg)})")
print(f"Cohen's d = {d:.3f}\n")

print("=== M&E Quality trend by 5-year cohort ===")
df2["fy_bin"] = pd.cut(df2["approval_fy"], bins=range(1990, 2025, 5), right=False)
print(df2.groupby("fy_bin", observed=False)["me_num"].agg(["mean", "count"]), "\n")

print("=== PforR vs IPF - raw comparison ===")
for label, mask in [("PforR", df2["pforr"] == 1), ("IPF", df2["Lending Instrument"] == "IPF")]:
    s = df2[mask]
    print(f"{label} (n={len(s)}): outcome={s['outcome_num'].mean():.3f}, M&E={s['me_num'].mean():.3f}")
