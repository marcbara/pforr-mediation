"""
06_leave_one_out.py
Leave-one-out sensitivity analysis for:
"How Governance Instruments Shape Project Learning"
Marc Bara, 2026

Drops each PforR project one at a time and re-estimates the baseline
outcome model. Reports the distribution of PforR coefficients to show
that no single project drives the result.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/IEG_ICRR_PPAR_Ratings_2025-12-15.xlsx"

# --- Load & recode (same as 02_regression.py) ---
try:
    df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
except Exception:
    df = pd.read_csv(DATA_PATH.replace(".xlsx", "").replace("IEG_ICRR_PPAR_Ratings_2025-12-15", "ieg_ratings") + ".csv")

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
df["me_num"]      = df["IEG Monitoring and Evaluation Quality Ratings"].map(me_map)
df["vol_ord"]     = df["Project Volume"].map(vol_map)
df["approval_fy"] = pd.to_numeric(df["Approval FY"], errors="coerce")
df["closing_fy"]  = pd.to_numeric(df["Closing FY"], errors="coerce")
df["duration"]    = df["closing_fy"] - df["approval_fy"]
df["fcs"]         = pd.to_numeric(df["Country FCS Status"], errors="coerce")
df["pforr"]       = (df["Lending Instrument"] == "PforR").astype(float)
df["region"]      = df["Region"].astype(str)

# --- Working sample ---
df2 = df[df["Lending Instrument"].isin(["IPF", "PforR"])].copy()
df2 = df2.dropna(subset=["outcome_num", "me_num", "pforr",
                          "approval_fy", "fcs", "vol_ord"])
df2 = df2[(df2["duration"] > 0) & (df2["duration"] < 30)]
df2 = df2.reset_index(drop=True)

FORMULA_CONTROLS = "approval_fy + vol_ord + fcs + C(region)"

# --- Baseline estimate ---
m_base = smf.ols(f"outcome_num ~ pforr + {FORMULA_CONTROLS}", data=df2).fit(cov_type="HC3")
beta_base = m_base.params["pforr"]

print(f"Working sample: n = {len(df2)}")
print(f"PforR projects: {int(df2['pforr'].sum())}")
print(f"Baseline PforR beta (outcome): {beta_base:+.4f}\n")

# --- Leave-one-out: drop each PforR project ---
pforr_idx = df2[df2["pforr"] == 1].index.tolist()
betas = []

print("Running leave-one-out analysis...")
for i, drop_idx in enumerate(pforr_idx):
    df_loo = df2.drop(index=drop_idx)
    m_loo = smf.ols(f"outcome_num ~ pforr + {FORMULA_CONTROLS}", data=df_loo).fit(cov_type="HC3")
    betas.append(m_loo.params["pforr"])
    if (i + 1) % 20 == 0:
        print(f"  ... {i + 1}/{len(pforr_idx)} done")

betas = np.array(betas)

print(f"\n=== Leave-One-Out Results (n = {len(betas)} iterations) ===")
print(f"  Baseline beta:     {beta_base:+.4f}")
print(f"  LOO mean beta:     {betas.mean():+.4f}")
print(f"  LOO sd beta:       {betas.std():.4f}")
print(f"  LOO min beta:      {betas.min():+.4f}")
print(f"  LOO max beta:      {betas.max():+.4f}")
print(f"  LOO range:      {betas.max() - betas.min():.4f}")
print(f"  Max deviation:  {np.max(np.abs(betas - beta_base)):.4f}")

# Check: how many LOO estimates remain significant at p < 0.01?
sig_count = 0
for drop_idx in pforr_idx:
    df_loo = df2.drop(index=drop_idx)
    m_loo = smf.ols(f"outcome_num ~ pforr + {FORMULA_CONTROLS}", data=df_loo).fit(cov_type="HC3")
    if m_loo.pvalues["pforr"] < 0.01:
        sig_count += 1

print(f"\n  LOO estimates significant at p < 0.01: {sig_count}/{len(pforr_idx)} ({100*sig_count/len(pforr_idx):.0f}%)")
print(f"  -> No single PforR project drives the result." if sig_count == len(pforr_idx)
      else f"  -> {len(pforr_idx) - sig_count} iteration(s) lose significance when one project is dropped.")
