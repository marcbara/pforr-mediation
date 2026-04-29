"""
05_psm_robustness.py
Propensity score matching robustness check for:
"How Governance Instruments Shape Project Learning"
Marc Bara, 2026

Estimates a propensity score for PforR adoption, matches each PforR
project to the nearest IPF project (without replacement), and re-runs
all three models on the matched sample.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
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
df["sector"]      = df["Global Practice"].astype(str)

# --- Working sample ---
df2 = df[df["Lending Instrument"].isin(["IPF", "PforR"])].copy()
df2 = df2.dropna(subset=["outcome_num", "me_num", "pforr",
                          "approval_fy", "fcs", "vol_ord"])
df2 = df2[(df2["duration"] > 0) & (df2["duration"] < 30)]
df2 = df2.reset_index(drop=True)

print(f"Working sample: n = {len(df2)}")
print(f"  PforR: {int(df2['pforr'].sum())}")
print(f"  IPF:   {int((df2['pforr'] == 0).sum())}\n")

# ===================================================================
# STEP 1: Estimate propensity scores
# ===================================================================

# Create dummies for region and sector
region_dummies = pd.get_dummies(df2["region"], prefix="reg", drop_first=True)
sector_dummies = pd.get_dummies(df2["sector"], prefix="sec", drop_first=True)

X = pd.concat([
    df2[["approval_fy", "vol_ord", "fcs"]],
    region_dummies,
    sector_dummies,
], axis=1).astype(float)

y = df2["pforr"].values

# Standardize continuous variables for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit logistic regression
logit = LogisticRegression(max_iter=5000, solver="lbfgs")
logit.fit(X_scaled, y)

df2["pscore"] = logit.predict_proba(X_scaled)[:, 1]

print("=== Propensity Score Distribution ===")
for label, mask in [("PforR", df2["pforr"] == 1), ("IPF", df2["pforr"] == 0)]:
    ps = df2.loc[mask, "pscore"]
    print(f"  {label}: mean={ps.mean():.4f}, sd={ps.std():.4f}, "
          f"min={ps.min():.4f}, max={ps.max():.4f}")
print()

# ===================================================================
# STEP 2: Nearest-neighbor matching (1:1, without replacement)
# ===================================================================

treated = df2[df2["pforr"] == 1].copy()
control = df2[df2["pforr"] == 0].copy()

# Compute pairwise distances on propensity score
dist_matrix = cdist(
    treated["pscore"].values.reshape(-1, 1),
    control["pscore"].values.reshape(-1, 1),
    metric="euclidean"
)

matched_control_idx = []
available = set(range(len(control)))

for i in range(len(treated)):
    # Find nearest available control
    dists = dist_matrix[i]
    sorted_idx = np.argsort(dists)
    for j in sorted_idx:
        if j in available:
            matched_control_idx.append(j)
            available.remove(j)
            break

matched_control = control.iloc[matched_control_idx]
matched = pd.concat([treated, matched_control], ignore_index=True)

print(f"=== Matched Sample ===")
print(f"  PforR: {int(matched['pforr'].sum())}")
print(f"  IPF:   {int((matched['pforr'] == 0).sum())}")
print(f"  Total: {len(matched)}\n")

# Check balance
print("=== Covariate Balance (matched sample) ===")
for var in ["approval_fy", "vol_ord", "fcs", "pscore"]:
    t_mean = matched.loc[matched["pforr"] == 1, var].mean()
    c_mean = matched.loc[matched["pforr"] == 0, var].mean()
    t_std  = matched.loc[matched["pforr"] == 1, var].std()
    c_std  = matched.loc[matched["pforr"] == 0, var].std()
    smd = (t_mean - c_mean) / np.sqrt((t_std**2 + c_std**2) / 2)
    print(f"  {var:<15} PforR={t_mean:.3f}  IPF={c_mean:.3f}  SMD={smd:+.3f}")
print()

# ===================================================================
# STEP 3: Re-run models on matched sample
# ===================================================================

FORMULA_CONTROLS = "approval_fy + vol_ord + fcs + C(region)"

m1_psm = smf.ols(f"me_num ~ pforr + {FORMULA_CONTROLS}", data=matched).fit(cov_type="HC3")
m2_psm = smf.ols(f"outcome_num ~ pforr + {FORMULA_CONTROLS}", data=matched).fit(cov_type="HC3")
m3_psm = smf.ols(f"outcome_num ~ me_num + pforr + {FORMULA_CONTROLS}", data=matched).fit(cov_type="HC3")


def report(label, model, key_vars):
    print(f"=== {label} ===")
    for v in key_vars:
        b  = model.params[v]
        se = model.bse[v]
        p  = model.pvalues[v]
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {v:<15} beta = {b:+.4f}  SE = {se:.4f}  p = {p:.4f} {stars}")
    print(f"  R2 = {model.rsquared:.4f}  adj-R2 = {model.rsquared_adj:.4f}  n = {int(model.nobs)}\n")


report("PSM Model 1 - DV: M&E Quality",     m1_psm, ["pforr"])
report("PSM Model 2 - DV: Outcome (total)",  m2_psm, ["pforr"])
report("PSM Model 3 - DV: Outcome (direct)", m3_psm, ["me_num", "pforr"])

# --- Mediation on matched sample ---
a_psm     = m1_psm.params["pforr"]
b_psm     = m3_psm.params["me_num"]
total_psm = m2_psm.params["pforr"]
indirect_psm = a_psm * b_psm
direct_psm   = m3_psm.params["pforr"]
pct_psm      = indirect_psm / total_psm * 100 if total_psm != 0 else float("nan")

print("=== Mediation on Matched Sample ===")
print(f"  Total effect:   {total_psm:+.4f}")
print(f"  a-path:         {a_psm:+.4f}")
print(f"  b-path:         {b_psm:+.4f}")
print(f"  Indirect:       {indirect_psm:+.4f}")
print(f"  Direct:         {direct_psm:+.4f}")
print(f"  % mediated:     {pct_psm:.1f}%\n")

# --- Compare to unmatched ---
print("=== Comparison: Unmatched vs PSM-Matched ===")
m2_full = smf.ols(f"outcome_num ~ pforr + {FORMULA_CONTROLS}", data=df2).fit(cov_type="HC3")
print(f"  Unmatched PforR beta (outcome): {m2_full.params['pforr']:+.4f}  "
      f"(SE={m2_full.bse['pforr']:.4f}, p={m2_full.pvalues['pforr']:.4f})")
print(f"  PSM       PforR beta (outcome): {m2_psm.params['pforr']:+.4f}  "
      f"(SE={m2_psm.bse['pforr']:.4f}, p={m2_psm.pvalues['pforr']:.4f})")
