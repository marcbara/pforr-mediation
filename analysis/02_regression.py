"""
02_regression.py
OLS regression models and mediation analysis for:
"Governance Instruments as Learning Architecture"
Marc Bara, 2026

Models
------
Model 1 : M&E Quality ~ PforR + controls          (a-path)
Model 2 : Outcome     ~ PforR + controls          (total effect)
Model 3 : Outcome     ~ M&E + PforR + controls    (direct effect + b-path)

Mediation (product-of-coefficients):
  indirect = a * b
  % mediated = indirect / total * 100
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/IEG_ICRR_PPAR_Ratings_2025-12-15.xlsx"

# --- Load & recode ---
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

# --- Working sample: IPF + PforR only, valid controls ---
df2 = df[df["Lending Instrument"].isin(["IPF", "PforR"])].copy()
df2 = df2.dropna(subset=["outcome_num", "me_num", "pforr",
                          "approval_fy", "fcs", "vol_ord"])
df2 = df2[(df2["duration"] > 0) & (df2["duration"] < 30)]

print(f"Working sample: n = {len(df2)}")
print(f"  PforR: {int(df2['pforr'].sum())}")
print(f"  IPF:   {int((df2['pforr'] == 0).sum())}\n")

FORMULA_CONTROLS = "approval_fy + vol_ord + fcs + C(region)"

# --- Model 1: PforR → M&E Quality ---
m1 = smf.ols(f"me_num ~ pforr + {FORMULA_CONTROLS}", data=df2).fit(cov_type="HC3")

# --- Model 2: PforR → Outcome (total effect) ---
m2 = smf.ols(f"outcome_num ~ pforr + {FORMULA_CONTROLS}", data=df2).fit(cov_type="HC3")

# --- Model 3: M&E + PforR → Outcome ---
m3 = smf.ols(f"outcome_num ~ me_num + pforr + {FORMULA_CONTROLS}", data=df2).fit(cov_type="HC3")


def report(label, model, key_vars):
    print(f"=== {label} ===")
    for v in key_vars:
        b  = model.params[v]
        se = model.bse[v]
        p  = model.pvalues[v]
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {v:<15} beta = {b:+.4f}  SE = {se:.4f}  p = {p:.4f} {stars}")
    print(f"  R2 = {model.rsquared:.4f}  adj-R2 = {model.rsquared_adj:.4f}  n = {int(model.nobs)}\n")


report("Model 1 - DV: M&E Quality",     m1, ["pforr", "approval_fy", "fcs"])
report("Model 2 - DV: Outcome (total)",  m2, ["pforr", "approval_fy", "fcs"])
report("Model 3 - DV: Outcome (direct)", m3, ["me_num", "pforr", "approval_fy"])

# --- Mediation ---
a        = m1.params["pforr"]       # PforR → M&E
b        = m3.params["me_num"]      # M&E   → Outcome (controlling PforR)
total    = m2.params["pforr"]       # PforR → Outcome (total)
indirect = a * b
direct   = m3.params["pforr"]
pct      = indirect / total * 100

print("=== Mediation (product-of-coefficients) ===")
print(f"  Total effect  (PforR -> Outcome):        {total:+.4f}")
print(f"  a-path        (PforR -> M&E):            {a:+.4f}")
print(f"  b-path        (M&E -> Outcome | PforR):  {b:+.4f}")
print(f"  Indirect      (a * b):                   {indirect:+.4f}")
print(f"  Direct        (PforR net of M&E):        {direct:+.4f}")
print(f"  % mediated through M&E:                 {pct:.1f}%\n")
