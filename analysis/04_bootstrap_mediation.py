"""
04_bootstrap_mediation.py
Bootstrap mediation confidence intervals + Sobel test.
Marc Bara, 2026

Uses numpy lstsq (fast) with region FE dummies.
Equivalent to full OLS with C(region) in each iteration.
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/IEG_ICRR_PPAR_Ratings_2025-12-15.xlsx"

# --- Load & recode (mirrors 02_regression.py) ---
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

df2 = df[df["Lending Instrument"].isin(["IPF", "PforR"])].copy()
df2 = df2.dropna(subset=["outcome_num", "me_num", "pforr",
                          "approval_fy", "fcs", "vol_ord"])
df2 = df2[(df2["duration"] > 0) & (df2["duration"] < 30)].reset_index(drop=True)

print(f"Working sample: n={len(df2)}, PforR={int(df2['pforr'].sum())}")

# --- Point estimates (statsmodels, HC3 SEs for Sobel) ---
m_me  = smf.ols("me_num ~ pforr + approval_fy + vol_ord + fcs + C(region)", data=df2).fit(cov_type="HC3")
m_out = smf.ols("outcome_num ~ me_num + pforr + approval_fy + vol_ord + fcs + C(region)", data=df2).fit(cov_type="HC3")

a_pt  = m_me.params["pforr"]
b_pt  = m_out.params["me_num"]
d_pt  = m_out.params["pforr"]
ind_pt  = a_pt * b_pt
tot_pt  = ind_pt + d_pt
pct_pt  = ind_pt / tot_pt * 100

print(f"\nPoint estimates:")
print(f"  a (PforR->M&E):    {a_pt:.4f}")
print(f"  b (M&E->Outcome):  {b_pt:.4f}")
print(f"  Indirect:          {ind_pt:.4f}")
print(f"  Direct:            {d_pt:.4f}")
print(f"  Total:             {tot_pt:.4f}")
print(f"  % mediated:        {pct_pt:.1f}%")

# --- Sobel test ---
se_a     = m_me.bse["pforr"]
se_b     = m_out.bse["me_num"]
sobel_se = np.sqrt(b_pt**2 * se_a**2 + a_pt**2 * se_b**2)
sobel_z  = ind_pt / sobel_se
sobel_p  = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
print(f"\nSobel test:  z={sobel_z:.3f},  SE={sobel_se:.4f},  p={sobel_p:.6f}")

# --- Bootstrap (numpy lstsq — fast) ---
region_dummies = pd.get_dummies(df2["region"], drop_first=True).values.astype(float)
X = np.column_stack([
    np.ones(len(df2)),
    df2["pforr"].values,
    df2["approval_fy"].values,
    df2["vol_ord"].values,
    df2["fcs"].values,
    region_dummies
])
Y_out = df2["outcome_num"].values.astype(float)
Y_me  = df2["me_num"].values.astype(float)
PFORR_COL = 1

def mediation_fast(X, Yme, Yout):
    coef_me,  _, _, _ = np.linalg.lstsq(X, Yme,  rcond=None)
    Xout = np.column_stack([X, Yme])
    coef_out, _, _, _ = np.linalg.lstsq(Xout, Yout, rcond=None)
    return coef_me[PFORR_COL], coef_out[-1], coef_out[PFORR_COL]  # a, b, direct

np.random.seed(42)
N_BOOT = 2000
n = len(df2)
boot_ind  = np.full(N_BOOT, np.nan)
boot_dir  = np.full(N_BOOT, np.nan)

for i in range(N_BOOT):
    idx = np.random.randint(0, n, n)
    try:
        a, b, d = mediation_fast(X[idx], Y_me[idx], Y_out[idx])
        boot_ind[i] = a * b
        boot_dir[i] = d
    except Exception:
        pass

valid = ~np.isnan(boot_ind)
print(f"\nValid bootstrap iterations: {valid.sum()} / {N_BOOT}")

boot_tot = boot_ind[valid] + boot_dir[valid]
boot_pct = boot_ind[valid] / boot_tot * 100

def ci(arr):
    return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

ci_ind = ci(boot_ind[valid])
ci_dir = ci(boot_dir[valid])
ci_tot = ci(boot_tot)
ci_pct = ci(boot_pct)

print(f"\n95% Bootstrap CIs (percentile method, {valid.sum()} iterations):")
print(f"  Indirect effect:   {ind_pt:.3f}  95% CI [{ci_ind[0]:.3f}, {ci_ind[1]:.3f}]")
print(f"  Direct effect:     {d_pt:.3f}  95% CI [{ci_dir[0]:.3f}, {ci_dir[1]:.3f}]")
print(f"  Total effect:      {tot_pt:.3f}  95% CI [{ci_tot[0]:.3f}, {ci_tot[1]:.3f}]")
print(f"  Percent mediated:  {pct_pt:.1f}%  95% CI [{ci_pct[0]:.1f}%, {ci_pct[1]:.1f}%]")
print("\nUse these numbers to update the mediation table in main.tex")
