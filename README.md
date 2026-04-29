# Disbursement-Linked Lending and Development Outcomes: Replication Code

**Marc Bara** · ESADE Business School · marcoantonio.bara@esade.edu

This repository contains the full analysis pipeline for a mediation analysis of the World Bank's Program-for-Results (PforR) instrument versus Investment Project Finance (IPF), using the Independent Evaluation Group (IEG) project ratings dataset. The repository is intended to support reproducibility of the empirical results; the manuscript itself is not hosted here.

---

## Research questions

- **RQ1:** Does disbursement-linked instrument design (PforR vs. IPF) predict M&E quality?
- **RQ2:** Does disbursement-linked instrument design predict project outcome ratings?
- **RQ3:** To what extent is the instrument-outcome relationship mediated by M&E quality?

---

## Headline results

Using the IEG project ratings dataset (n = 6,089 completed World Bank projects, 1995-2020):

| Model | Dependent variable | PforR beta | SE | p |
|---|---|---|---|---|
| 1 | M&E Quality (1-4 scale) | +0.227 | 0.045 | <0.001 |
| 2 | Outcome rating (1-6, total effect) | +0.310 | 0.076 | <0.001 |
| 3 | Outcome rating (1-6, direct effect) | +0.119 | 0.062 | 0.055 |

All models include region fixed effects and HC3 robust standard errors.

**61.5% of the PforR outcome premium is mediated through M&E quality.**
Indirect effect = 0.191, 95% bootstrap CI [0.114, 0.267], Sobel z = 5.01, p < 0.0001.

### Robustness checks

| Specification | PforR beta | p |
|---|---|---|
| Baseline (region FE) | +0.310 | <0.001 |
| + Sector FE | +0.323 | <0.001 |
| Country FE | +0.242 | 0.003 |
| Country FE + Sector FE | +0.257 | 0.002 |
| M&E equation, Country FE | +0.206 | <0.001 |
| Propensity score matched (n=240) | +0.287 | 0.011 |

Placebo test (500 random draws of n=120 from pre-2012 IPF pool): actual beta = 0.310 exceeds 100% of placebo distribution (empirical p = 0.000).

Leave-one-out sensitivity (drop each PforR project in turn): PforR coefficient ranges from +0.299 to +0.326 across 120 iterations. No single project drives the result.

On the propensity-score-matched sample, the share mediated through M&E quality reaches 100% (full mediation), with the direct effect of PforR vanishing once observable selection is addressed.

---

## Repository structure

```
analysis/
  01_exploration.py          : descriptive statistics, variable distributions, trends
  02_regression.py           : OLS models + product-of-coefficients mediation analysis
  03_robustness.py           : country/sector FE specifications + placebo test (500 iter)
  04_bootstrap_mediation.py  : 2,000-iteration bootstrap CIs + Sobel test
  05_psm_robustness.py       : propensity score matching (1:1 nearest-neighbor)
  06_leave_one_out.py        : leave-one-out sensitivity analysis
data/
  .gitkeep                   : folder tracked; data file gitignored (download below)
```

---

## Reproduce the results

**1. Download the data**

```
https://ieg.worldbankgroup.org/sites/default/files/Data/IEG_ICRR_PPAR_Ratings_2025-12-15.xlsx
```

Save as `data/IEG_ICRR_PPAR_Ratings_2025-12-15.xlsx`.

**2. Install dependencies**

```bash
pip install pandas numpy scipy statsmodels openpyxl
```

**3. Run**

```bash
python analysis/01_exploration.py          # descriptive stats
python analysis/02_regression.py           # main results + mediation
python analysis/03_robustness.py           # country/sector FE + placebo test
python analysis/04_bootstrap_mediation.py  # bootstrap CIs + Sobel test
python analysis/05_psm_robustness.py       # propensity score matching (1:1 NN)
python analysis/06_leave_one_out.py        # leave-one-out sensitivity
```

No other dependencies. All scripts are self-contained and print results to stdout.

---

## Theoretical positioning

The study identifies M&E quality as the mechanism connecting disbursement-linked instrument design to project performance, contributing to the development effectiveness literature on results-based financing:

- **Witter et al. (2012, Cochrane):** mixed evidence on performance-based financing in health — this study tests one channel (measurement-infrastructure investment) the PBF literature has not systematically isolated.
- **Renmans et al. (2016, Health Policy and Planning):** results-based financing as a "black box" in need of mechanism disaggregation — this study provides a quantitative mediation decomposition.
- **Clist (2016, World Bank Research Observer):** payment-by-results in development aid as appealing but weakly evidenced — this study supplies large-sample evidence for one specific RBF instrument.
- **Heinzel & Liese (2021, Review of International Organizations):** supervision quality predicts outcomes in IPF — this study extends the comparison to PforR and models mediation explicitly.
- **Mumssen, Johannes, & Kumar (2010, World Bank):** output-based aid lessons-learned — this study formalizes the comparative test those experiences invite.

---

## Contact

For questions about the code or to request the working paper, contact Marc Bara at marcoantonio.bara@esade.edu.
