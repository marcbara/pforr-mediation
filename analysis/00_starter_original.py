import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load the exact December 2025 file (drop it in the same folder)
df = pd.read_excel('IEG_ICRR_PPAR_Ratings_2025-12-15.xlsx')

# Quick column overview (run once to see names)
print(df.columns.tolist())

# Filter to infrastructure sectors (adjust column name if needed)
infra = df[df['Sector'].astype(str).str.contains('Transport|Energy|Water|Urban|Infrastructure', case=False, na=False)].copy()

# Create AI-governance proxy (post-2020 = likely AI tools)
infra['AI_proxy'] = (infra['ApprovalFY'] >= 2021).astype(int)

# Simple dependent variable: Outcome rating (higher = better acceptability/learning proxy)
infra['Outcome'] = pd.to_numeric(infra['Outcome'], errors='coerce')

# Basic regression
X = sm.add_constant(infra[['AI_proxy', 'Commitment_USD_M', 'Region']])  # add more controls as needed
y = infra['Outcome']
model = sm.OLS(y, X, missing='drop').fit()
print(model.summary())

# Quick plot
sns.boxplot(data=infra, x='AI_proxy', y='Outcome')
plt.title('Outcome Ratings: Pre- vs Post-AI Proxy')
plt.savefig('quick_results_plot.png')
print('Script finished - plot saved!')