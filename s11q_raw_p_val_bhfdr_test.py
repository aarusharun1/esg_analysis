import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

# load "ALL_data" table that contains T-Test P-Value
df = pd.read_csv("outputs/s11_esg_sector_comparison_relevant_metrics.csv")

pcol = "T-Test P-Value"

mask = df[pcol].notna()
pvals = df.loc[mask, pcol].astype(float).values

# Benjamini–Hochberg FDR where q=0.05
rej, qvals, alphacSidak, alphacBonf = multipletests(pvals, alpha=0.05, method="fdr_bh")

df.loc[mask, "FDR_qvalue_BH"] = qvals
df.loc[mask, "FDR_significant_0p05"] = rej

df.to_csv("outputs/s11q_BHFDR_test/s11_esg_sector_comparison_relevant_metrics_with_FDR.csv", index=False)

print("Total tests:", len(pvals))
print("Significant at raw p<0.05:", int((pvals < 0.05).sum()))
print("Significant after FDR (BH) q<0.05:", int(rej.sum()))

