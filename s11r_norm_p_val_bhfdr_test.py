import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

df = pd.read_csv("outputs/s11r_norm_bhfdr/s11_esg_sector_comparison_norm_relevant_metrics.csv")

# first for t-test
pcol = "T-Test P-Value"

mask = df[pcol].notna()
pvals = df.loc[mask, pcol].astype(float).values

# Benjamini–Hochberg FDR at q=0.05
rej, qvals, alphacSidak, alphacBonf = multipletests(pvals, alpha=0.05, method="fdr_bh")

df.loc[mask, "FDR_qvalue_BH"] = qvals
df.loc[mask, "FDR_significant_0p05"] = rej

# now for MWU test
mwu_col = "MWU P-Value"

mwu_mask = df[mwu_col].notna()
mwu_pvals = df.loc[mwu_mask, mwu_col].astype(float).values

# Benjamini–Hochberg FDR at q=0.05 for MWU
mwu_rej, mwu_qvals, alphacSidakM, alphacBonfM = multipletests(mwu_pvals, alpha=0.05, method="fdr_bh")

df.loc[mwu_mask, "MWU_FDR_qvalue_BH"] = mwu_qvals
df.loc[mwu_mask, "MWU_FDR_significant_0p05"] = mwu_rej

df.to_csv("outputs/s11r_norm_bhfdr/s11_esg_sector_comparison_norm_relevant_metrics_with_FDR.csv", index=False)

# Quick summary
print("Total tests:", len(pvals))
print("Significant at raw p<0.05:", int((pvals < 0.05).sum()))
print("Significant after FDR (BH) q<0.05:", int(rej.sum()))

print("Total tests:", len(mwu_pvals))
print("Significant at raw p<0.05:", int((mwu_pvals < 0.05).sum()))
print("Significant after FDR (BH) q<0.05:", int(mwu_rej.sum()))