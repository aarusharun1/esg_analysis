import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

analysis_flag = 5  # Change to 2, 3, 4, 5 as needed

main_file = "processed_data/out19_all_companies_rank_fin.csv"
output_dir = "outputs/s11e_visualizations"
os.makedirs(output_dir, exist_ok=True)

print(os.getcwd())

# main company-level data
df = pd.read_csv(main_file)

# mark ESG ranked
df["ESG_Ranked"] = df["ESG Rank"].notna().astype(int)


def analysis_2_violin_grouped_by_sector_normalized():
    sig_df = pd.read_csv("outputs/s11_sector_comparison_normalized.csv")
    for sector in df["GICS Sector"].dropna().unique():
        sector_df = df[df["GICS Sector"] == sector].copy()
        sig_metrics = sig_df[
            (sig_df["Sector"] == sector) & (sig_df["T-Test Significant"] == True)
        ]["Metric"].unique()

        if len(sig_metrics) == 0:
            continue

        norm_data = []

        for metric in sig_metrics:
            if metric not in sector_df.columns:
                continue

            metric_data = sector_df[[metric, "ESG_Ranked"]].dropna()
            if len(metric_data) < 10:
                continue

            metric_mean = sector_df[metric].mean()
            metric_std = sector_df[metric].std()
            if metric_std == 0 or np.isnan(metric_std):
                continue

            temp = sector_df[[metric, "ESG_Ranked"]].dropna().copy()
            temp["Metric"] = metric
            temp["Z-Score"] = (temp[metric] - metric_mean) / metric_std
            #temp["Z-Score"] = temp["Z-Score"].clip(lower=-3, upper=3)

            norm_data.append(temp[["Z-Score", "ESG_Ranked", "Metric"]])

        if not norm_data:
            continue

        combined = pd.concat(norm_data, ignore_index=True)
        combined["ESG_Ranked"] = combined["ESG_Ranked"].map({1: "Ranked", 0: "Unranked"})

        plt.figure(figsize=(12, 6))
        sns.violinplot(data=combined, x="Metric", y="Z-Score", hue="ESG_Ranked",
                       palette="Set2", split=True, cut=0)
        plt.title(f"Sector: {sector} - Normalized Metric Comparison")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(-5, 5) 
        plt.tight_layout()

        sector_safe = sector.replace("/", "_").replace(" ", "_")
        plt.savefig(f"{output_dir}/violin_grouped_{sector_safe}.png")
        plt.close()


def analysis_3_heatmap_percent_diff():
    rename_dict = {
        "TotalRevenue": "Total Revenue",
        "GrossProfit": "Gross Profit",
        "NetIncome": "Net Income",
        "OperatingIncome": "Operating Income",
        "Gross Margin per": "Gross Margin",
        "Net Margin per": "Net Margin",
        "Operating Margin per": "Operating Margin",
        "Stock_Performance per": "Stock Performance",
        "DilutedEPS": "Diluted EPS"
    }
    desired_metric_order = list(rename_dict.values())

    # RAW Dataset
    sig_df_raw = pd.read_csv("outputs/s11_esg_sector_comparison.csv")

    avg_df = sig_df_raw[sig_df_raw["T-Test Significant"] == True].copy()
    avg_df["Metric"] = avg_df["Metric"].map(rename_dict)
    avg_df = avg_df.dropna(subset=["Metric"])

    ranked_avg = avg_df.pivot(index="Sector", columns="Metric", values="Ranked Avg metric")
    unranked_avg = avg_df.pivot(index="Sector", columns="Metric", values="Unranked Avg metric")
    percent_diff_avg = (ranked_avg - unranked_avg) / unranked_avg.abs() * 100
    percent_diff_avg = percent_diff_avg.reindex(columns=desired_metric_order)

    plt.figure(figsize=(12, 8))
    sns.heatmap(percent_diff_avg, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
    plt.title("Heatmap of % Difference in Average Metrics (Ranked vs Unranked)")
    plt.xlabel("Metric")
    plt.ylabel("Sector")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_avg_percent_diff_raw.png")
    plt.close()

    median_df = sig_df_raw[sig_df_raw["MWU Significant"] == True].copy()
    median_df["Metric"] = median_df["Metric"].map(rename_dict)
    median_df = median_df.dropna(subset=["Metric"])

    if not median_df.empty:
        ranked_med = median_df.pivot(index="Sector", columns="Metric", values="Ranked Median metric")
        unranked_med = median_df.pivot(index="Sector", columns="Metric", values="Unranked Median metric")
        percent_diff_med = (ranked_med - unranked_med) / unranked_med.abs() * 100
        percent_diff_med = percent_diff_med.reindex(columns=desired_metric_order)

        plt.figure(figsize=(12, 8))
        sns.heatmap(percent_diff_med, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
        plt.title("Heatmap of % Difference in Median Metrics (Ranked vs Unranked)")
        plt.xlabel("Metric")
        plt.ylabel("Sector")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/heatmap_median_percent_diff_raw.png")
        plt.close()

    # NORMALIZED Dataset
    sig_df_norm = pd.read_csv("outputs/s11_sector_comparison_normalized.csv")

    avg_df_norm = sig_df_norm[sig_df_norm["T-Test Significant"] == True].copy()
    avg_df_norm["Metric"] = avg_df_norm["Metric"].map(rename_dict)
    avg_df_norm = avg_df_norm.dropna(subset=["Metric"])

    ranked_avg_norm = avg_df_norm.pivot(index="Sector", columns="Metric", values="Ranked Avg metric")
    unranked_avg_norm = avg_df_norm.pivot(index="Sector", columns="Metric", values="Unranked Avg metric")
    percent_diff_avg_norm = (ranked_avg_norm - unranked_avg_norm) / unranked_avg_norm.abs() * 100
    percent_diff_avg_norm = percent_diff_avg_norm.reindex(columns=desired_metric_order)

    plt.figure(figsize=(12, 8))
    sns.heatmap(percent_diff_avg_norm, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
    plt.title("Heatmap of % Difference in Average Metrics (Ranked vs Unranked) [Normalized]")
    plt.xlabel("Metric")
    plt.ylabel("Sector")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_avg_percent_diff_norm.png")
    plt.close()

    median_df_norm = sig_df_norm[sig_df_norm["MWU Significant"] == True].copy()
    median_df_norm["Metric"] = median_df_norm["Metric"].map(rename_dict)
    median_df_norm = median_df_norm.dropna(subset=["Metric"])

    if not median_df_norm.empty:
        ranked_med_norm = median_df_norm.pivot(index="Sector", columns="Metric", values="Ranked Median metric")
        unranked_med_norm = median_df_norm.pivot(index="Sector", columns="Metric", values="Unranked Median metric")
        percent_diff_med_norm = (ranked_med_norm - unranked_med_norm) / unranked_med_norm.abs() * 100
        percent_diff_med_norm = percent_diff_med_norm.reindex(columns=desired_metric_order)

        plt.figure(figsize=(12, 8))
        sns.heatmap(percent_diff_med_norm, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
        plt.title("Heatmap of % Difference in Median Metrics (Ranked vs Unranked) [Normalized, MWU]")
        plt.xlabel("Metric")
        plt.ylabel("Sector")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/heatmap_median_percent_diff_norm.png")
        plt.close()


def analysis_4_tstat_scatter():
    sig_df = pd.read_csv("outputs/s11_esg_sector_comparison.csv")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sig_df.dropna(subset=["T-Statistic"]),
                    x="Metric", y="T-Statistic", hue="Sector", style="T-Test Significant", s=100)
    plt.axhline(0, color="grey", linestyle="--")
    plt.xticks(rotation=45, ha="right")
    plt.title("T-Statistic by Metric and Sector")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatter_tstat.png")
    plt.close()

def analysis_5_ttest_pval_heatmap():
    rename_dict = {
        "TotalRevenue": "Total Revenue",
        "GrossProfit": "Gross Profit",
        "NetIncome": "Net Income",
        "OperatingIncome": "Operating Income",
        "Gross Margin per": "Gross Margin",
        "Net Margin per": "Net Margin",
        "Operating Margin per": "Operating Margin",
        "Stock_Performance per": "Stock Performance",
        "DilutedEPS": "Diluted EPS"
    }
    desired_metric_order = list(rename_dict.values())

    def format_pval(val):
        if pd.isna(val):
            return ""
        elif val < 0.01:
            return f"{val:.1e}"
        else:
            return f"{val:.2f}"

    def plot_pval_heatmap(df, title, out_path):
        df["Metric"] = df["Metric"].map(rename_dict)
        df = df.dropna(subset=["Metric"])
        heatmap_data = df.pivot(index="Sector", columns="Metric", values="T-Test P-Value")
        heatmap_data = heatmap_data.reindex(columns=desired_metric_order)

        # Define color map: green for p < 0.05, red otherwise
        def color_func(val):
            if pd.isna(val):
                return "#ffffff"
            return "#90ee90" if val < 0.05 else "#ffcccb"

        color_matrix = heatmap_data.applymap(color_func)
        annot_matrix = heatmap_data.applymap(format_pval)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=annot_matrix, fmt="", cmap=["#90ee90", "#ffcccb"],
                    cbar=False, linewidths=0.5, linecolor="gray", mask=heatmap_data.isna(),
                    annot_kws={"size": 9}, square=False)

        # Apply custom colors
        for (i, j), val in np.ndenumerate(heatmap_data.values):
            if not pd.isna(val):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color_func(val), lw=0))

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Metric")
        ax.set_ylabel("Sector")
        plt.xticks(rotation=45, ha="right")

        # Legend
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color="#90ee90", label="P < 0.05"),
            mpatches.Patch(color="#ffcccb", label="P ≥ 0.05")
        ]
        plt.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    # RAW
    df_raw = pd.read_csv("outputs/s11_esg_sector_comparison_ALL_data.csv")
    plot_pval_heatmap(df_raw, "T-Test P-Values by Sector (Raw Metrics)",
                      f"{output_dir}/heatmap_ttest_pvals_raw.png")

    # NORMALIZED
    df_norm = pd.read_csv("outputs/s11_esg_sector_comparison_normalized_ALL_data.csv")
    plot_pval_heatmap(df_norm, "T-Test P-Values by Sector (Normalized Metrics)",
                      f"{output_dir}/heatmap_ttest_pvals_normalized.png")


if __name__ == "__main__":
    if analysis_flag == 1:
        pass
    elif analysis_flag == 2:
        analysis_2_violin_grouped_by_sector_normalized()
    elif analysis_flag == 3:
        analysis_3_heatmap_percent_diff()
    elif analysis_flag == 4:
        analysis_4_tstat_scatter()
    elif analysis_flag == 5:
        analysis_5_ttest_pval_heatmap()
    else:
        print("Invalid flag. Choose 1 to 5.")
