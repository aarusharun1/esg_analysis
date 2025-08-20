import pandas as pd

file_path = "processed_data/out16_Merged_Data_Ranked_and_Unranked_TRY.xlsx"
df = pd.read_excel(file_path, sheet_name="Merged_Data")

#  metric and metadata columns
meta_columns = [
    'Ticker', 'Company Name', 'Country', 'GICS Sector',
    'fullTimeEmployees', 'Average Rank', 'Years Appeared',
    'Original Currency', 'ReportDate_2024'
]

# ranking info
ranking_columns = [col for col in df.columns if col.endswith("_Ranking")]

# "Incr_" columns and extract base metric names
incr_columns_master = [col for col in df.columns if "Incr_" in col]
incr_base_names = set(col.rsplit("_", 1)[0] for col in incr_columns_master)

# financial metrics to normalize
financial_metrics = ["TotalRevenue", "NetIncome", "OperatingIncome", "GrossProfit", "DilutedEPS"]

# target column order
desired_column_order = [
    "Year", "Ranking", "Stock_Performance_Year", "Stock_Performance",
    "Ticker", "Company Name", "Country", "GICS Sector", "fullTimeEmployees",
    "Average Rank", "Years Appeared", "Original Currency", "ReportDate_2024",
    "TotalRevenueIncr", "NetIncomeIncr", "OperatingIncomeIncr", "GrossProfitIncr", "DilutedEPSIncr",
    "TotalRevenue", "NetIncome", "OperatingIncome", "GrossProfit", "DilutedEPS"
]

unranked_rows = []

for _, row in df.iterrows():
    for year in range(2016, 2026):
        ranking_col = f"{year}_Ranking"
        if ranking_col in df.columns and pd.isna(row[ranking_col]):
            prev_year = year - 1
            output_row = {
                "Year": year,
                "Ranking": 0,
                "Stock_Performance_Year": prev_year,
                "Stock_Performance": row.get(f"{prev_year}_Stock_Performance", None),
                "Average Rank": row.get("Average Rank", 0),
                "Years Appeared": row.get("Years Appeared", 0),
            }

            # metadata
            for col in meta_columns:
                if col not in output_row:
                    output_row[col] = row.get(col, None)

            # Incr columns for 2023–2025
            if year in [2023, 2024, 2025]:
                for base in incr_base_names:
                    colname = f"{base}_{prev_year}"
                    if colname in df.columns:
                        output_row[base] = row.get(colname, None)

            # financial metrics for 2022–2025
            if year in [2022, 2023, 2024, 2025]:
                for metric in financial_metrics:
                    colname = f"{metric}_{prev_year}"
                    if colname in df.columns:
                        output_row[metric] = row.get(colname, None)

            unranked_rows.append(output_row)

unranked_df = pd.DataFrame(unranked_rows)

unranked_df_ordered = unranked_df[desired_column_order]

output_csv_path = "out18_unranked_all_years_final_ordered.csv"
unranked_df_ordered.to_csv(output_csv_path, index=False)

print(f"data saved to csv")
