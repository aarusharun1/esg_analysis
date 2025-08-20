import pandas as pd

file_path = "processed_data/out16_Merged_Data_Ranked_and_Unranked_TRY.xlsx"
df = pd.read_excel(file_path, sheet_name="Merged_Data")

# metadata columns to always include
meta_columns = [
    'Ticker', 'Company Name', 'Country', 'GICS Sector',
    'fullTimeEmployees', 'Average Rank', 'Years Appeared',
    'Original Currency', 'ReportDate_2024'  
]

ranking_columns = [col for col in df.columns if col.endswith("_Ranking")]

# *_Incr_* columns
incr_columns_master = [col for col in df.columns if "Incr_" in col]
incr_base_names = set(col.rsplit("_", 1)[0] for col in incr_columns_master)

# financial metrics to normalize
financial_metrics = ["TotalRevenue", "NetIncome", "OperatingIncome", "GrossProfit", "DilutedEPS"]

output_rows = []

for _, row in df.iterrows():
    for rank_col in ranking_columns:
        try:
            rank_year = int(rank_col.split("_")[0])
        except ValueError:
            continue

        if pd.notna(row[rank_col]):
            prev_year = rank_year - 1
            output_row = {
                "Year": rank_year,
                "Ranking": row[rank_col],
                "Stock_Performance_Year": prev_year,
                "Stock_Performance": row.get(f"{prev_year}_Stock_Performance", None),
            }

            # add all metadata 
            for col in meta_columns:
                output_row[col] = row.get(col, None)

            # add *_Incr_* columns (only if ranking year is 2023–2025)
            if rank_year in [2023, 2024, 2025]:
                for base in incr_base_names:
                    colname = f"{base}_{prev_year}"
                    output_row[base] = row.get(colname, None)

            # add raw financial metrics (only if ranking year is 2022–2025)
            if rank_year in [2022, 2023, 2024, 2025]:
                for metric in financial_metrics:
                    colname = f"{metric}_{prev_year}"
                    output_row[metric] = row.get(colname, None)

            output_rows.append(output_row)

output_df = pd.DataFrame(output_rows)

output_csv_path = "out17_normalized_ranking_with_currency_and_reportdate.csv"
output_df.to_csv(output_csv_path, index=False)

print(f"done. saved to csv")
