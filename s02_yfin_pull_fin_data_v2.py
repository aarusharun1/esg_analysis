import yfinance as yf
import pandas as pd
import time

# load ticker 
tickers = pd.read_csv("processed_data/out11_unranked_companies.csv")["Ticker"].values

financial_data = []

# to avoid repeated API calls
fx_cache = {}

#  fetch Yahoo Finance price history
def safe_yf_history(ticker_obj, **kwargs):
    for attempt in range(3):
        try:
            data = ticker_obj.history(**kwargs)
            if not data.empty:
                return data
        except Exception as e:
            print(f"attempt {attempt + 1} failed: {e}")
        time.sleep(10)
    print("all attempts failed, returning empty data.")
    return pd.DataFrame()

# financial fields to collect
key_cols = [
    "TotalRevenue", "CostOfRevenue", "GrossProfit", "OperatingExpense", 
    "OperatingIncome", "NetIncome", "DilutedEPS", "InterestExpense",
    "IncomeTaxExpense", "NormalizedEBITDA", "EBITDA", "EBIT", 
    "NormalizedIncome", "TotalExpenses", "DilutedAverageShares", 
    "BasicAverageShares", "BasicEPS", "ResearchAndDevelopment", 
    "SellingGeneralAndAdministrative", "DepreciationAmortization"
]
key_cols = list(dict.fromkeys(key_cols))  # Remove duplicates

# company info fields
info_fields = [
    "fullTimeEmployees", "sector", "industry", "marketCap", "enterpriseValue",
    "beta", "dividendYield", "trailingPE", "forwardPE", 
    "priceToSalesTrailing12Months", "priceToBook", "totalDebt", 
    "debtToEquity", "operatingMargins", "profitMargins", "country"
]


for ticker in tickers:
    try:
        company = yf.Ticker(ticker)
        fin = company.get_income_stmt(freq="yearly")  # annual financials

        if fin is not None and not fin.empty:
            #  years as rows and metrics as columns
            fin = fin.T.reset_index()
            fin.rename(columns={"index": "Year-Month-Date"}, inplace=True)

            #  currency and ticker columns
            currency = company.info.get("financialCurrency", "N/A")
            fin["Original Currency"] = currency
            fin["Ticker"] = ticker            

            #  company-level info 
            company_info_data = {
                field: company.info.get(field, None) for field in info_fields
            }

            latest_fx_rate = 1.0  

            
            for i, row in fin.iterrows():                
                if isinstance(row["Year-Month-Date"], pd.Timestamp):
                    date_str = row["Year-Month-Date"].strftime("%Y-%m-%d")
                    year = row["Year-Month-Date"].year
                else:
                    date_str = str(row["Year-Month-Date"])
                    year = int(date_str[:4])

                #  FX rate
                fx_rate = 1.0
                if currency != "USD":
                    key = (currency, date_str)
                    if key in fx_cache:
                        fx_rate = fx_cache[key]
                    else:
                        fx_ticker = f"{currency}USD=X"
                        fx = yf.Ticker(fx_ticker)
                        fx_data = safe_yf_history(fx, end=date_str, period="5d")
                        if not fx_data.empty:
                            fx_rate = fx_data["Close"].iloc[-1]
                            fx_cache[key] = fx_rate
                        else:
                            print(f"warning: no FX data for {currency} on {date_str}, using 1.0")
                latest_fx_rate = fx_rate

                # Apply FX rate to numeric financials
                for col in fin.columns:
                    if col not in [
                        "Year-Month-Date", "DilutedAverageShares", "BasicAverageShares",
                        "TaxRateForCalcs", "Ticker", "Original Currency"
                    ]:
                        value = pd.to_numeric(row[col], errors="coerce")
                        fin.at[i, col] = value * fx_rate

                
            # company-level monetary values
            for field in ["marketCap", "enterpriseValue", "totalDebt"]:
                if company_info_data.get(field) is not None:
                    company_info_data[field] = company_info_data[field] * latest_fx_rate

            #  desired columns
            cols_to_keep = ["Year-Month-Date", "Ticker", "Original Currency"]
            for col in key_cols:
                if col in fin.columns:
                    cols_to_keep.append(col)
            fin_filtered = fin[cols_to_keep].copy()

            # add company info to each row
            for field, value in company_info_data.items():
                fin_filtered[field] = value

            # remove duplicate columns (just in case)
            fin_filtered = fin_filtered.loc[:, ~fin_filtered.columns.duplicated()]

            # add to list
            financial_data.append(fin_filtered)

            print(f"finished processing {ticker}")
        

    except Exception as e:
        print(f"error retrieving {ticker}: {e}")

if financial_data:
    df_financials = pd.concat(financial_data, ignore_index=True)
    df_financials.to_csv("processed_data/out13_unranked_financials_usd_enriched.csv", index=False)
    print("financial data with company info saved to csv")
else:
    print("no financial data retrieved")
