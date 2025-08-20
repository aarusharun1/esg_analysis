import yfinance as yf
import pandas as pd
import time

# load ticker
tickers = pd.read_csv("processed_data/out5_rankings_tickers.csv")["Ticker"].values

# get price history with retries
def safe_yf_history(ticker_obj, **kwargs):
    for attempt in range(2):
        try:
            data = ticker_obj.history(**kwargs)
            if not data.empty:
                return data
        except Exception as e:
            print(f"attempt {attempt + 1} failed: {e}")     
    
    return pd.DataFrame()

# years we want to calculate
years = list(range(2014, 2025))

# each row is a dict 
stock_returns = []

for ticker in tickers:
    try:
        company = yf.Ticker(ticker)
        row_data = {"Ticker": ticker}
        
        for year in years:
            stock_data = safe_yf_history(company, start=f"{year}-01-01", end=f"{year}-12-31")
            
            if not stock_data.empty:
                stock_data = stock_data.dropna(subset=["Open", "Close"])
                
                if not stock_data.empty:
                    first_open = stock_data["Open"].iloc[0]
                    last_close = stock_data["Close"].iloc[-1]
                    pct_change = ((last_close - first_open) / first_open) * 100
                    row_data[str(year)] = pct_change                    
                else:                    
                    row_data[str(year)] = None
            else:                
                row_data[str(year)] = None

        stock_returns.append(row_data)
        print(f"finished processing {ticker}")
        
        #time.sleep(1)

    except Exception as e:
        print(f"error retrieving {ticker}: {e}")

# convert list of dicts to DataFrame
df_returns = pd.DataFrame(stock_returns)

df_returns.to_csv("processed_data/out7_stock_returns_2016_2025.csv", index=False)
print("stock returns saved to csv")
