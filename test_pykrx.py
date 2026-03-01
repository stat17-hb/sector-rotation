import pandas as pd
from pykrx import stock

print("Fetching index tickers...")
try:
    tickers = stock.get_index_ticker_list()
    print("Tickers:", tickers[:10], "...")
    if tickers:
        print("Fetching index name for", tickers[0])
        print(stock.get_index_ticker_name(tickers[0]))
except Exception as e:
    print("Error fetching tickers:", e)

print("\nFetching KOSPI OHLCV...")
try:
    df = stock.get_index_ohlcv("20240101", "20240110", "1001")
    print(df.head())
    print("Columns:", df.columns)
except Exception as e:
    print("Error fetching OHLCV:", e)
