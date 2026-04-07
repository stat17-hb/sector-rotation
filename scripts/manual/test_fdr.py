import FinanceDataReader as fdr

try:
    print("Testing FinanceDataReader KS11 (KOSPI)...")
    df = fdr.DataReader('KS11', '2024-01-01', '2024-01-10')
    print("KOSPI Data:")
    print(df.head())
except Exception as e:
    print("Error fetching KS11:", e)

try:
    print("\nTesting FinanceDataReader KRX Sector Index (e.g., 1001 KOSPI)...")
    # KRX indices in FDR might use different symbols or might not be supported directly. 
    # Usually FDR uses investing.com or Naver for domestic indices.
    # Let's try to get krx sector index list or data
    # KOSPI 200 is KS200
    df = fdr.DataReader('KS200', '2024-01-01', '2024-01-10')
    print("KOSPI 200 Data:")
    print(df.head())
except Exception as e:
    print("Error fetching KS200:", e)

try:
    print("\nTesting stock listing (KRX)...")
    df = fdr.StockListing('KRX')
    print("KRX Listing:")
    print(df.head())
except Exception as e:
    print("Error fetching KRX listing:", e)
