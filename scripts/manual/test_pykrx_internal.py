from pykrx.website.krx.market.ticker import IndexTicker

ticker = IndexTicker()
print("PyKRX DataFrame columns:", getattr(ticker, 'df', None))
