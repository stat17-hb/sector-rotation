import traceback
from pykrx import stock

try:
    stock.get_index_ticker_list()
except Exception as e:
    traceback.print_exc()

try:
    stock.get_index_ohlcv("20240101", "20240110", "1001")
except Exception as e:
    traceback.print_exc()
