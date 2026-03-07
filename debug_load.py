import datetime
import yaml
import logging

logging.basicConfig(level=logging.DEBUG)

from src.data_sources.krx_indices import load_sector_prices
from src.data_sources.krx_openapi import get_krx_openapi_key

def main():
    with open("config/sector_map.yml", encoding="utf-8") as f:
        sm = yaml.safe_load(f)

    codes = [str(sm["benchmark"]["code"])]
    for r in sm.get("regimes", {}).values():
        for s in r.get("sectors", []):
            codes.append(str(s["code"]))

    market_end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    end_str = market_end_date.strftime("%Y%m%d")
    start_str = (market_end_date - datetime.timedelta(days=365*3)).strftime("%Y%m%d")

    print(f"Loading from {start_str} to {end_str}")
    status, df = load_sector_prices(codes, start_str, end_str)
    print("Final Status:", status)
    print("DataFrame rows:", len(df))

if __name__ == "__main__":
    main()
