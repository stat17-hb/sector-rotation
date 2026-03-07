import yaml
from src.data_sources.krx_openapi import fetch_index_ohlcv_openapi_batch_detailed, get_krx_openapi_key, get_krx_openapi_url
import datetime

def test_fetch():
    with open("config/sector_map.yml", encoding="utf-8") as f:
        sm = yaml.safe_load(f)

    codes = [str(sm["benchmark"]["code"])]
    for r in sm.get("regimes", {}).values():
        for s in r.get("sectors", []):
            codes.append(str(s["code"]))

    print(f"Fetching codes: {codes}")
    
    # Use yesterday for testing
    bas_dd = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d")
    if datetime.datetime.now().weekday() == 6: # Sunday
        bas_dd = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y%m%d")
    elif datetime.datetime.now().weekday() == 0: # Monday
        bas_dd = (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y%m%d")
    else:
        bas_dd = "20260306" # Friday, known good business day

    k = get_krx_openapi_key()
    suc, fail, det = fetch_index_ohlcv_openapi_batch_detailed(codes, bas_dd, bas_dd, auth_key=k)
    print("SUCCESS CODES:", list(suc.keys()))
    print("FAILURES:")
    for c, f in fail.items():
        print(f"[{c}]: {f}")

if __name__ == "__main__":
    test_fetch()
