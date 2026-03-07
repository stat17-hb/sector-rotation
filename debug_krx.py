import os
import sys
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from src.data_sources.krx_openapi import (
        get_krx_openapi_key,
        get_krx_openapi_url,
        _request_with_retry
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_fetch():
    key = get_krx_openapi_key()
    if not key:
        print("KRX_OPENAPI_KEY is not set or empty.")
        return

    print(f"Key loaded. Length: {len(key)}")
    api_id = "krx_dd_trd"
    url = get_krx_openapi_url(api_id)
    
    # Try a recent business day
    bas_dd = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    print(f"Fetching for {bas_dd} from {url}...")
    
    try:
        ret = _request_with_retry(
            url=url,
            auth_key=key,
            params={"basDd": bas_dd}
        )
        print("Success!")
        if isinstance(ret, dict):
            print("Response keys:", ret.keys())
            if "OutBlock_1" in ret or "output" in ret:
                print("Appears to have data.")
            else:
                print("Response data snippet:", str(ret)[:200])
        else:
            print("Response is not a dict:", type(ret))
    except Exception as e:
        print(f"Exception during fetch: {type(e).__name__} - {e}")

if __name__ == "__main__":
    test_fetch()
