import sys
sys.path.insert(0, '.')
from src.data_sources.common import load_secret_or_env, request_json_with_retry
import json

api_key = load_secret_or_env('KOSIS_API_KEY')
url = 'https://kosis.kr/openapi/Param/statisticsParameterData.do'

# itmId=ALL, objL1=T(전국)으로 DT_1J22003 항목 코드 전체 조회
params = {
    'method': 'getList',
    'apiKey': api_key,
    'itmId': 'ALL',
    'format': 'json',
    'jsonVD': 'Y',
    'prdSe': 'M',
    'startPrdDe': '202502',
    'endPrdDe': '202502',
    'orgId': '101',
    'tblId': 'DT_1J22003',
    'objL1': 'T',  # 전국
}
data = request_json_with_retry(url, params=params, client_name='KOSIS')
if isinstance(data, list):
    print(f'total rows: {len(data)}')
    if data:
        print('KEYS:', list(data[0].keys()))
    for r in data:
        print(f"ITM_ID={r.get('ITM_ID','?'):10s} ITM_NM={r.get('ITM_NM','?'):30s} DT={r.get('DT','?')}")
else:
    print(json.dumps(data, ensure_ascii=False, indent=2)[:1000])

print("\n--- Also try DT_1J22003 with item_id variations ---")
# 변화율 전용 테이블 시도: DT_1J22001 (전년동월대비), DT_1J22002 (전월대비)
for tbl in ['DT_1J22001', 'DT_1J22002']:
    params2 = {**params, 'tblId': tbl, 'itmId': 'T', 'objL1': 'ALL'}
    d2 = request_json_with_retry(url, params=params2, client_name='KOSIS')
    if isinstance(d2, list) and d2:
        sample = d2[0]
        print(f"{tbl}: rows={len(d2)}, sample DT={sample.get('DT')}, ITM_NM={sample.get('ITM_NM')}, C1={sample.get('C1')}")
    else:
        print(f"{tbl}: {d2}")
