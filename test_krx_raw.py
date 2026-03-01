import requests
from datetime import datetime

url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
headers = {
    "Referer": "http://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd",
    "User-Agent": "Mozilla/5.0",
}
payload = {
    "bld": "dbms/MDC/STAT/standard/MDCSTAT00101",
    "idxIndMidclssCd": "01",
    "trdDd": "20240419"
}

res = requests.post(url, data=payload, headers=headers)
data = res.json()
print("Keys:", data.keys())
if data.get('output'):
    print("First item:", data['output'][0])
else:
    print("Full response:", data)
