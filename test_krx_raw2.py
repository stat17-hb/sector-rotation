import requests

url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
headers = {
    "Referer": "http://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}
payload = {
    "bld": "dbms/MDC/STAT/standard/MDCSTAT00101",
    "idxIndMidclssCd": "01",
    "trdDd": "20240419"
}

res = requests.post(url, data=payload, headers=headers)
print("Status:", res.status_code)
print("Headers:", res.headers)
print("Response length:", len(res.text))
print("Response text:", res.text[:500])
