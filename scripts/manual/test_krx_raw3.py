import requests

url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
headers = {
    "Referer": "http://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd",  # Wait, I'll use HTTPS
}
headers["Referer"] = "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd"
headers["User-Agent"] = "Mozilla/5.0"

payload = {
    "bld": "dbms/MDC/STAT/standard/MDCSTAT00101",
    "idxIndMidclssCd": "01",
    "trdDd": "20240419"
}

res = requests.post(url, data=payload, headers=headers)
print("Status:", res.status_code)
print("Response text:", res.text[:500])
