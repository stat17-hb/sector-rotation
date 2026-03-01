import requests

session = requests.Session()
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "http://data.krx.co.kr",
}
session.headers.update(headers)

# Get initially blocked page or index.cmd to get cookies
session.get("http://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd")

url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
session.headers.update({
    "Referer": "http://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd",
})

payload = {
    "bld": "dbms/MDC/STAT/standard/MDCSTAT00101",
    "idxIndMidclssCd": "01",
    "trdDd": "20240419"
}

res = session.post(url, data=payload)
print("Status:", res.status_code)
print("Cookies:", session.cookies.get_dict())
print("Response text length:", len(res.text))
if res.status_code == 200 and len(res.text) > 20:
    print("Success! Got data.")
else:
    print("Response text:", res.text[:500])
