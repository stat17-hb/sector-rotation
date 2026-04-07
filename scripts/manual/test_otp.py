import requests

session = requests.Session()
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "http://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd",
}
session.headers.update(headers)

# 1. Get OTP
otp_url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
otp_payload = {
    "bld": "dbms/MDC/STAT/standard/MDCSTAT00101",
    "locale": "ko_KR",
    "idxIndMidclssCd": "01",
    "trdDd": "20240419"
}
# wait, getJsonData *is* the data URL. OTP url is:
otp_url_real = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
# Actually I need to check how FinanceDataReader or others do it.
# They usually POST to http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd with bld
# Let's try downloading PyKRX from github master to see if there's a fix we can copy.
