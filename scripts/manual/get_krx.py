import urllib.request
req = urllib.request.Request('https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd', headers={'User-Agent': 'Mozilla/5.0'})
res = urllib.request.urlopen(req)
html = res.read().decode('utf-8')
with open('krx_index.html', 'w', encoding='utf-8') as f:
    f.write(html)
