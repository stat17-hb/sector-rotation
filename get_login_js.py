import urllib.request
import re

html = open('krx_index.html', encoding='utf-8').read()
js_matches = re.findall(r'src="([^"]+\.js[^"]*)"', html)
print(f"Found {len(js_matches)} js files")

for js_path in js_matches:
    if js_path.startswith('/'):
        url = "https://data.krx.co.kr" + js_path
    elif js_path.startswith('http'):
        url = js_path
    else:
        url = "https://data.krx.co.kr/" + js_path
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req)
        content = res.read().decode('utf-8')
        if 'MDCCOMS001D1' in content:
            print(f"--- MATCH IN {url} ---")
            idx = content.find('MDCCOMS001D1')
            start = max(0, idx - 500)
            end = min(len(content), idx + 1000)
            print(content[start:end])
    except Exception as e:
        pass
