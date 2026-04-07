import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

req = urllib.request.Request('https://api.github.com/repos/sharebook-kr/pykrx/issues/276/comments')
try:
    with urllib.request.urlopen(req, context=ctx) as response:
        data = json.loads(response.read().decode())
        with open('issue_comments.txt', 'w', encoding='utf-8') as f:
            for c in data:
                f.write(f"--- {c['user']['login']} ---\n")
                f.write(c['body'] + "\n\n")
except Exception as e:
    print(f"Error: {e}")
