from pykrx.website.krx.krxio import KrxWebIo
from datetime import datetime

class IndexTickerScraper:
    def __init__(self):
        self.date = datetime.today().strftime("%Y%m%d")
        
    def fetch(self):
        krx = KrxWebIo()
        res = krx.post(
            bld="dbms/MDC/STAT/standard/MDCSTAT00101",
            trdDd=self.date,
        )
        return res.json()

if __name__ == "__main__":
    import json
    scraper = IndexTickerScraper()
    res = scraper.fetch()
    print(json.dumps(res, ensure_ascii=False)[:500])
