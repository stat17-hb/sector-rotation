import unittest

from python.collectors.krx_investor_flow.normalize import normalize_investor_flow_rows


class TestNormalizeInvestorFlowRows(unittest.TestCase):
    def test_normalizes_block1_rows_to_canonical_schema(self) -> None:
        response = {
            "block1": [
                {
                    "TRD_DD": "20260409",
                    "MKT_NM": "KOSPI",
                    "INVST_TP_NM": "외국인",
                    "BID_TRDVAL": "1,234",
                    "ASK_TRDVAL": "2,345",
                    "NETPRPS_TRDVAL": "-1,111",
                    "ISU_SRT_CD": "005930",
                    "ISU_ABBRV": "삼성전자",
                }
            ]
        }

        rows = normalize_investor_flow_rows(response)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.trade_date, "20260409")
        self.assertEqual(row.market, "KOSPI")
        self.assertEqual(row.investor_type, "외국인")
        self.assertEqual(row.buy_amount, 1234)
        self.assertEqual(row.sell_amount, 2345)
        self.assertEqual(row.net_buy_amount, -1111)
        self.assertEqual(row.ticker, "005930")
        self.assertEqual(row.ticker_name, "삼성전자")

    def test_uses_fallback_defaults_when_fields_are_missing(self) -> None:
        response = {"block1": [{"investor_type": "개인", "buy_amount": "10"}]}

        rows = normalize_investor_flow_rows(
            response,
            default_trade_date="20260409",
            default_market="ALL",
        )

        self.assertEqual(rows[0].trade_date, "20260409")
        self.assertEqual(rows[0].market, "ALL")
        self.assertEqual(rows[0].buy_amount, 10)


if __name__ == "__main__":
    unittest.main()
