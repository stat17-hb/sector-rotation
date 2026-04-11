import unittest

from python.collectors.krx_investor_flow.config import (
    MARKET_INVESTOR_FLOW_BLD,
    STOCK_INVESTOR_FLOW_BLD_CANDIDATES,
    build_market_payload,
    build_stock_payload,
)


class TestProbePayloadContract(unittest.TestCase):
    def test_market_payload_contains_expected_contract_fields(self) -> None:
        payload = build_market_payload(
            start_date="20260401",
            end_date="20260410",
            market_id="ALL",
        )

        self.assertEqual(payload["bld"], MARKET_INVESTOR_FLOW_BLD)
        self.assertEqual(payload["strtDd"], "20260401")
        self.assertEqual(payload["endDd"], "20260410")
        self.assertEqual(payload["mktId"], "ALL")
        self.assertEqual(payload["csvxls_isNo"], "false")

    def test_stock_payload_uses_bld_and_optional_ticker_finder_field(self) -> None:
        payload = build_stock_payload(
            start_date="20260401",
            end_date="20260410",
            isu_cd="KR7005930003",
            ticker="005930",
            bld=STOCK_INVESTOR_FLOW_BLD_CANDIDATES[0],
        )

        self.assertEqual(payload["bld"], STOCK_INVESTOR_FLOW_BLD_CANDIDATES[0])
        self.assertEqual(payload["isuCd"], "KR7005930003")
        self.assertEqual(payload["tboxisuCd_finder_stkisu0_25"], "005930/\\isuCd")


if __name__ == "__main__":
    unittest.main()
