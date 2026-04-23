from __future__ import annotations

import src.data_sources.krx_constituents as constituents


class _DummyStockModule:
    def __init__(self, result):
        self.result = result
        self.calls: list[tuple[str, str | None, bool]] = []

    def get_index_portfolio_deposit_file(self, ticker, date=None, alternative=False):
        self.calls.append((ticker, date, alternative))
        return self.result


def test_lookup_index_constituents_uses_pykrx_wrapper_first(monkeypatch):
    stock_module = _DummyStockModule(["005930", "000660"])
    monkeypatch.setattr(
        constituents,
        "read_index_constituent_payload",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("raw fallback should not run")),
    )

    result = constituents.lookup_index_constituents(
        stock_module,
        sector_code="5044",
        candidate_dates=["20260410", "20260409"],
    )

    assert result.tickers == ["005930", "000660"]
    assert result.resolved_from == "20260410"
    assert result.source == "pykrx_wrapper"
    assert stock_module.calls == [("5044", "20260410", False)]


def test_lookup_index_constituents_falls_back_to_raw_payload(monkeypatch):
    stock_module = _DummyStockModule([])
    monkeypatch.setattr(
        constituents,
        "read_index_constituent_payload",
        lambda **kwargs: {
            "OutBlock_1": [
                {"ISU_SRT_CD": "005930"},
                {"ISU_SRT_CD": "000660"},
            ]
        },
    )

    result = constituents.lookup_index_constituents(
        stock_module,
        sector_code="5044",
        candidate_dates=["20260410"],
    )

    assert result.tickers == ["005930", "000660"]
    assert result.resolved_from == "20260410"
    assert result.source == "krx_raw_payload"
    assert stock_module.calls == [("5044", "20260410", False)]


def test_lookup_index_constituents_reports_failure_detail_when_all_paths_are_empty(monkeypatch):
    stock_module = _DummyStockModule([])
    monkeypatch.setattr(constituents, "read_index_constituent_payload", lambda **kwargs: {"output": []})

    result = constituents.lookup_index_constituents(
        stock_module,
        sector_code="5044",
        candidate_dates=["20260410", "20260409"],
    )

    assert result.tickers == []
    assert "empty constituent list across candidate dates 20260409..20260410" in result.failure_detail
    assert "raw-empty" in result.failure_detail


def test_lookup_index_constituents_short_circuits_on_auth_required(monkeypatch):
    stock_module = _DummyStockModule([])
    monkeypatch.setattr(
        constituents,
        "read_index_constituent_payload",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("AUTH_REQUIRED: KRX Data Marketplace login is required")
        ),
    )

    result = constituents.lookup_index_constituents(
        stock_module,
        sector_code="5044",
        candidate_dates=["20260410", "20260409"],
    )

    assert result.tickers == []
    assert result.failure_detail.startswith("AUTH_REQUIRED:")


def test_read_index_constituent_payload_raises_auth_required_for_login_html(monkeypatch):
    class _Response:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = "<html><body>로그인 페이지</body></html>"

        @staticmethod
        def json():
            raise ValueError("not json")

    monkeypatch.setattr(constituents, "request_krx_data", lambda *args, **kwargs: _Response())
    monkeypatch.setattr(
        constituents,
        "get_krx_login_state",
        lambda: {"configured": False, "authenticated": None, "detail": "KRX_ID/KRX_PW not configured"},
    )

    try:
        constituents.read_index_constituent_payload(trade_date="20260410", sector_code="5044")
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert str(exc).startswith("AUTH_REQUIRED:")


def test_read_index_constituent_payload_treats_empty_body_as_auth_required_when_not_configured(monkeypatch):
    class _Response:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = ""

        @staticmethod
        def json():
            raise ValueError("not json")

    monkeypatch.setattr(constituents, "request_krx_data", lambda *args, **kwargs: _Response())
    monkeypatch.setattr(
        constituents,
        "get_krx_login_state",
        lambda: {"configured": False, "authenticated": None, "detail": "KRX_ID/KRX_PW not configured"},
    )

    try:
        constituents.read_index_constituent_payload(trade_date="20260410", sector_code="5044")
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert str(exc).startswith("AUTH_REQUIRED:")


def test_read_index_constituent_payload_treats_generic_html_as_auth_required_after_failed_login(monkeypatch):
    class _Response:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = "<html><body>temporary page</body></html>"

        @staticmethod
        def json():
            raise ValueError("not json")

    monkeypatch.setattr(constituents, "request_krx_data", lambda *args, **kwargs: _Response())
    monkeypatch.setattr(
        constituents,
        "get_krx_login_state",
        lambda: {"configured": True, "authenticated": False, "detail": "CD999: login failed"},
    )

    try:
        constituents.read_index_constituent_payload(trade_date="20260410", sector_code="5044")
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        message = str(exc)
        assert message.startswith("AUTH_REQUIRED:")
        assert "CD999" in message


def test_candidate_reference_dates_applies_constituent_history_floor():
    dates = constituents.candidate_reference_dates("20140505", periods=10)

    assert dates
    assert min(dates) >= constituents.PYKRX_CONSTITUENT_HISTORY_FLOOR


def test_collect_constituent_request_evidence_captures_request_payload_and_matches(monkeypatch):
    class _Response:
        @staticmethod
        def json():
            return {
                "output": [
                    {"ISU_SRT_CD": "005930", "ISU_ABBRV": "삼성전자"},
                    {"ISU_SRT_CD": "000660", "ISU_ABBRV": "SK하이닉스"},
                ]
            }

    monkeypatch.setattr(constituents, "request_krx_data", lambda *args, **kwargs: _Response())

    evidence = constituents._collect_constituent_request_evidence(
        trade_date="20260417",
        sector_code="5044",
        target_ticker="005930",
    )

    assert evidence["sector_code"] == "5044"
    assert evidence["trade_date"] == "20260417"
    assert evidence["request_params"] == {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT00601",
        "locale": "ko_KR",
        "indIdx2": "044",
        "indIdx": "5",
        "trdDd": "20260417",
    }
    assert evidence["payload_keys"] == ["output"]
    assert evidence["matched_rows"] == [{"ISU_SRT_CD": "005930", "ISU_ABBRV": "삼성전자"}]
    assert evidence["extracted_tickers"] == ["005930", "000660"]
    assert evidence["extraction_block"] == "output"
    assert evidence["source"] == "krx_raw_payload"


def test_classify_overlap_evidences_detects_upstream_source_behavior():
    evidences = [
        {
            "sector_code": "1155",
            "request_params": {"bld": "x", "indIdx": "1", "indIdx2": "155", "trdDd": "20260417"},
            "matched_rows": [{"ISU_SRT_CD": "005930"}],
            "extracted_tickers": ["005930", "000660"],
        },
        {
            "sector_code": "5042",
            "request_params": {"bld": "x", "indIdx": "5", "indIdx2": "042", "trdDd": "20260417"},
            "matched_rows": [{"ISU_SRT_CD": "005930"}],
            "extracted_tickers": ["005930", "005380"],
        },
    ]

    verdict = constituents._classify_overlap_evidences(evidences=evidences, target_ticker="005930")

    assert verdict == "upstream_source_behavior"


def test_classify_overlap_evidences_detects_request_mapping_bug():
    evidences = [
        {
            "sector_code": "5042",
            "request_params": {"bld": "x", "indIdx": "5", "indIdx2": "042", "trdDd": "20260417"},
            "matched_rows": [{"ISU_SRT_CD": "005930"}],
            "extracted_tickers": ["005930"],
        },
        {
            "sector_code": "5044",
            "request_params": {"bld": "x", "indIdx": "5", "indIdx2": "042", "trdDd": "20260417"},
            "matched_rows": [{"ISU_SRT_CD": "005930"}],
            "extracted_tickers": ["005930"],
        },
    ]

    verdict = constituents._classify_overlap_evidences(evidences=evidences, target_ticker="005930")

    assert verdict == "request_mapping_bug"


def test_classify_overlap_evidences_detects_payload_parse_bug():
    evidences = [
        {
            "sector_code": "5042",
            "request_params": {"bld": "x", "indIdx": "5", "indIdx2": "042", "trdDd": "20260417"},
            "matched_rows": [{"ISU_SRT_CD": "005930"}],
            "extracted_tickers": ["000660"],
        }
    ]

    verdict = constituents._classify_overlap_evidences(evidences=evidences, target_ticker="005930")

    assert verdict == "payload_parse_bug"


def test_classify_overlap_evidences_detects_non_colliding_request_mapping_bug():
    evidences = [
        {
            "sector_code": "5044",
            "request_params": {"bld": "x", "indIdx": "5", "indIdx2": "042", "trdDd": "20260417"},
            "matched_rows": [{"ISU_SRT_CD": "005930"}],
            "extracted_tickers": ["005930"],
        }
    ]

    verdict = constituents._classify_overlap_evidences(evidences=evidences, target_ticker="005930")

    assert verdict == "request_mapping_bug"


def test_collect_same_date_ticker_overlap_artifact_builds_verdict(monkeypatch):
    monkeypatch.setattr(
        constituents,
        "_collect_constituent_request_evidence",
        lambda **kwargs: {
            "sector_code": kwargs["sector_code"],
            "trade_date": kwargs["trade_date"],
            "request_params": {
                "bld": "dbms/MDC/STAT/standard/MDCSTAT00601",
                "indIdx": kwargs["sector_code"][0],
                "indIdx2": kwargs["sector_code"][1:],
                "trdDd": kwargs["trade_date"],
            },
            "payload_keys": ["output"],
            "matched_rows": [{"ISU_SRT_CD": "005930"}],
            "extracted_tickers": ["005930"],
            "resolved_from": kwargs["trade_date"],
            "source": "krx_raw_payload",
            "extraction_block": "output",
        },
    )

    artifact = constituents.collect_same_date_ticker_overlap_artifact(
        trade_date="20260417",
        sector_codes=["1155", "5042", "5044"],
        target_ticker="005930",
    )

    assert artifact["trade_date"] == "20260417"
    assert artifact["target_ticker"] == "005930"
    assert artifact["sector_codes"] == ["1155", "5042", "5044"]
    assert artifact["verdict"] == "upstream_source_behavior"
    assert len(artifact["evidences"]) == 3
