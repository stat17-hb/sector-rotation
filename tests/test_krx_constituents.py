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
