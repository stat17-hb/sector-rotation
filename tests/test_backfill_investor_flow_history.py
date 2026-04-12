from __future__ import annotations

import scripts.backfill_investor_flow_history as backfill_script


def test_select_sample_windows_picks_earliest_middle_recent():
    windows = [
        ("20250101", "20250110"),
        ("20250113", "20250124"),
        ("20250127", "20250207"),
        ("20250210", "20250221"),
        ("20250224", "20250307"),
        ("20250310", "20250321"),
        ("20250324", "20250404"),
    ]

    selected = backfill_script._select_sample_windows(windows)

    assert selected == [
        ("20250101", "20250110"),
        ("20250113", "20250124"),
        ("20250127", "20250207"),
        ("20250210", "20250221"),
        ("20250324", "20250404"),
    ]


def test_classify_failure_distinguishes_auth_transport_and_gap():
    assert backfill_script._classify_failure({"status": "LIVE", "failed_codes": {}, "failed_days": []}) == "ok"
    assert backfill_script._classify_failure(
        {"failed_codes": {"sector:5044": "AUTH_REQUIRED: login needed"}, "failed_days": []}
    ) == "hard_auth"
    assert backfill_script._classify_failure(
        {"failed_codes": {"20250102": "timeout while connecting"}, "failed_days": ["20250102"]}
    ) == "transient_transport"
    assert backfill_script._classify_failure(
        {"failed_codes": {"20250102": "No normalized raw investor-flow rows were collected."}, "failed_days": ["20250102"]}
    ) == "data_gap"


def test_run_phase1_validation_uses_validation_reason_and_no_progress(monkeypatch):
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        backfill_script,
        "_discover_oldest_with_retry",
        lambda **kwargs: ("20250102", {"oldest_collectable_date": "20250102"}, [{"attempt": 1, "status": "LIVE"}]),
    )

    def _fake_run_chunk(**kwargs):
        calls.append(kwargs)
        return {
            "start": kwargs["start_date_str"],
            "end": kwargs["end_date_str"],
            "passed": True,
            "attempts": [{"attempt": 1, "passed": True}],
            "summary": {"status": "LIVE", "failed_days": [], "failed_codes": {}, "rows": 1},
            "classification": "ok",
            "failed_day_ratio": 0.0,
        }

    monkeypatch.setattr(backfill_script, "_run_chunk_with_retry", _fake_run_chunk)

    report = backfill_script._run_phase1_validation(
        sector_map={"regimes": {}},
        end_date_str="20250215",
        market="KR",
        oldest_date="",
        earliest_candidate_str="19900101",
        chunk_business_days=10,
        retry_attempts=3,
        retry_sleep_sec=30.0,
    )

    assert report["passed"] is True
    assert calls
    assert all(call["track_progress"] is False for call in calls)
    assert all(call["reason"] == backfill_script.HISTORICAL_BACKFILL_VALIDATION_REASON for call in calls)


def test_run_phase1_validation_aborts_after_two_consecutive_failures(monkeypatch):
    attempts = {"count": 0}

    monkeypatch.setattr(
        backfill_script,
        "_discover_oldest_with_retry",
        lambda **kwargs: ("20250102", {"oldest_collectable_date": "20250102"}, []),
    )

    def _fake_run_chunk(**kwargs):
        attempts["count"] += 1
        return {
            "start": kwargs["start_date_str"],
            "end": kwargs["end_date_str"],
            "passed": False,
            "attempts": [{"attempt": 1, "passed": False}],
            "summary": {"status": "SAMPLE", "failed_days": ["20250102"], "failed_codes": {"20250102": "timeout while connecting"}},
            "classification": "transient_transport",
            "failed_day_ratio": 0.05,
        }

    monkeypatch.setattr(backfill_script, "_run_chunk_with_retry", _fake_run_chunk)

    report = backfill_script._run_phase1_validation(
        sector_map={"regimes": {}},
        end_date_str="20250131",
        market="KR",
        oldest_date="",
        earliest_candidate_str="19900101",
        chunk_business_days=5,
        retry_attempts=3,
        retry_sleep_sec=30.0,
    )

    assert report["passed"] is False
    assert report["aborted"] is True
    assert "two consecutive sample chunks" in report["abort_reason"]
    assert attempts["count"] == 2


def test_run_phase2_full_backfill_uses_historical_reason_and_progress(monkeypatch):
    calls: list[dict[str, object]] = []

    def _fake_run_chunk(**kwargs):
        calls.append(kwargs)
        return {
            "start": kwargs["start_date_str"],
            "end": kwargs["end_date_str"],
            "passed": True,
            "attempts": [{"attempt": 1, "passed": True}],
            "summary": {"status": "LIVE", "failed_days": [], "failed_codes": {}, "rows": 5},
            "classification": "ok",
            "failed_day_ratio": 0.0,
        }

    monkeypatch.setattr(backfill_script, "_run_chunk_with_retry", _fake_run_chunk)
    monkeypatch.setattr(
        backfill_script,
        "read_investor_flow_raw_date_bounds",
        lambda **kwargs: {"min_trade_date": "20250102", "max_trade_date": "20250131"},
    )
    monkeypatch.setattr(
        backfill_script,
        "read_investor_flow_backfill_progress_cursor",
        lambda **kwargs: "20250131",
    )

    report = backfill_script._run_phase2_full_backfill(
        sector_map={"regimes": {}},
        end_date_str="20250131",
        market="KR",
        oldest_collectable_date="20250102",
        chunk_business_days=10,
        retry_attempts=3,
        retry_sleep_sec=30.0,
    )

    assert report["passed"] is True
    assert calls
    assert all(call["track_progress"] is True for call in calls)
    assert all(call["reason"] == backfill_script.HISTORICAL_BACKFILL_REASON for call in calls)


def test_discover_oldest_with_retry_does_not_retry_hard_auth(monkeypatch):
    state = {"calls": 0}

    def _fail(**kwargs):
        state["calls"] += 1
        raise RuntimeError("AUTH_REQUIRED: login needed")

    monkeypatch.setattr(backfill_script, "discover_oldest_collectable_date", _fail)
    monkeypatch.setattr(backfill_script.time, "sleep", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sleep should not run")))

    try:
        backfill_script._discover_oldest_with_retry(
            sector_map={"regimes": {}},
            end_date_str="20250131",
            earliest_candidate_str="19900101",
            market="KR",
            retry_attempts=3,
            retry_sleep_sec=30.0,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected discovery failure")

    assert state["calls"] == 1


def test_run_phase2_full_backfill_checks_persisted_closure_state(monkeypatch):
    monkeypatch.setattr(
        backfill_script,
        "_run_chunk_with_retry",
        lambda **kwargs: {
            "start": kwargs["start_date_str"],
            "end": kwargs["end_date_str"],
            "passed": True,
            "attempts": [{"attempt": 1, "passed": True}],
            "summary": {"status": "LIVE", "failed_days": [], "failed_codes": {}, "rows": 5},
            "classification": "ok",
            "failed_day_ratio": 0.0,
        },
    )
    monkeypatch.setattr(
        backfill_script,
        "read_investor_flow_raw_date_bounds",
        lambda **kwargs: {"min_trade_date": "20250102", "max_trade_date": "20250131"},
    )
    monkeypatch.setattr(
        backfill_script,
        "read_investor_flow_backfill_progress_cursor",
        lambda **kwargs: "20250131",
    )

    report = backfill_script._run_phase2_full_backfill(
        sector_map={"regimes": {}},
        end_date_str="20250131",
        market="KR",
        oldest_collectable_date="20250102",
        chunk_business_days=10,
        retry_attempts=3,
        retry_sleep_sec=30.0,
    )

    assert report["passed"] is True
    assert report["closure_checks"]["progress_reached_target_end"] is True
    assert report["closure_checks"]["min_trade_date_matches_oldest"] is True


def test_run_phase2_full_backfill_fails_when_min_trade_date_does_not_match(monkeypatch):
    monkeypatch.setattr(
        backfill_script,
        "_run_chunk_with_retry",
        lambda **kwargs: {
            "start": kwargs["start_date_str"],
            "end": kwargs["end_date_str"],
            "passed": True,
            "attempts": [{"attempt": 1, "passed": True}],
            "summary": {"status": "LIVE", "failed_days": [], "failed_codes": {}, "rows": 5},
            "classification": "ok",
            "failed_day_ratio": 0.0,
        },
    )
    monkeypatch.setattr(
        backfill_script,
        "read_investor_flow_raw_date_bounds",
        lambda **kwargs: {"min_trade_date": "20250103", "max_trade_date": "20250131"},
    )
    monkeypatch.setattr(
        backfill_script,
        "read_investor_flow_backfill_progress_cursor",
        lambda **kwargs: "20250131",
    )

    report = backfill_script._run_phase2_full_backfill(
        sector_map={"regimes": {}},
        end_date_str="20250131",
        market="KR",
        oldest_collectable_date="20250102",
        chunk_business_days=10,
        retry_attempts=3,
        retry_sleep_sec=30.0,
    )

    assert report["passed"] is False
    assert report["closure_checks"]["min_trade_date_matches_oldest"] is False


def test_main_requires_manual_non_regression_gate_for_success(monkeypatch):
    monkeypatch.setattr(
        backfill_script,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "market": "KR",
                "mode": "full",
                "end_date": "20250131",
                "oldest_date": "20250102",
                "earliest_candidate": "19900101",
                "chunk_business_days": 20,
                "retry_attempts": 3,
                "retry_sleep_sec": 30.0,
                "assume_non_regression_passed": False,
            },
        )(),
    )
    monkeypatch.setattr(
        backfill_script,
        "_load_configs",
        lambda *args, **kwargs: (
            {"benchmark_code": "1001"},
            {"regimes": {}},
            {},
            object(),
        ),
    )
    monkeypatch.setattr(
        backfill_script,
        "_run_phase2_full_backfill",
        lambda **kwargs: {"status": "LIVE", "passed": True, "manual_non_regression_required": True},
    )

    assert backfill_script.main() == 1


def test_historical_backfill_partial_day_failure_does_not_advance_progress(monkeypatch):
    import src.data_sources.krx_investor_flow as flow_source
    import src.data_sources.warehouse as warehouse
    import pandas as pd

    def _collect(**kwargs):
        day = kwargs["start"]
        raw = pd.DataFrame(
            {
                "trade_date": pd.to_datetime([day]),
                "ticker": ["005930"],
                "ticker_name": ["삼성전자"],
                "investor_type": ["외국인"],
                "buy_amount": [100],
                "sell_amount": [50],
                "net_buy_amount": [50],
            }
        )
        return raw, pd.DataFrame(), {"status": "LIVE", "failed_days": [day], "failed_codes": {"sector:5044": "AUTH_REQUIRED: login needed"}}

    monkeypatch.setattr(flow_source, "collect_sector_investor_flow", _collect)

    summary = flow_source.run_historical_investor_flow_backfill(
        sector_map={"regimes": {"Recovery": {"sectors": [{"code": "5044", "name": "KRX 반도체"}]}}},
        start_date_str="20250102",
        end_date_str="20250102",
        oldest_collectable_date="20250102",
        market="KR",
        track_progress=True,
        reason=flow_source.HISTORICAL_BACKFILL_REASON,
    )

    assert summary["status"] == "SAMPLE"
    assert warehouse.read_investor_flow_backfill_progress_cursor(market="KR") is None
