from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.dashboard import analysis


def test_build_heatmap_display_transposes_monthly_frame():
    frame = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [3.0, 4.0]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    display = analysis._build_heatmap_display(frame)
    assert list(display.index) == ["A", "B"]
    assert list(display.columns) == ["2024-01", "2024-02"]


def test_extract_heatmap_selection_reads_customdata():
    event = {
        "selection": {
            "points": [
                {"customdata": ["2024-02", "KRX Semiconductor"]},
            ]
        }
    }
    assert analysis._extract_heatmap_selection(event) == ("2024-02", "KRX Semiconductor")


def test_top_pick_sort_key_prioritizes_action_then_rs_divergence():
    strong = SimpleNamespace(action="Strong Buy", rs=1.2, rs_ma=1.0)
    watch = SimpleNamespace(action="Watch", rs=2.0, rs_ma=1.0)
    stronger = SimpleNamespace(action="Strong Buy", rs=1.4, rs_ma=1.0)

    ordered = sorted([watch, strong, stronger], key=analysis._top_pick_sort_key)
    assert ordered == [stronger, strong, watch]


def test_top_pick_sort_key_uses_hybrid_rank_then_raw_when_active():
    signal_a = SimpleNamespace(action="Strong Buy", momentum_method="hybrid_return_rank_v1", mom_rank=2, mom_raw=0.12)
    signal_b = SimpleNamespace(action="Strong Buy", momentum_method="hybrid_return_rank_v1", mom_rank=1, mom_raw=0.08)
    signal_c = SimpleNamespace(action="Strong Buy", momentum_method="hybrid_return_rank_v1", mom_rank=2, mom_raw=0.25)

    ordered = sorted([signal_a, signal_b, signal_c], key=analysis._top_pick_sort_key)

    assert ordered == [signal_b, signal_c, signal_a]
