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

