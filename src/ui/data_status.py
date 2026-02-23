"""
Pure functions for data status checking and button state management.

R9 — SAMPLE mode: pure helper functions, no Streamlit dependency.
These can be tested without a Streamlit runtime.
"""
from __future__ import annotations


def is_sample_mode(data_status: dict) -> bool:
    """Return True if any data source is in SAMPLE mode.

    Args:
        data_status: Dict mapping source names to DataStatus strings.
                     Example: {"price": "SAMPLE", "ecos": "LIVE"}

    Returns:
        True if any value equals "SAMPLE", False otherwise.
    """
    return any(v == "SAMPLE" for v in data_status.values())


def get_button_states(data_status: dict) -> dict[str, bool]:
    """Return enabled/disabled state for each dashboard button.

    Button rules (R9):
    - 시장데이터 갱신 (refresh_market):  always enabled — attempts to escape SAMPLE
    - 매크로데이터 갱신 (refresh_macro):  always enabled — attempts to escape SAMPLE
    - 전체 재계산 (recompute):           disabled in SAMPLE (meaningless on synthetic data)

    Args:
        data_status: Dict mapping source names to DataStatus strings.

    Returns:
        dict with keys "refresh_market", "refresh_macro", "recompute".
    """
    sample = is_sample_mode(data_status)
    return {
        "refresh_market": True,
        "refresh_macro": True,
        "recompute": not sample,
    }
