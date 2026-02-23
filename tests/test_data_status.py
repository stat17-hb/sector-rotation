"""Pure function tests for data_status module. (4 tests)"""
from __future__ import annotations

import pytest

from src.ui.data_status import get_button_states, is_sample_mode


class TestDataStatus:
    def test_is_sample_mode_true_when_any_sample(self):
        """is_sample_mode returns True when any source is SAMPLE."""
        assert is_sample_mode({"price": "SAMPLE", "ecos": "LIVE"}) is True

    def test_is_sample_mode_false_when_no_sample(self):
        """is_sample_mode returns False when no source is SAMPLE."""
        assert is_sample_mode({"price": "LIVE", "ecos": "CACHED"}) is False

    def test_get_button_states_recompute_disabled_in_sample(self):
        """recompute button is disabled when any source is SAMPLE."""
        states = get_button_states({"price": "SAMPLE"})
        assert states["recompute"] is False

    def test_get_button_states_recompute_enabled_in_live(self):
        """recompute button is enabled when no source is SAMPLE."""
        states = get_button_states({"price": "LIVE"})
        assert states["recompute"] is True
        assert states["refresh_market"] is True
        assert states["refresh_macro"] is True
