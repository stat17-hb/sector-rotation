from __future__ import annotations

from src.ui.copy import (
    ALL_ACTION_KEY,
    DEFAULT_UI_LOCALE,
    get_action_filter_label,
    get_action_label,
    get_all_action_label,
    get_decision_label,
    get_heatmap_palette_label,
    get_ui_text,
    normalize_action_filter,
)


def test_ui_copy_defaults_to_korean():
    assert DEFAULT_UI_LOCALE == "ko"
    assert get_all_action_label() == "전체"
    assert get_action_label("Watch") == "[~] 관망 (Watch)"
    assert get_decision_label("Strong Buy", held=True) == "추가 매수 후보"
    assert get_heatmap_palette_label("classic") == "기본 빨강/초록"
    assert get_ui_text("signals_empty") == "신호 데이터가 없습니다."
    assert get_ui_text("hero_method_badge") == "규칙 기반 판단"
    assert "규칙 기반 판단" in get_ui_text("judgment_disclaimer_caption")


def test_ui_copy_supports_english_fallback():
    assert get_all_action_label("en") == "All"
    assert get_action_label("Watch", "en") == "[~] Watch"
    assert get_decision_label("Strong Buy", held=False, locale="en") == "New buy candidate"
    assert get_heatmap_palette_label("contrast", "en") == "High-contrast red/green"
    assert get_action_filter_label(ALL_ACTION_KEY, "en") == "All"
    assert get_ui_text("hero_pit_badge", "en") == "confirmed_regime primary"
    assert "not PIT" in get_ui_text("sector_fit_note_none", "en")
    assert "lag0 nowcast" in get_ui_text("judgment_disclaimer_caption", "en")


def test_normalize_action_filter_maps_legacy_all_variants():
    assert normalize_action_filter("All") == ALL_ACTION_KEY
    assert normalize_action_filter("전체") == ALL_ACTION_KEY
    assert normalize_action_filter(ALL_ACTION_KEY) == ALL_ACTION_KEY
    assert normalize_action_filter("Watch") == "Watch"
