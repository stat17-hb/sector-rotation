from __future__ import annotations

from src.ui.copy import (
    ALL_ACTION_KEY,
    DEFAULT_UI_LOCALE,
    format_flow_cue_label,
    get_action_filter_label,
    get_action_label,
    get_all_action_label,
    get_decision_label,
    get_flow_sigma_subject_label,
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


def test_format_flow_cue_label_uses_copy_template_and_unavailable_fallback():
    assert format_flow_cue_label("supportive", 1.25) == "수급 우호 (+1.2σ)"
    assert format_flow_cue_label("adverse", None) == "수급 역풍"
    assert format_flow_cue_label("supportive", 1.25, "en") == "Supportive (+1.2σ)"


def test_flow_sigma_explainer_copy_renders_runtime_windows():
    body = get_ui_text(
        "flow_sigma_explainer_body",
        short_window=20,
        long_window=60,
    )
    assert "20" in body
    assert "60" in body
    assert "장기 표준편차" in body
    assert "투자자별 cue" in body or "투자자별" in body


def test_ui_copy_marks_execution_and_research_scope_honestly():
    assert "규칙 기반" in get_ui_text("judgment_disclaimer_caption")
    assert "not action-driving" in get_ui_text("sector_fit_note_none", "en")
    assert "하단" in get_ui_text("command_bar_eyebrow")
    assert "연구 뷰" in get_ui_text("command_bar_title")
    assert "상단 실전 대응 보드" in get_ui_text("command_bar_scope_note")
    assert "기본 판단 규칙은 바꾸지 않습니다" in get_ui_text("analysis_toolbar_title")


def test_ui_copy_marks_all_flow_reference_only_states_as_non_actionable():
    for key in (
        "flow_reference_only_partial",
        "flow_reference_only_transient",
        "flow_reference_only_stale",
    ):
        text = get_ui_text(key)
        assert "reference-only" in text
        assert "상단 실행 보드" in text
        assert "반영되지 않았습니다" in text
