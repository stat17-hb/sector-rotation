"""Locale-aware UI copy helpers."""
from __future__ import annotations

from typing import Literal


UiLocale = Literal["ko", "en"]
DEFAULT_UI_LOCALE: UiLocale = "ko"
ALL_ACTION_KEY = "__ALL__"

ACTION_IDS: tuple[str, ...] = ("Strong Buy", "Watch", "Hold", "Avoid", "N/A")
POSITION_MODE_IDS: tuple[str, ...] = ("all", "held", "new")
RANGE_PRESET_IDS: tuple[str, ...] = ("1Y", "3Y", "5Y", "ALL", "CUSTOM")
CYCLE_PHASE_IDS: tuple[str, ...] = (
    "ALL",
    "RECOVERY_EARLY",
    "RECOVERY_LATE",
    "EXPANSION_EARLY",
    "EXPANSION_LATE",
    "SLOWDOWN_EARLY",
    "SLOWDOWN_LATE",
    "CONTRACTION_EARLY",
    "CONTRACTION_LATE",
)
HEATMAP_PALETTE_IDS: tuple[str, ...] = ("classic", "contrast", "blue_orange")

_GENERAL_TEXT: dict[str, dict[UiLocale, str]] = {
    "all_action_label": {"ko": "전체", "en": "All"},
    "summary_all_signals": {"ko": "전체 신호", "en": "All signals"},
    "summary_alerted_only": {"ko": "알림 있는 항목", "en": "Only alerted"},
    "scope_matching_regime": {"ko": "현재 국면 해당", "en": "Matching current regime"},
    "scope_full_universe": {"ko": "전체 유니버스", "en": "Full universe"},
    "command_bar_eyebrow": {"ko": "빠른 필터", "en": "Quick filters"},
    "command_bar_title": {"ko": "분석 캔버스로 돌아가기 전에 실시간 신호 보드를 정제하세요.", "en": "Refine the live signal board before returning to the analysis canvas."},
    "filter_action": {"ko": "액션 필터", "en": "Action filter"},
    "filter_regime_only": {"ko": "현재 국면만", "en": "Current regime only"},
    "filter_position_scope": {"ko": "포지션 범위", "en": "Position scope"},
    "filter_alerted_only": {"ko": "알림 있는 항목만", "en": "Alerted only"},
    "summary_regime_label": {"ko": "국면", "en": "Regime"},
    "summary_action_label": {"ko": "액션", "en": "Action"},
    "summary_scope_label": {"ko": "범위", "en": "Scope"},
    "summary_positions_label": {"ko": "포지션", "en": "Positions"},
    "summary_alerts_label": {"ko": "알림", "en": "Alerts"},
    "top_picks_empty_held_missing": {
        "ko": "포트폴리오 액션 추천을 위해 보유 섹터를 먼저 추가하세요.",
        "en": "Add held sectors first to enable portfolio action recommendations.",
    },
    "top_picks_empty_held": {
        "ko": "현재 결정 규칙에 부합하는 보유 섹터가 없습니다.",
        "en": "No held sectors match the current decision rules.",
    },
    "top_picks_empty_new": {
        "ko": "현재 결정 규칙에 부합하는 신규 매수 아이디어가 없습니다.",
        "en": "No new-buy ideas match the current decision rules.",
    },
    "top_picks_empty_all": {
        "ko": "현재 필터 조건에 맞는 섹터가 없습니다.",
        "en": "No sectors match the current filter set.",
    },
    "signals_empty": {
        "ko": "신호 데이터가 없습니다.",
        "en": "No signal data available.",
    },
    "signals_filtered_empty": {
        "ko": "활성 필터 조건에 맞는 섹터가 없습니다.",
        "en": "No sectors match the active filters.",
    },
    "top_picks_showing": {
        "ko": "조건에 맞는 섹터 {total}개 중 상위 {limit}개를 표시합니다.",
        "en": "Showing top {limit} of {total} matching sectors.",
    },
    "provisional_caption": {
        "ko": "* 잠정 매크로 데이터의 영향을 받은 섹터가 포함됩니다.",
        "en": "* Includes sectors influenced by provisional macro data.",
    },
    "col_rank": {"ko": "순위", "en": "Rank"},
    "col_sector": {"ko": "섹터", "en": "Sector"},
    "col_decision": {"ko": "투자 판단", "en": "Decision"},
    "col_reason": {"ko": "사유", "en": "Reason"},
    "col_risk": {"ko": "리스크", "en": "Risk"},
    "col_invalidation": {"ko": "무효화 조건", "en": "Invalidation"},
    "col_alerts": {"ko": "알림", "en": "Alerts"},
    "col_held": {"ko": "보유", "en": "Held"},
    "col_in_regime": {"ko": "국면 적합", "en": "In Regime"},
    "col_action": {"ko": "액션", "en": "Action"},
    "col_etf": {"ko": "매수 ETF", "en": "ETF"},
    "col_volatility": {"ko": "변동성", "en": "Volatility"},
    "col_mdd_3m": {"ko": "MDD (3개월)", "en": "MDD (3M)"},
    "period_1m": {"ko": "1개월", "en": "1M"},
    "period_3m": {"ko": "3개월", "en": "3M"},
    "hero_regime_confirmed": {"ko": "확정 국면", "en": "Confirmed regime"},
    "hero_regime_provisional": {"ko": "잠정 국면", "en": "Provisional regime"},
    "hero_provisional_badge": {
        "ko": "잠정 매크로 데이터 포함",
        "en": "Includes provisional macro prints",
    },
    "hero_eyebrow": {"ko": "매크로 국면", "en": "Macro regime"},
    "hero_leading_index": {"ko": "선행지수", "en": "Leading index"},
    "hero_cpi_yoy": {"ko": "CPI 전년비", "en": "CPI YoY"},
    "status_current_regime": {"ko": "현재 국면", "en": "Current regime"},
    "status_market_data": {"ko": "시장 데이터", "en": "Market data"},
    "status_macro_data": {"ko": "매크로 데이터", "en": "Macro data"},
    "status_yield_curve": {"ko": "수익률 곡선 (Yield curve)", "en": "Yield curve"},
    "status_regime_confirmed": {"ko": "확정", "en": "Confirmed"},
    "status_regime_provisional": {"ko": "잠정", "en": "Provisional"},
    "status_market_detail": {"ko": "로컬 캐시 / 실시간 가격", "en": "Warehouse / live price path"},
    "status_macro_detail": {"ko": "월간 매크로 저장소", "en": "Monthly macro warehouse"},
    "status_yield_inverted": {"ko": "역전", "en": "Inverted"},
    "status_yield_normal": {"ko": "정상", "en": "Normal"},
    "status_yield_detail": {"ko": "3년 국채 대비 기준금리", "en": "3Y government bond versus base rate"},
    "action_summary_universe": {"ko": "유니버스", "en": "Universe"},
    "action_summary_filtered_count": {"ko": "필터 적용 후 섹터 수", "en": "Filtered sectors in view"},
    "action_distribution_title": {"ko": "액션 분포", "en": "Action distribution"},
    "action_distribution_yaxis": {"ko": "섹터 수", "en": "Sector count"},
    "cycle_palette_label": {"ko": "사이클 팔레트", "en": "Cycle palette"},
    "cycle_phase_control": {"ko": "사이클 국면", "en": "Cycle phase"},
    "cycle_empty_message": {
        "ko": "선택 기간에 대한 경기 국면 이력이 없습니다.",
        "en": "No cycle history is available for the selected window.",
    },
    "cycle_title": {"ko": "사이클 타임라인 (월별)", "en": "Cycle timeline (monthly)"},
    "cycle_status_label": {"ko": "상태", "en": "Status"},
    "cycle_status_indeterminate": {"ko": "판단 유보", "en": "Indeterminate"},
    "cycle_status_selected": {"ko": "선택됨", "en": "Selected"},
    "cycle_status_current": {"ko": "현재", "en": "Current"},
    "cycle_status_context": {"ko": "맥락", "en": "Context"},
    "cycle_summary_missing": {"ko": "섹터 요약 정보가 없습니다.", "en": "No sector summary is available."},
    "decision_data_check": {"ko": "데이터 확인", "en": "Data check"},
    "reason_regime_fit": {"ko": "국면 적합", "en": "Regime fit"},
    "reason_rs_vs_trend": {"ko": "RS {value:+.1f}% (추세 대비)", "en": "RS {value:+.1f}% vs trend"},
    "reason_trend_intact": {"ko": "추세 유지", "en": "Trend intact"},
    "reason_return_3m": {"ko": "3개월 {value:+.1f}%", "en": "3M {value:+.1f}%"},
    "reason_need_confirming_strength": {"ko": "추가 강도 확인 필요", "en": "Need more confirming strength"},
    "risk_regime_mismatch": {"ko": "국면 불일치", "en": "Regime mismatch"},
    "risk_rs_below_trend": {"ko": "RS {value:+.1f}% (추세 하회)", "en": "RS {value:+.1f}% below trend"},
    "risk_trend_weakened": {"ko": "추세 약화", "en": "Trend weakened"},
    "risk_volatility": {"ko": "20일 변동성 {value:.1f}%", "en": "20D vol {value:.1f}%"},
    "risk_none": {"ko": "주요 리스크 없음", "en": "No major risk flags"},
    "invalid_wait_for_data": {"ko": "벤치마크 및 섹터 가격 데이터를 기다리세요.", "en": "Wait for benchmark and sector price coverage."},
    "invalid_break_regime_fit": {"ko": "국면 적합성 소실 또는 RS가 추세 하회 시 무효화.", "en": "Invalidate if regime fit breaks or RS falls below trend."},
    "invalid_rs_below_trend": {"ko": "다음 검토 시까지 RS가 추세 하회 지속 시 무효화.", "en": "Invalidate if RS remains below trend through the next review."},
    "invalid_regime_mismatch_persists": {"ko": "국면 불일치 지속 및 더 강한 로테이션 출현 시 무효화.", "en": "Invalidate if regime mismatch persists and stronger rotations appear."},
    "invalid_promote_after_improve": {"ko": "국면 적합성과 RS 추세가 모두 개선된 후 상향 조정.", "en": "Promote only after regime fit and RS trend both improve."},
    "rs_trend_above": {"ko": "추세 상회", "en": "Above trend"},
    "rs_trend_below": {"ko": "추세 하회", "en": "Below trend"},
    "alerts_none": {"ko": "없음", "en": "None"},
    "regime_fit_yes": {"ko": "적합", "en": "Fit"},
    "regime_fit_no": {"ko": "불일치", "en": "Mismatch"},
    "conclusion_template": {
        "ko": "{decision} | 국면: {regime_fit} | RS 추세: {rs_trend} | 3개월: {return_3m} | 변동성: {volatility_20d} | 알림: {alerts_text}",
        "en": "{decision} | Regime: {regime_fit} | RS trend: {rs_trend} | 3M: {return_3m} | Volatility: {volatility_20d} | Alerts: {alerts_text}",
    },
}

_ACTION_LABELS: dict[str, dict[UiLocale, str]] = {
    "Strong Buy": {"ko": "[+] 강력 매수 (Strong Buy)", "en": "[+] Strong Buy"},
    "Watch": {"ko": "[~] 관망 (Watch)", "en": "[~] Watch"},
    "Hold": {"ko": "[=] 유지 (Hold)", "en": "[=] Hold"},
    "Avoid": {"ko": "[x] 회피 (Avoid)", "en": "[x] Avoid"},
    "N/A": {"ko": "[-] N/A", "en": "[-] N/A"},
}

_DECISION_LABELS: dict[str, dict[str, dict[UiLocale, str]]] = {
    "held": {
        "Strong Buy": {"ko": "추가 매수 후보", "en": "Add candidate"},
        "Watch": {"ko": "유지 / 모니터링", "en": "Hold / monitor"},
        "Hold": {"ko": "비중 축소 / 교체", "en": "Reduce / rotate"},
        "Avoid": {"ko": "매도 / 청산 검토", "en": "Sell / exit review"},
        "N/A": {"ko": "데이터 확인", "en": "Data check"},
    },
    "new": {
        "Strong Buy": {"ko": "신규 매수 후보", "en": "New buy candidate"},
        "Watch": {"ko": "관심 종목", "en": "Watchlist"},
        "Hold": {"ko": "신규 진입 불가", "en": "Not a fresh buy"},
        "Avoid": {"ko": "회피", "en": "Avoid"},
        "N/A": {"ko": "데이터 확인", "en": "Data check"},
    },
}

_POSITION_MODE_LABELS: dict[str, dict[UiLocale, str]] = {
    "all": {"ko": "전체 섹터", "en": "All sectors"},
    "held": {"ko": "보유 섹터", "en": "Held positions"},
    "new": {"ko": "신규 아이디어", "en": "New ideas"},
}

_REGIME_SUBLABELS: dict[str, dict[UiLocale, str]] = {
    "Recovery": {"ko": "초기 사이클 반등", "en": "Early-cycle rebound"},
    "Expansion": {"ko": "위험 선호 성장 국면", "en": "Risk-on growth phase"},
    "Slowdown": {"ko": "성장 둔화", "en": "Growth cooling"},
    "Contraction": {"ko": "방어적 사이클", "en": "Defensive cycle"},
    "Indeterminate": {"ko": "신호 혼재 (판단 유보)", "en": "Signal mix is inconclusive"},
}

_RANGE_PRESET_LABELS: dict[str, dict[UiLocale, str]] = {
    "1Y": {"ko": "1Y", "en": "1Y"},
    "3Y": {"ko": "3Y", "en": "3Y"},
    "5Y": {"ko": "5Y", "en": "5Y"},
    "ALL": {"ko": "전체", "en": "All"},
    "CUSTOM": {"ko": "직접 설정", "en": "Custom"},
}

_CYCLE_PHASE_LABELS: dict[str, dict[UiLocale, str]] = {
    "ALL": {"ko": "전체 국면", "en": "All phases"},
    "RECOVERY_EARLY": {"ko": "회복 / 초기", "en": "Recovery / Early"},
    "RECOVERY_LATE": {"ko": "회복 / 후기", "en": "Recovery / Late"},
    "EXPANSION_EARLY": {"ko": "확장 / 초기", "en": "Expansion / Early"},
    "EXPANSION_LATE": {"ko": "확장 / 후기", "en": "Expansion / Late"},
    "SLOWDOWN_EARLY": {"ko": "둔화 / 초기", "en": "Slowdown / Early"},
    "SLOWDOWN_LATE": {"ko": "둔화 / 후기", "en": "Slowdown / Late"},
    "CONTRACTION_EARLY": {"ko": "수축 / 초기", "en": "Contraction / Early"},
    "CONTRACTION_LATE": {"ko": "수축 / 후기", "en": "Contraction / Late"},
}

_HEATMAP_PALETTE_LABELS: dict[str, dict[UiLocale, str]] = {
    "classic": {"ko": "기본 빨강/초록", "en": "Classic red/green"},
    "contrast": {"ko": "고대비 빨강/초록", "en": "High-contrast red/green"},
    "blue_orange": {"ko": "파랑/주황 발산형", "en": "Blue/orange diverging"},
}

_CYCLE_PALETTE_ITEMS: tuple[tuple[str, str], ...] = (
    ("Recovery", "cycle-recovery"),
    ("Expansion", "cycle-expansion"),
    ("Slowdown", "cycle-slowdown"),
    ("Contraction", "cycle-contraction"),
    ("Indeterminate", "cycle-indeterminate"),
)


def normalize_locale(locale: str | None) -> UiLocale:
    normalized = str(locale or DEFAULT_UI_LOCALE).strip().lower()
    return "en" if normalized == "en" else "ko"


def get_ui_text(key: str, locale: str | None = DEFAULT_UI_LOCALE, **kwargs: object) -> str:
    normalized_locale = normalize_locale(locale)
    template = _GENERAL_TEXT.get(key, {}).get(normalized_locale)
    if template is None:
        raise KeyError(f"Unknown UI text key: {key}")
    return template.format(**kwargs)


def get_all_action_label(locale: str | None = DEFAULT_UI_LOCALE) -> str:
    return get_ui_text("all_action_label", locale)


def normalize_action_filter(value: str | None) -> str:
    normalized = str(value or "").strip()
    if normalized in {"", ALL_ACTION_KEY, "All", "전체"}:
        return ALL_ACTION_KEY
    for locale in ("ko", "en"):
        if normalized == get_all_action_label(locale):
            return ALL_ACTION_KEY
    return normalized


def is_all_action_filter(value: str | None) -> bool:
    return normalize_action_filter(value) == ALL_ACTION_KEY


def get_action_label(action: str, locale: str | None = DEFAULT_UI_LOCALE) -> str:
    normalized_locale = normalize_locale(locale)
    return _ACTION_LABELS.get(action, {}).get(normalized_locale, f"[?] {action}")


def get_action_filter_label(value: str, locale: str | None = DEFAULT_UI_LOCALE) -> str:
    normalized = normalize_action_filter(value)
    if normalized == ALL_ACTION_KEY:
        return get_all_action_label(locale)
    return get_action_label(normalized, locale)


def get_decision_label(action: str, *, held: bool, locale: str | None = DEFAULT_UI_LOCALE) -> str:
    normalized_locale = normalize_locale(locale)
    bucket = "held" if held else "new"
    return _DECISION_LABELS[bucket].get(action, {}).get(normalized_locale, get_ui_text("decision_data_check", normalized_locale))


def get_position_mode_label(value: str, locale: str | None = DEFAULT_UI_LOCALE) -> str:
    normalized_locale = normalize_locale(locale)
    return _POSITION_MODE_LABELS.get(value, {}).get(normalized_locale, value)


def get_regime_subtitle(regime: str, locale: str | None = DEFAULT_UI_LOCALE) -> str:
    normalized_locale = normalize_locale(locale)
    return _REGIME_SUBLABELS.get(regime, {}).get(normalized_locale, regime)


def get_range_preset_label(preset: str, locale: str | None = DEFAULT_UI_LOCALE) -> str:
    normalized_locale = normalize_locale(locale)
    return _RANGE_PRESET_LABELS.get(preset, {}).get(normalized_locale, preset)


def get_cycle_phase_label(phase_key: str, locale: str | None = DEFAULT_UI_LOCALE) -> str:
    normalized_locale = normalize_locale(locale)
    return _CYCLE_PHASE_LABELS.get(phase_key, {}).get(normalized_locale, phase_key)


def get_heatmap_palette_label(palette: str, locale: str | None = DEFAULT_UI_LOCALE) -> str:
    normalized_locale = normalize_locale(locale)
    return _HEATMAP_PALETTE_LABELS.get(palette, {}).get(normalized_locale, palette)


def get_cycle_palette_items(locale: str | None = DEFAULT_UI_LOCALE) -> list[tuple[str, str]]:
    normalized_locale = normalize_locale(locale)
    return [
        (_REGIME_SUBLABELS.get(label, {}).get(normalized_locale, label), css_class)
        for label, css_class in _CYCLE_PALETTE_ITEMS
    ]


__all__ = [
    "ACTION_IDS",
    "ALL_ACTION_KEY",
    "CYCLE_PHASE_IDS",
    "DEFAULT_UI_LOCALE",
    "HEATMAP_PALETTE_IDS",
    "POSITION_MODE_IDS",
    "RANGE_PRESET_IDS",
    "UiLocale",
    "get_action_filter_label",
    "get_action_label",
    "get_all_action_label",
    "get_cycle_palette_items",
    "get_cycle_phase_label",
    "get_decision_label",
    "get_heatmap_palette_label",
    "get_position_mode_label",
    "get_range_preset_label",
    "get_regime_subtitle",
    "get_ui_text",
    "is_all_action_filter",
    "normalize_action_filter",
    "normalize_locale",
]
