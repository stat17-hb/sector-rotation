"""Locale-aware UI copy helpers."""
from __future__ import annotations

from collections.abc import Mapping
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
FLOW_PROFILE_IDS: tuple[str, ...] = (
    "foreign_lead",
    "institutional_confirmation",
    "contrarian_retail",
)

_GENERAL_TEXT: dict[str, dict[UiLocale, str]] = {
    "all_action_label": {"ko": "전체", "en": "All"},
    "summary_all_signals": {"ko": "전체 신호", "en": "All signals"},
    "summary_alerted_only": {"ko": "알림 있는 항목", "en": "Only alerted"},
    "scope_matching_regime": {"ko": "현재 국면 해당", "en": "Matching current regime"},
    "scope_full_universe": {"ko": "전체 유니버스", "en": "Full universe"},
    "analysis_toolbar_eyebrow": {"ko": "리서치 범위", "en": "Research scope"},
    "analysis_toolbar_title": {
        "ko": "리서치 캔버스 범위 조정",
        "en": "Adjust research canvas scope",
    },
    "analysis_toolbar_description": {
        "ko": "아래 리서치 캔버스와 상세 탭의 분석 범위를 맞춥니다. 상단 보유/신규 대응 보드의 기본 판단 규칙은 바꾸지 않습니다.",
        "en": "Set the analysis scope for the research canvas and detail tabs below. This does not change the base judgment rules of the upper held/new decision boards.",
    },
    "analysis_toolbar_period_label": {"ko": "기간", "en": "Window"},
    "analysis_toolbar_cycle_label": {"ko": "사이클", "en": "Cycle"},
    "analysis_toolbar_sector_label": {"ko": "섹터", "en": "Sector"},
    "analysis_toolbar_start_date": {"ko": "시작일", "en": "Start date"},
    "analysis_toolbar_end_date": {"ko": "종료일", "en": "End date"},
    "analysis_toolbar_preset_label": {"ko": "빠른 기간 선택", "en": "Quick window presets"},
    "analysis_toolbar_apply": {"ko": "적용", "en": "Apply"},
    "stock_lookup_eyebrow": {"ko": "종목 → 섹터 조회", "en": "Stock -> sector lookup"},
    "stock_lookup_title": {
        "ko": "현재 적용 섹터 찾기",
        "en": "Find current sector",
    },
    "stock_lookup_description": {
        "ko": "종목명 또는 종목코드로 현재 적용 중인 섹터를 찾습니다. 조회 성공 시 현재 캔버스 섹터 선택만 좁히며, 상단 기본 판단 규칙은 바꾸지 않습니다.",
        "en": "Find the current dashboard sector for a stock by name or ticker/code. On success this narrows only the current canvas sector selection and does not change the upper decision rules.",
    },
    "stock_lookup_label": {"ko": "종목명 또는 종목코드", "en": "Stock name or ticker/code"},
    "stock_lookup_apply": {"ko": "섹터 찾기", "en": "Find sector"},
    "stock_lookup_market_note_kr": {
        "ko": "KR은 KRX 공식 섹터 구성종목 포함 여부 기준으로 섹터를 찾습니다.",
        "en": "KR resolves sectors by constituent membership within the official KRX sector universe.",
    },
    "stock_lookup_market_note_us": {
        "ko": "US는 발행사 업종 분류를 대시보드 섹터로 매핑합니다.",
        "en": "US maps issuer classification onto the dashboard sector universe.",
    },
    "stock_lookup_matches_label": {"ko": "매칭 섹터", "en": "Matched sectors"},
    "stock_lookup_selected_suffix": {"ko": "현재 적용", "en": "Selected"},
    "stock_lookup_provenance_label": {"ko": "근거", "en": "Provenance"},
    "command_bar_eyebrow": {"ko": "하단 상세 뷰 필터", "en": "Downstream detail filters"},
    "command_bar_title": {
        "ko": "아래 요약·차트·테이블·탭의 연구 뷰만 정제합니다.",
        "en": "Refine only the downstream research views below: summary, charts, tables, and tabs.",
    },
    "command_bar_scope_note": {
        "ko": "이 필터는 아래 요약·차트·테이블 등 downstream research view만 정제하며, 상단 실전 대응 보드와 분석 캔버스는 바꾸지 않습니다.",
        "en": "These filters refine only the downstream filtered tab views (summary, charts, tables); they do not change the upper decision boards or analysis canvas.",
    },
    "filter_action": {"ko": "대응 필터", "en": "Action filter"},
    "filter_regime_only": {"ko": "현재 국면만", "en": "Current regime only"},
    "filter_position_scope": {"ko": "포지션 범위", "en": "Position scope"},
    "filter_alerted_only": {"ko": "알림 있는 항목만", "en": "Alerted only"},
    "summary_regime_label": {"ko": "국면", "en": "Regime"},
    "summary_action_label": {"ko": "대응", "en": "Action"},
    "summary_scope_label": {"ko": "범위", "en": "Scope"},
    "summary_positions_label": {"ko": "포지션", "en": "Positions"},
    "summary_alerts_label": {"ko": "알림", "en": "Alerts"},
    "decision_context_held": {"ko": "보유 검토", "en": "Held review"},
    "decision_context_new": {"ko": "신규 후보", "en": "New candidate"},
    "decision_lane_eyebrow": {"ko": "의사결정 지원", "en": "Decision support"},
    "decision_lane_title": {"ko": "규칙 기반 섹터 후보", "en": "Rules-based sector candidates"},
    "decision_lane_description": {
        "ko": "아래 보유/신규 보드는 현재 규칙 기준의 의사결정 지원 표면입니다. 하단 리서치 캔버스와 탭은 검증용입니다.",
        "en": "The held/new boards below are decision-support surfaces under the current rules. The lower research canvas and tabs are for validation.",
    },
    "decision_card_thesis": {"ko": "핵심 판단", "en": "Core thesis"},
    "decision_card_why": {"ko": "왜", "en": "Why"},
    "decision_card_invalidation": {"ko": "무효화 조건", "en": "Invalidation"},
    "decision_card_confidence": {"ko": "판단 신뢰도", "en": "Decision confidence"},
    "decision_card_regime_fit": {"ko": "국면", "en": "Regime"},
    "decision_card_sector_fit": {"ko": "실증 적합도", "en": "Empirical fit"},
    "top_picks_empty_held_missing": {
        "ko": "보유 섹터 검토 후보를 보려면 보유 섹터를 먼저 추가하세요.",
        "en": "Add held sectors first to enable held-sector review candidates.",
    },
    "top_picks_empty_held": {
        "ko": "현재 결정 규칙에 부합하는 보유 섹터가 없습니다.",
        "en": "No held sectors match the current decision rules.",
    },
    "top_picks_empty_new": {
        "ko": "현재 결정 규칙에 부합하는 신규 검토 후보가 없습니다.",
        "en": "No new review candidates match the current decision rules.",
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
    "col_decision": {"ko": "신호 판단", "en": "Signal judgment"},
    "col_reason": {"ko": "사유", "en": "Reason"},
    "col_risk": {"ko": "리스크", "en": "Risk"},
    "col_invalidation": {"ko": "무효화 조건", "en": "Invalidation"},
    "col_alerts": {"ko": "알림", "en": "Alerts"},
    "col_held": {"ko": "보유", "en": "Held"},
    "col_in_regime": {"ko": "국면 적합", "en": "In Regime"},
    "col_macro_context": {"ko": "매크로 참고", "en": "Macro Context"},
    "col_action": {"ko": "대응", "en": "Action"},
    "col_taxonomy": {"ko": "세부 분류", "en": "Taxonomy"},
    "col_etf": {"ko": "참고 ETF", "en": "ETF reference"},
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
    "hero_method_badge": {
        "ko": "규칙 기반 판단",
        "en": "Rules-based heuristic",
    },
    "hero_pit_badge": {
        "ko": "confirmed_regime 기준",
        "en": "confirmed_regime primary",
    },
    "hero_contraction_badge": {
        "ko": "역사 문서 분리",
        "en": "Historical docs separated",
    },
    "hero_eyebrow": {"ko": "거시경제 국면", "en": "Macro regime"},
    "hero_leading_index": {"ko": "선행지수", "en": "Leading index"},
    "hero_cpi_yoy": {"ko": "CPI 전년비", "en": "CPI YoY"},
    "status_current_regime": {"ko": "현재 국면", "en": "Current regime"},
    "status_market_data": {"ko": "시장 데이터", "en": "Market data"},
    "status_macro_data": {"ko": "매크로 데이터", "en": "Macro data"},
    "status_yield_curve": {"ko": "수익률 곡선 (Yield curve)", "en": "Yield curve"},
    "status_regime_confirmed": {"ko": "확정", "en": "Confirmed"},
    "status_regime_provisional": {"ko": "잠정", "en": "Provisional"},
    "status_market_detail": {"ko": "로컬 캐시 / 가격 경로", "en": "Warehouse / price path"},
    "status_macro_detail": {"ko": "월간 매크로 저장소", "en": "Monthly macro warehouse"},
    "status_yield_inverted": {"ko": "역전", "en": "Inverted"},
    "status_yield_normal": {"ko": "정상", "en": "Normal"},
    "status_yield_detail": {"ko": "3년 국채 대비 기준금리", "en": "3Y government bond versus base rate"},
    "action_summary_universe": {"ko": "전체 분석 대상", "en": "Universe"},
    "action_summary_filtered_count": {"ko": "필터 적용 후 섹터 수", "en": "Filtered sectors in view"},
    "action_distribution_title": {"ko": "투자의견 분포", "en": "Action distribution"},
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
    "reason_momentum_percentile": {"ko": "모멘텀 백분위 {value:.0f}p", "en": "Momentum percentile {value:.0f}p"},
    "reason_momentum_raw": {"ko": "중기 초과수익 {value:+.1f}%", "en": "Medium-term excess return {value:+.1f}%"},
    "reason_rs_vs_trend": {"ko": "RS {value:+.1f}% (추세 대비)", "en": "RS {value:+.1f}% vs trend"},
    "reason_trend_intact": {"ko": "추세 유지", "en": "Trend intact"},
    "reason_trend_200dma": {"ko": "200DMA 상회", "en": "Above 200DMA"},
    "reason_return_3m": {"ko": "3개월 {value:+.1f}%", "en": "3M {value:+.1f}%"},
    "reason_sector_fit_rank": {"ko": "lag0 nowcast 실증 순위 {rank}/{total}", "en": "lag0 nowcast empirical rank {rank}/{total}"},
    "reason_need_confirming_strength": {"ko": "추가 강도 확인 필요", "en": "Need more confirming strength"},
    "judgment_disclaimer_caption": {
        "ko": "최종 표기는 규칙 기반 의사결정 지원 신호입니다. 실증 적합도는 lag0 nowcast reference only이며 PIT/action 신호가 아닙니다. 기본 모형(국면 × 모멘텀), 환율(FX) 안전장치, 실험적 수급 보정을 분리해 해석하세요.",
        "en": "Final labels are rules-based decision-support signals. Empirical fit is a lag0 nowcast reference only, not a PIT or action-driving signal. Read the base shell (regime x momentum), FX safety filter, and experimental flow overlay separately.",
    },
    "judgment_structure_label": {"ko": "판단 구조", "en": "Judgment stack"},
    "judgment_structure_base": {"ko": "기본 모형", "en": "Base shell"},
    "judgment_structure_fx": {"ko": "환율 안전장치", "en": "FX safety filter"},
    "judgment_structure_flow": {"ko": "실험적 수급 보정", "en": "Experimental flow overlay"},
    "judgment_confidence_limited": {"ko": "제한적 판단 규칙", "en": "Limited heuristic"},
    "judgment_confidence_flow": {"ko": "실험 보정 포함", "en": "Includes experimental overlay"},
    "risk_regime_mismatch": {"ko": "국면 불일치", "en": "Regime mismatch"},
    "risk_momentum_rank_fail": {"ko": "모멘텀 상위권 미달", "en": "Momentum rank below threshold"},
    "risk_rs_below_trend": {"ko": "RS {value:+.1f}% (추세 하회)", "en": "RS {value:+.1f}% below trend"},
    "risk_price_below_200dma": {"ko": "200DMA 하회", "en": "Below 200DMA"},
    "risk_trend_weakened": {"ko": "추세 약화", "en": "Trend weakened"},
    "risk_volatility": {"ko": "20일 변동성 {value:.1f}%", "en": "20D vol {value:.1f}%"},
    "risk_none": {"ko": "주요 리스크 없음", "en": "No major risk flags"},
    "invalid_wait_for_data": {"ko": "벤치마크 및 섹터 가격 데이터를 기다리세요.", "en": "Wait for benchmark and sector price coverage."},
    "invalid_kr_momentum_break": {"ko": "모멘텀 약화 또는 추세 이탈이 확인되면 재평가합니다.", "en": "Reassess if momentum weakens or trend breaks."},
    "invalid_kr_reentry": {"ko": "모멘텀 회복과 추세 확인이 함께 나오면 상향 조정합니다.", "en": "Promote only after momentum and trend recover together."},
    "invalid_break_regime_fit": {"ko": "국면 적합성 소실 또는 RS가 추세 하회 시 무효화.", "en": "Invalidate if regime fit breaks or RS falls below trend."},
    "invalid_rs_below_trend": {"ko": "다음 검토 시까지 RS가 추세 하회 지속 시 무효화.", "en": "Invalidate if RS remains below trend through the next review."},
    "invalid_hybrid_break": {"ko": "국면 적합성 소실, 모멘텀 상위권 이탈, 또는 200DMA 하회 시 무효화.", "en": "Invalidate if regime fit breaks, momentum rank falls, or price drops below 200DMA."},
    "invalid_hybrid_recovery": {"ko": "모멘텀 상위권 복귀와 200DMA 회복이 함께 확인된 뒤 상향 조정.", "en": "Promote only after momentum rank and 200DMA both recover."},
    "invalid_regime_mismatch_persists": {"ko": "국면 불일치 지속 및 더 강한 로테이션 출현 시 무효화.", "en": "Invalidate if regime mismatch persists and stronger rotations appear."},
    "invalid_promote_after_improve": {"ko": "국면 적합성과 RS 추세가 모두 개선된 후 상향 조정.", "en": "Promote only after regime fit and RS trend both improve."},
    "rs_trend_above": {"ko": "추세 상회", "en": "Above trend"},
    "rs_trend_below": {"ko": "추세 하회", "en": "Below trend"},
    "momentum_state_strong": {"ko": "상위 모멘텀", "en": "Top momentum"},
    "momentum_state_weak": {"ko": "약한 모멘텀", "en": "Weak momentum"},
    "alerts_none": {"ko": "없음", "en": "None"},
    "regime_fit_yes": {"ko": "적합", "en": "Fit"},
    "regime_fit_no": {"ko": "불일치", "en": "Mismatch"},
    "regime_reference": {"ko": "참고: {value}", "en": "Reference: {value}"},
    "sector_fit_missing": {"ko": "실증 rank 없음", "en": "No empirical rank"},
    "sector_fit_card": {"ko": "실증 적합도 (lag0)", "en": "Empirical fit (lag0)"},
    "sector_fit_note_none": {"ko": "lag0 nowcast reference only · not PIT · not action-driving", "en": "lag0 nowcast reference only · not PIT · not action-driving"},
    "conclusion_template": {
        "ko": "{decision} | 국면: {regime_fit} | RS 추세: {rs_trend} | 3개월: {return_3m} | 변동성: {volatility_20d} | 알림: {alerts_text}",
        "en": "{decision} | Regime: {regime_fit} | RS trend: {rs_trend} | 3M: {return_3m} | Volatility: {volatility_20d} | Alerts: {alerts_text}",
    },
    "conclusion_template_hybrid": {
        "ko": "{decision} | 국면: {regime_fit} | 모멘텀: {momentum_state} | 3개월: {return_3m} | 변동성: {volatility_20d} | 알림: {alerts_text}",
        "en": "{decision} | Regime: {regime_fit} | Momentum: {momentum_state} | 3M: {return_3m} | Volatility: {volatility_20d} | Alerts: {alerts_text}",
    },
    "conclusion_template_kr": {
        "ko": "{decision} | 매크로: {macro_context} | 모멘텀: {momentum_state} | 3개월: {return_3m} | 변동성: {volatility_20d} | 알림: {alerts_text}",
        "en": "{decision} | Macro: {macro_context} | Momentum: {momentum_state} | 3M: {return_3m} | Volatility: {volatility_20d} | Alerts: {alerts_text}",
    },
    "flow_unavailable": {
        "ko": "실험적 수급 데이터 없음",
        "en": "Experimental flow data unavailable",
    },
    "flow_supportive": {"ko": "수급 우호", "en": "Flow supportive"},
    "flow_neutral": {"ko": "수급 중립", "en": "Flow neutral"},
    "flow_adverse": {"ko": "수급 역풍", "en": "Flow adverse"},
    "flow_profile_label": {"ko": "수급 해석 유형", "en": "Flow profile"},
    "flow_status_label": {"ko": "투자자 수급", "en": "Investor flow"},
    "flow_refresh_button": {"ko": "투자자수급 갱신", "en": "Refresh investor flow"},
    "flow_sidebar_caption": {
        "ko": "비공식/실험 데이터는 수동 갱신 후 캐시로만 읽습니다.",
        "en": "Unofficial experimental data is refreshed manually and read from cache only.",
    },
    "flow_summary_title": {"ko": "투자자 수급 스냅샷", "en": "Investor-flow snapshot"},
    "flow_summary_description": {
        "ko": "상단 의사결정 지원 보드에 덧붙는 실험적 수급 overlay를 요약합니다. reference-only 상태에서는 최종 신호 판단을 바꾸지 않습니다.",
        "en": "Summarize the experimental investor-flow overlay next to the upper decision-support boards. In reference-only states it does not change the final signal label.",
    },
    "flow_summary_limit_note": {
        "ko": "강한 수급 신호가 보이는 상위 4개 섹터만 요약합니다.",
        "en": "Only the top 4 sectors with the strongest flow signal are summarized here.",
    },
    "flow_tab_warning": {
        "ko": "Unofficial / Experimental: 비공식 KRX 수급 데이터는 수동 갱신 후 캐시에서만 읽습니다.",
        "en": "Unofficial / Experimental: investor-flow data is read from the local cache after manual refresh only.",
    },
    "flow_tab_empty": {
        "ko": "표시할 투자자 수급 데이터가 없습니다. 사이드바에서 투자자수급 갱신을 실행하세요.",
        "en": "No investor-flow data is available yet. Run the manual refresh from the sidebar.",
    },
    "flow_reference_only_partial": {
        "ko": "현재 표시 중인 수급 데이터는 partial preview입니다. reference-only로만 보며, 상단 의사결정 지원 보드의 최종 신호에는 반영되지 않았습니다.",
        "en": "The visible investor-flow data is a partial preview. Treat it as reference-only; it is not reflected in the final signal label.",
    },
    "flow_reference_only_transient": {
        "ko": "현재 표시 중인 수급 데이터는 warehouse 저장 실패 후 임시 preview입니다. 현재 세션의 reference-only 값이며, 상단 의사결정 지원 보드의 최종 신호에는 반영되지 않았습니다.",
        "en": "The visible investor-flow data is a transient preview after warehouse write failure. It is session-only reference and is not reflected in the final signal label.",
    },
    "flow_reference_only_stale": {
        "ko": "현재 표시 중인 수급 데이터는 최신 complete 기준이 아닌 cached snapshot입니다. reference-only로만 보며, 상단 의사결정 지원 보드의 최종 신호에는 반영되지 않았습니다.",
        "en": "The visible investor-flow data is a cached snapshot that is not current to the latest complete cursor. Treat it as reference-only; it is not reflected in the final signal label.",
    },
    "flow_reference_only_summary_hint": {
        "ko": "상세 수급 수치는 투자자 수급 탭에서 참고하세요. 현재 값은 reference-only입니다.",
        "en": "See the investor-flow tab for the detailed reference snapshot.",
    },
    "flow_reference_only_action_hidden": {
        "ko": "현재 상태는 reference-only라 수급 기반 의견 변화 표를 숨기고 참여 주체 비교표와 raw snapshot만 표시합니다.",
        "en": "In this state, the action-change table is hidden while the participant comparison and raw snapshot remain visible.",
    },
    "flow_reference_only_action_hidden_raw_only": {
        "ko": "현재 상태는 reference-only라 수급 기반 의견 변화 표를 숨기고 raw snapshot만 표시합니다.",
        "en": "In this state, the action-change table is hidden and only the raw snapshot is shown.",
    },
    "summary_tab_role_caption": {
        "ko": "이 탭은 상단 의사결정 지원 보드의 요약/감사 레이어입니다. 기준 표면은 메인 페이지의 보유/신규 후보 보드입니다.",
        "en": "This tab is an additive recap and audit layer for the upper decision-support boards. The main held/new boards remain the reference surface.",
    },
    "analysis_canvas_eyebrow": {"ko": "리서치 캔버스", "en": "Research canvas"},
    "analysis_canvas_title": {"ko": "신호 판단 검증용 섹터 비교", "en": "Sector comparison for validating the signal context"},
    "analysis_canvas_description": {
        "ko": "아래 히트맵·사이클·상세 패널은 상단 의사결정 지원 보드를 검증하고 맥락을 넓히는 연구 표면입니다. 직접 신호 판단을 바꾸지는 않습니다.",
        "en": "The heatmap, cycle, and detail panels below are research surfaces used to validate the upper decision-support boards and add context. They do not directly change the signal label.",
    },
    "flow_col_profile": {"ko": "프로필", "en": "Profile"},
    "flow_col_state": {"ko": "수급 상태", "en": "Flow state"},
    "flow_col_score": {"ko": "수급 σ", "en": "Flow sigma"},
    "flow_col_adjustment": {"ko": "신호 변화", "en": "Signal change"},
    "flow_col_foreign": {"ko": "외국인", "en": "Foreign"},
    "flow_col_institutional": {"ko": "기관", "en": "Institutional"},
    "flow_col_retail": {"ko": "개인", "en": "Retail"},
    "flow_col_latest": {"ko": "최근 비율", "en": "Latest ratio"},
    "flow_cue_sigma": {"ko": "{state} ({z:+.1f}σ)", "en": "{state} ({z:+.1f}σ)"},
    "flow_sigma_explainer_toggle": {"ko": "수급 상태 σ 계산 설명", "en": "How flow-state sigma is calculated"},
    "flow_sigma_explainer_body": {
        "ko": "`수급 상태 σ`의 기본 재료는 투자자 주체별 `net_flow_ratio` 시계열입니다.\n\n- 계산식: `(단기 평균 - 장기 평균) / 장기 표준편차`\n- 단기 평균: 최근 `{short_window}`개 관측치 평균\n- 장기 평균/표준편차: 최근 `{long_window}`개 관측치 기준\n- 개별 투자자 판정: `z >= 0.5`면 수급 우호, `z <= -0.5`면 수급 역풍, 그 사이는 수급 중립\n- 화면에 보이는 투자자별 cue와 `σ` 표기는 같은 규칙을 사용합니다.\n- 따라서 `Latest ratio`가 플러스여도 최근 모멘텀이 장기 기준보다 약하면 중립 또는 역풍으로 표시될 수 있습니다.",
        "en": "`Flow-state sigma` starts from each participant group's `net_flow_ratio` time series.\n\n- Formula: `(short mean - long mean) / long standard deviation`\n- Short mean: average of the latest `{short_window}` observations\n- Long mean/std: computed over the latest `{long_window}` observations\n- Per-participant classification: `z >= 0.5` is supportive, `z <= -0.5` is adverse, and values in between are neutral\n- The participant cues and sigma labels shown in the UI use the same rule.\n- So a positive `Latest ratio` can still be shown as neutral or adverse if recent momentum is weak versus the longer baseline.",
    },
}

_ACTION_LABELS: dict[str, dict[UiLocale, str]] = {
    "Strong Buy": {"ko": "[+] 강한 후보 (Strong Buy)", "en": "[+] Strong candidate"},
    "Watch": {"ko": "[~] 관망 (Watch)", "en": "[~] Watch"},
    "Hold": {"ko": "[=] 유지 (Hold)", "en": "[=] Hold"},
    "Avoid": {"ko": "[x] 회피 (Avoid)", "en": "[x] Avoid"},
    "N/A": {"ko": "[-] N/A", "en": "[-] N/A"},
}

_DECISION_LABELS: dict[str, dict[str, dict[UiLocale, str]]] = {
    "held": {
        "Strong Buy": {"ko": "추가 검토 후보", "en": "Add-review candidate"},
        "Watch": {"ko": "유지 / 모니터링", "en": "Hold / monitor"},
        "Hold": {"ko": "비중 축소 / 교체", "en": "Reduce / rotate"},
        "Avoid": {"ko": "축소 / 이탈 검토", "en": "Reduce / exit review"},
        "N/A": {"ko": "데이터 확인", "en": "Data check"},
    },
    "new": {
        "Strong Buy": {"ko": "신규 검토 후보", "en": "New review candidate"},
        "Watch": {"ko": "관찰 후보", "en": "Watch candidate"},
        "Hold": {"ko": "신규 검토 보류", "en": "Fresh review deferred"},
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

_FLOW_PROFILE_LABELS: dict[str, dict[UiLocale, str]] = {
    "foreign_lead": {"ko": "외국인 주도형", "en": "Foreign-led"},
    "institutional_confirmation": {"ko": "기관 확인형", "en": "Institutional confirmation"},
    "contrarian_retail": {"ko": "개인 역지표형", "en": "Contrarian retail"},
}

_FLOW_PROFILE_SIGMA_SUBJECT_LABELS: dict[str, dict[UiLocale, str]] = {
    "foreign_lead": {"ko": "외국인 σ", "en": "foreign sigma"},
    "institutional_confirmation": {"ko": "기관 σ", "en": "institutional sigma"},
    "contrarian_retail": {"ko": "개인 역지표 σ", "en": "contrarian retail sigma"},
}

_FLOW_STATE_LABELS: dict[str, dict[UiLocale, str]] = {
    "supportive": {"ko": "수급 우호", "en": "Supportive"},
    "neutral": {"ko": "수급 중립", "en": "Neutral"},
    "adverse": {"ko": "수급 역풍", "en": "Adverse"},
    "unavailable": {"ko": "실험 데이터 없음", "en": "Unavailable"},
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


def get_flow_profile_label(profile: str, locale: str | None = DEFAULT_UI_LOCALE) -> str:
    normalized_locale = normalize_locale(locale)
    return _FLOW_PROFILE_LABELS.get(profile, {}).get(normalized_locale, profile)


def get_flow_sigma_subject_label(profile: str, locale: str | None = DEFAULT_UI_LOCALE) -> str:
    normalized_locale = normalize_locale(locale)
    normalized_profile = str(profile or "").strip().lower()
    return _FLOW_PROFILE_SIGMA_SUBJECT_LABELS.get(normalized_profile, {}).get(
        normalized_locale,
        normalized_profile or profile,
    )


def get_flow_state_label(state: str, locale: str | None = DEFAULT_UI_LOCALE) -> str:
    normalized_locale = normalize_locale(locale)
    return _FLOW_STATE_LABELS.get(state, {}).get(normalized_locale, state)


def format_flow_cue_label(
    state: str,
    zscore: float | int | None,
    locale: str | None = DEFAULT_UI_LOCALE,
) -> str:
    normalized_locale = normalize_locale(locale)
    state_label = get_flow_state_label(state, normalized_locale)
    try:
        numeric_z = float(zscore) if zscore is not None else None
    except (TypeError, ValueError):
        numeric_z = None
    if numeric_z is None:
        return state_label
    if numeric_z != numeric_z:
        return state_label
    return get_ui_text("flow_cue_sigma", normalized_locale, state=state_label, z=numeric_z)


def get_flow_reference_only_note(
    status: str,
    fresh: bool,
    detail: Mapping[str, object] | None = None,
    locale: str | None = DEFAULT_UI_LOCALE,
) -> str:
    normalized_status = str(status or "").strip().upper()
    normalized_locale = normalize_locale(locale)
    payload = dict(detail or {})

    if normalized_status == "SAMPLE":
        return ""
    if fresh and not bool(payload.get("bootstrap_partial_preview")) and not bool(payload.get("warehouse_write_skipped")):
        return ""
    if bool(payload.get("warehouse_write_skipped")):
        return get_ui_text("flow_reference_only_transient", normalized_locale)
    if bool(payload.get("bootstrap_partial_preview")):
        return get_ui_text("flow_reference_only_partial", normalized_locale)
    if normalized_status in {"LIVE", "CACHED"} and not fresh:
        return get_ui_text("flow_reference_only_stale", normalized_locale)
    return ""


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
    "FLOW_PROFILE_IDS",
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
    "get_flow_profile_label",
    "get_flow_sigma_subject_label",
    "get_flow_reference_only_note",
    "get_flow_state_label",
    "format_flow_cue_label",
    "get_heatmap_palette_label",
    "get_position_mode_label",
    "get_range_preset_label",
    "get_regime_subtitle",
    "get_ui_text",
    "is_all_action_filter",
    "normalize_action_filter",
    "normalize_locale",
]
