# 2026-05-17 - Remove Low-Value Overview Change Heatmap

## Goal
- 사용자가 지적한 `상위/하위 변화` 카드형 heatmap 블록을 overview evidence에서 제거한다.
- 남는 영역은 섹터 원장, 상대강도 추이, 수출 추이 중심으로 재정렬한다.
- 점수 계산, 정렬, taxonomy, 차트 데이터 계약은 변경하지 않는다.

## Checklist
- [x] screenshot 구간의 렌더 경로 확인
- [x] heatmap 렌더러와 overview 호출 제거
- [x] evidence column layout 및 copy 정리
- [x] 관련 테스트 계약 갱신
- [x] focused verification 및 screenshot smoke 실행
- [x] 결과 기록

## Review
- Changed:
  `상위/하위 변화` 카드형 heatmap renderer와 overview evidence 내 호출을 제거했다.
  수출 지표가 없을 때의 evidence layout을 `원장 + 상대강도 추이` 2열로 바꿨다.
  evidence header copy에서 제거된 보조 수익률 변화 언급을 걷어냈다.
- Verification:
  `python -m py_compile src\ui\panels.py src\ui\css.py tests\test_ui_components.py` -> passed
  `python -m pytest -q tests/test_ui_components.py -k "toss_overview_dashboard or render_toss_overview_dashboard_gives_export_chart_full_width" --basetemp "$env:TEMP\pytest-remove-overview-heatmap"` -> `3 passed, 101 deselected`
  `python -m pytest -q tests/test_ui_theme.py tests/test_ui_components.py --basetemp "$env:TEMP\pytest-remove-overview-heatmap-ui"` -> `121 passed`
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\overview-without-change-heatmap-desktop-20260517.png --app app.py --port 8552 --debug-port 9274 --url http://127.0.0.1:8552 --width 1440 --height 1100 --timeout 120 --min-text-len 300` -> passed, screenshot bytes `204225`
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\overview-without-change-heatmap-mobile-20260517.png --app app.py --port 8553 --debug-port 9275 --url http://127.0.0.1:8553 --width 390 --height 900 --timeout 120 --min-text-len 240 --mobile-user-agent` -> passed, screenshot bytes `38511`
  Browser DOM absence check -> `hasChangeTitle: false`, `heatmapNodes: 0`
  `git diff --check -- src\ui\panels.py src\ui\css.py tests\test_ui_components.py tasks\todo.md tasks\lessons.md` -> passed with CRLF warnings only

# 2026-05-17 - Designer Spacing Correction

## Goal
- 이전 spacing pass 이후에도 붙어 보이는 overview 섹션 리듬을 실제 DOM 측정 기준으로 재보정한다.
- 단순 토큰 확대가 아니라 top workbench / controls / evidence를 분리된 작업 표면으로 재구성한다.
- 계산, 라우팅, taxonomy 선택 상태, 표/차트 데이터 계약은 변경하지 않는다.

## Frontend Notes
- Visual thesis: 하나의 압축 카드 안에 모든 것이 붙어 있는 느낌을 줄이고, `분류/판단`, `조회/필터`, `증거`의 세 작업 표면을 명확히 분리한다.
- Spacing thesis: major section gap은 28px 이상, related internal gap은 16~20px 기준으로 맞춘다.
- Measured issue: desktop DOM에서 `overview-review-candidates` -> `overview-command-surface` gap이 1px, `overview-workbench-header` -> `overview-taxonomy-surface` gap이 16px였다.

## Checklist
- [x] 실제 렌더 DOM gap 측정
- [x] overview top/control/evidence surface 구조 분리
- [x] major/minor spacing token과 CSS 재보정
- [x] desktop/mobile gap 재측정 및 screenshot smoke
- [x] focused tests 실행
- [x] 결과 기록

## Review
- Changed:
  overview workbench를 `분류/판단`, `조회/필터`, `증거` 3개 표면으로 분리했다.
  `section_gap` / `section_gap_tight` / `section_gap_loose` 토큰을 다시 보정하고, 모바일 override가 major gap을 다시 줄이지 않도록 수정했다.
  후보 카드, market command, form, bordered Streamlit wrapper의 vertical rhythm을 같은 토큰 기준으로 정리했다.
- DOM Evidence:
  desktop `overview-workbench-header -> overview-taxonomy-surface`: 16px -> 30px
  desktop `overview-review-candidates -> overview-command-surface`: 1px -> 60px
  mobile `overview-workbench-header -> overview-taxonomy-surface`: 31px
  mobile `overview-review-candidates -> overview-command-surface`: 60px
  desktop/mobile horizontal overflow: false
- Verification:
  `python -m py_compile config\theme.py src\ui\css.py src\ui\panels.py tests\test_ui_theme.py tests\test_ui_components.py` -> passed
  `python -m pytest -q tests/test_ui_theme.py tests/test_ui_components.py --basetemp "$env:TEMP\pytest-designer-spacing-ui"` -> `121 passed`
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\designer-spacing-desktop-20260517.png --app app.py --port 8548 --debug-port 9268 --url http://127.0.0.1:8548 --width 1440 --height 1100 --timeout 120 --min-text-len 300` -> passed, screenshot bytes `204225`
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\designer-spacing-mobile-20260517.png --app app.py --port 8549 --debug-port 9269 --url http://127.0.0.1:8549 --width 390 --height 900 --timeout 120 --min-text-len 240 --mobile-user-agent` -> passed, screenshot bytes `38511`
  `git diff --check -- config\theme.py src\ui\css.py src\ui\panels.py tests\test_ui_theme.py tasks\todo.md tasks\lessons.md` -> passed with CRLF warnings only
  `rg -n "[ \t]$" config\theme.py src\ui\css.py src\ui\panels.py tests\test_ui_theme.py tasks\todo.md tasks\lessons.md` -> no trailing whitespace
- Dev Server:
  Running at `http://127.0.0.1:8550`, health `ok`.

# 2026-05-17 - UX Section Spacing Audit

## Goal
- `design-taste-frontend` 기준으로 현재 Streamlit UI의 섹션 간격을 점검한다.
- 특히 overview workbench, evidence, taxonomy, 후보/시장/차트 섹션 사이의 과밀한 리듬을 완화한다.
- 데이터 계산, 라우팅, taxonomy 표시 계약은 변경하지 않는다.

## Checklist
- [x] current UI/CSS render path 확인
- [x] 섹션 간격 토큰과 overview-specific spacing 조정
- [x] CSS 계약 테스트 갱신
- [x] focused verification 및 visual smoke 실행
- [x] 결과 기록

## Review
- Changed:
  `section_gap` 계열 토큰을 넓혀 페이지/패널/overview 섹션의 기본 vertical rhythm을 완화했다.
  `overview` workbench, taxonomy, 후보 카드, 시장/조회, evidence, 차트 제목/보조 섹션의 margin/gap을 같은 토큰 체계로 맞췄다.
  모바일 override도 기존 0.2~0.5rem대 간격 대신 `section_gap_tight` 기준으로 정리했다.
- Verification:
  `python -m py_compile config\theme.py src\ui\css.py tests\test_ui_theme.py` -> passed
  `python -m pytest -q tests/test_ui_theme.py tests/test_ui_components.py --basetemp "$env:TEMP\pytest-spacing-ui-2"` -> `121 passed`
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\ux-section-spacing-desktop-20260517.png --app app.py --port 8540 --debug-port 9260 --url http://127.0.0.1:8540 --width 1440 --height 1100 --timeout 120 --min-text-len 300` -> passed, screenshot bytes `216821`
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\ux-section-spacing-mobile-20260517.png --app app.py --port 8541 --debug-port 9261 --url http://127.0.0.1:8541 --width 390 --height 900 --timeout 120 --min-text-len 240 --mobile-user-agent` -> passed, screenshot bytes `40060`
  `git diff --check -- config\theme.py src\ui\css.py tests\test_ui_theme.py tasks\todo.md` -> passed with CRLF warnings only

# 2026-05-17 - Overview Evidence Readability Pass

## Goal
- `섹터 원장과 차트` 구간의 과밀한 3열 배치를 줄여 표와 차트 가독성을 회복한다.
- 같은 overview evidence container 안에서 확인 가능하게 유지한다.
- 신호 계산, 정렬, 수출 지표 데이터 계약은 변경하지 않는다.

## Frontend Notes
- Visual thesis: 압축된 원장형 콘솔은 유지하되, 읽어야 하는 차트는 숨 쉴 수 있는 폭을 준다.
- Content plan: 원장과 상대강도는 첫 행의 주 작업 영역, 수출 추이는 같은 섹션의 보조 행으로 분리한다.
- Interaction thesis: 스크롤/hover 동작은 유지하고, x축과 범례가 겹치지 않도록 차트 tick 밀도만 낮춘다.

## Checklist
- [x] screenshot 기반 과밀 원인 확인
- [x] evidence layout 3열 압축 해소
- [x] export chart tick/marker density 조정
- [x] focused verification 및 visual smoke 실행
- [x] 결과 기록

## Review
- Changed:
  수출 지표가 있는 `섹터 원장과 차트` 구간을 3열 단일 행에서 `원장 + 상대강도` 2열 행과 `수출 추이` 전체폭 보조 행으로 바꿨다.
  수출 차트 기본 표시 기간은 18개월로 줄이고, 12개월 초과 시 월 tick을 2개월 간격으로 낮췄다.
  marker 크기와 선 굵기를 조금 줄여 점/라벨 밀도를 낮췄고, 차트 높이는 360px로 늘렸다.
- Verification:
  `python -m py_compile src\ui\panels.py tests\test_ui_components.py` -> passed
  `python -m pytest -q tests/test_ui_components.py -k "toss_overview_dashboard or sector_export_trend_figure" --basetemp "$env:TEMP\pytest-evidence-readability-components"` -> `4 passed`
  `python -m pytest -q tests/test_ui_components.py --basetemp "$env:TEMP\pytest-evidence-readability-ui-components"` -> `104 passed`
  `git diff --check -- src\ui\panels.py tests\test_ui_components.py tasks\todo.md` -> passed with CRLF warnings only
  `rg -n "[ \t]$" src\ui\panels.py tests\test_ui_components.py tasks\todo.md` -> no trailing whitespace
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\overview-evidence-readability-20260517.png --app app.py --port 8534 --debug-port 9254 --url http://127.0.0.1:8534 --width 1440 --height 1200 --timeout 120 --min-text-len 300` -> passed, screenshot bytes `247177`, DOM text length `3664`
  Running dev server at `http://127.0.0.1:8533`, health `ok`

# 2026-05-17 - Unified Dashboard Workbench Rebuild

## Goal
- 중복되는 dashboard overview 화면 블록을 줄이고, 핵심 판단 정보를 첫 화면에서 최대한 확인하게 만든다.
- 기존 신호 계산, taxonomy schema, 데이터 수집/갱신 라우팅은 변경하지 않는다.
- 별도 페이지는 상세 분석용으로 유지하되, `overview`는 한 화면 워크벤치 역할을 하게 한다.

## Frontend Notes
- Visual thesis: 운용 데스크의 조밀한 리서치 터미널처럼, 무광 표면과 얇은 구분선으로 한 화면에서 분류축, 후보, 시장 상태, 원장, 차트를 동시에 스캔하게 한다.
- Content plan: 상단 workbench는 taxonomy/검토 후보/시장 상태/조회/필터를 합치고, 하단은 섹터 원장과 상대강도·수출·변화 차트를 한 압축 grid로 배치한다.
- Interaction thesis: taxonomy 레이어 전환과 기간/정렬 필터는 같은 작업 표면 안에 고정하고, 후보 카드 hover와 표/차트 스크롤은 기존보다 작은 이동으로 유지한다.

## Checklist
- [x] current overview/navigation render path 확인
- [x] one-screen workbench 계획 기록
- [x] 중복 컨테이너를 통합한 overview workbench 구현
- [x] dense dashboard CSS 계약 갱신
- [x] focused tests 및 visual smoke 검증
- [x] 결과 기록

## Review
- Changed:
  `signals`, `constituents`, `flow` top-level dashboard entries를 숨기고 old page id는 `overview`로 정규화했다.
  `overview`는 unified workbench header, taxonomy/후보, 시장/조회, 필터, 섹터 원장, 상대강도/수출/히트맵을 두 개의 조밀한 container 안에서 한 흐름으로 보게 재구성했다.
  기존 상세 `research`와 KR `quality` 화면은 유지했다.
- Verification:
  `python -m py_compile app.py src\dashboard\tabs.py src\ui\panels.py src\ui\css.py tests\test_dashboard_tabs.py tests\test_ui_components.py tests\test_ui_theme.py` -> passed
  `python -m pytest -q tests/test_dashboard_tabs.py -k "dashboard_page_options or normalize_dashboard_page_id or resolve_dashboard_page_title or render_dashboard_tabs_routes" --basetemp "$env:TEMP\pytest-unified-dashboard-tabs"` -> `7 passed`
  `python -m pytest -q tests/test_ui_components.py -k "toss_overview_dashboard or overview_taxonomy_surface or overview_sector_table" --basetemp "$env:TEMP\pytest-unified-dashboard-components"` -> `5 passed`
  `python -m pytest -q tests/test_ui_theme.py -k "inject_css_includes_new_dashboard_layout_classes" --basetemp "$env:TEMP\pytest-unified-dashboard-theme"` -> `1 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py tests/test_ui_components.py tests/test_ui_theme.py --basetemp "$env:TEMP\pytest-unified-dashboard-ui"` -> `162 passed`
  `git diff --check -- app.py src\dashboard\tabs.py src\ui\panels.py src\ui\css.py tests\test_dashboard_tabs.py tests\test_ui_components.py tests\test_ui_theme.py tasks\todo.md` -> passed with CRLF warnings only
  `rg -n "[ \t]$" app.py src\dashboard\tabs.py src\ui\panels.py src\ui\css.py tests\test_dashboard_tabs.py tests\test_ui_components.py tests\test_ui_theme.py tasks\todo.md` -> no trailing whitespace
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\unified-dashboard-workbench-20260517.png --app app.py --port 8532 --debug-port 9252 --url http://127.0.0.1:8532 --width 1440 --height 1024 --timeout 120 --min-text-len 300` -> passed, screenshot bytes `215004`, DOM text length `3563`
  Local dev server started at `http://127.0.0.1:8533`, health `ok`

# 2026-05-17 - Theme Taxonomy First Page Recomposition

## Goal
- 첫 화면을 `theme_taxonomy`의 기본산업, 크로스테마, 상품테마 축이 먼저 보이도록 재구성한다.
- 기존 신호 계산, `sector_map.yml`, taxonomy schema, warehouse sync 계약은 변경하지 않는다.
- 검토 후보, 표, 히트맵은 taxonomy 표시명을 기본으로 유지하되 런타임 섹터명은 추적 문맥으로 보존한다.

## Frontend Notes
- Visual thesis: 운용 콘솔처럼 차분하고 조밀하게, 첫 화면은 시장 카드보다 분류축 지도와 후보 판단이 먼저 읽히는 taxonomy workbench로 만든다.
- Content plan: taxonomy 상태/분류축 지도, 매수·매도 검토 후보, 시장·조회·필터, 섹터 원장, 추이/히트맵 순서로 재배치한다.
- Interaction thesis: taxonomy/기존 섹터 레이어 전환은 유지하고, 후보 카드 hover와 표/차트 스크롤 반응은 기존 속도로 유지한다.

## Checklist
- [x] current overview/taxonomy render path 확인
- [x] taxonomy-first 화면 계획 기록
- [x] taxonomy 요약/분류축 지도 렌더러 추가
- [x] overview 섹션 순서 재배치
- [x] CSS와 테스트 계약 갱신
- [x] focused verification 실행
- [x] 결과 기록

## Review
- Changed:
  overview 첫 화면 상단을 `Theme Taxonomy` workbench로 재배치했다.
  taxonomy 버전, 커버리지, 기본산업/크로스테마/상품테마 수, 현재 표시 레이어를 같은 표면에서 보여준다.
  기존 숨김 expander 문맥은 overview에서는 제거하고, 상위 신호 분류축 지도와 매수/매도 검토 후보를 taxonomy 표면 아래에 바로 배치했다.
  시장/조회/필터는 후보 이후의 증거·조작 영역으로 낮췄고, 계산 로직과 taxonomy schema는 변경하지 않았다.
- Verification:
  `python -m py_compile src\ui\panels.py src\ui\css.py tests\test_ui_components.py tests\test_ui_theme.py` -> passed
  `python -m pytest -q tests/test_ui_components.py -k "overview_taxonomy or toss_overview_dashboard" --basetemp "$env:TEMP\pytest-taxonomy-page-recomposition-components"` -> `3 passed, 100 deselected`
  `python -m pytest -q tests/test_ui_theme.py -k "inject_css_includes_new_dashboard_layout_classes" --basetemp "$env:TEMP\pytest-taxonomy-page-recomposition-theme"` -> `1 passed, 16 deselected`
  `python -m pytest -q tests/test_ui_components.py tests/test_ui_theme.py --basetemp "$env:TEMP\pytest-taxonomy-page-recomposition-ui"` -> `120 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py tests/test_theme_taxonomy_adapter.py --basetemp "$env:TEMP\pytest-taxonomy-page-recomposition-tabs-adapter"` -> `45 passed`
  `git diff --check -- src\ui\panels.py src\ui\css.py tests\test_ui_components.py tests\test_ui_theme.py tasks\todo.md` -> passed with CRLF warnings only
  `rg -n "[ \t]$" src\ui\panels.py src\ui\css.py tests\test_ui_components.py tests\test_ui_theme.py` -> no trailing whitespace
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\theme-taxonomy-page-recomposition-20260517.png --app app.py --port 8528 --debug-port 9248 --url http://127.0.0.1:8528 --width 1440 --height 1024 --timeout 120 --min-text-len 300` -> passed, screenshot bytes `206678`
  Local dev server started at `http://127.0.0.1:8529`, health `ok`

# 2026-05-17 - Overview Sector Table Intrinsic Width

## Goal
- `섹터 모멘텀 & 상대강도` 표의 각 컬럼을 내용 최대 길이 기준으로만 차지하게 한다.
- full-width 섹션 배치는 유지하되, 표 자체는 불필요하게 패널 전체 폭으로 늘리지 않는다.
- 기존 신호 계산, 정렬, 차트 데이터는 변경하지 않는다.

## Frontend Notes
- Visual thesis: 운영 대시보드 표는 원장처럼 조밀하게 정렬하고, 남는 폭은 빈 데이터 그리드가 아니라 주변 여백으로 남긴다.
- Content plan: 표 컬럼은 순위, 섹터, 모멘텀, 상대강도, 3M의 실제 최대 텍스트 폭만 사용한다.
- Interaction thesis: 기존 스크롤/차트 상호작용은 유지하고, 표는 intrinsic width로 더 빠르게 스캔되게 한다.

## Checklist
- [x] fixed percentage column CSS 확인
- [x] intrinsic table/wrapper sizing으로 변경
- [x] CSS 계약 테스트 갱신
- [x] focused verification 실행
- [x] 결과 기록

## Review
- Changed:
  `.overview-sector-table`을 `table-layout: fixed` / `width: 100%`에서 `table-layout: auto` / `width: max-content`로 변경했다.
  `.overview-sector-table-wrap`도 `width: fit-content`로 줄여 표 border가 내용 폭만 감싸게 했다.
  숫자/섹터 컬럼은 `white-space: nowrap`을 유지해 각 컬럼이 가장 긴 셀 길이에 맞춰진다.
- Verification:
  `python -m py_compile src\ui\panels.py src\ui\css.py tests\test_ui_components.py tests\test_ui_theme.py` -> passed
  `python -m pytest -q tests/test_ui_components.py -k "overview_sector_table or toss_overview_dashboard_reflows" --basetemp "$env:TEMP\pytest-overview-table-intrinsic-components"` -> `3 passed, 98 deselected`
  `python -m pytest -q tests/test_ui_theme.py -k "inject_css_includes_new_dashboard_layout_classes" --basetemp "$env:TEMP\pytest-overview-table-intrinsic-theme"` -> `1 passed, 16 deselected`
  `python -m pytest -q tests/test_ui_components.py tests/test_ui_theme.py --basetemp "$env:TEMP\pytest-overview-table-intrinsic-regression"` -> `118 passed`
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\overview-table-intrinsic-width-20260517.png --app app.py --port 8526 --debug-port 9246 --url http://127.0.0.1:8526 --width 1920 --height 1080 --timeout 120 --min-text-len 300` -> passed
  CDP DOM check at 1920px -> panel `1390px`, table/wrap `524px`, `cellWidths=[38,317,49,59,61]`, `horizontalOverflow=false`, `tableLayout=auto`, `containsNan=false`

# 2026-05-17 - Overview Sector Momentum Table Reflow

## Goal
- `섹터 모멘텀 & 상대강도` 표를 가로 스크롤 없이 한 화면 폭에서 읽히게 재배치한다.
- 기존 신호 계산, 정렬, 차트 데이터는 변경하지 않는다.
- 데스크톱에서는 표를 충분한 폭의 독립 섹션으로 올리고, 차트들은 아래에서 균형 있게 이어지게 한다.

## Frontend Notes
- Visual thesis: 데이터 작업대처럼 차분하고 조밀하되, 핵심 표는 좁은 보조 컬럼이 아니라 독립 원장으로 읽히게 한다.
- Content plan: 필터/조회 컨텍스트, 핵심 섹터 표, 상대강도·수출 추이 차트, 변화 히트맵 순서로 배치한다.
- Interaction thesis: 기존 필터와 차트 상호작용은 유지하고, 표는 이름 줄바꿈과 고정 숫자 폭으로 가로 이동 없이 스캔하게 한다.

## Checklist
- [x] 현재 overview 섹션 구조와 테이블 CSS 계약 확인
- [x] 표를 독립 섹션으로 재배치하고 테이블 컬럼 폭/줄바꿈 보정
- [x] 회귀 테스트로 섹션 순서와 CSS 계약 고정
- [x] focused verification 실행
- [x] 결과 기록

## Review
- Changed:
  `섹터 모멘텀 & 상대강도` 표를 기존 1.08/2.08 좌측 컬럼에서 독립 full-width 섹션으로 올렸다.
  하단 차트는 수출 지표가 있으면 상대강도/수출 추이를 2열로, 수출 지표가 없으면 상대강도/히트맵을 2열로 배치한다.
  표는 `<colgroup>` 기반 fixed layout으로 바꾸고 섹터명 컬럼만 줄바꿈되게 해 가로 스크롤을 없앴다.
  혼합 taxonomy row에서 optional 값이 `nan` 보조 문구로 노출되는 케이스도 방지했다.
- Verification:
  `python -m py_compile src\ui\panels.py src\ui\css.py tests\test_ui_components.py tests\test_ui_theme.py` -> passed
  `python -m pytest -q tests/test_ui_components.py -k "overview_sector_table or toss_overview_dashboard_reflows" --basetemp "$env:TEMP\pytest-overview-table-reflow-components-2"` -> `3 passed, 98 deselected`
  `python -m pytest -q tests/test_ui_theme.py -k "inject_css_includes_new_dashboard_layout_classes" --basetemp "$env:TEMP\pytest-overview-table-reflow-theme-2"` -> `1 passed, 16 deselected`
  `python -m pytest -q tests/test_ui_components.py tests/test_ui_theme.py --basetemp "$env:TEMP\pytest-overview-table-reflow-regression-2"` -> `118 passed`
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\overview-table-reflow-20260517.png --app app.py --port 8523 --debug-port 9243 --url http://127.0.0.1:8523 --width 1920 --height 1080 --timeout 120 --min-text-len 300` -> passed
  CDP DOM check at 1920px -> `.overview-sector-table-wrap` `clientWidth=1378`, `scrollWidth=1378`, `horizontalOverflow=false`, `tableLayout=fixed`, `containsNan=false`
  `git diff --check -- src\ui\panels.py src\ui\css.py tests\test_ui_components.py tests\test_ui_theme.py tasks\todo.md` -> passed with CRLF warnings only

# 2026-05-17 - DuckDB Status Probe Connection Conflict

## Goal
- 상대강도 분석 페이지 로딩 중 `same database file with a different configuration` 오류를 제거한다.
- DuckDB 읽기 캐시는 read-only 우선 정책을 유지한다.
- 같은 Python 프로세스에 이미 read/write 연결이 열린 경우 상태 조회가 예외로 중단되지 않게 한다.

## Checklist
- [x] 연결 생성 지점과 상태 probe 경로 확인
- [x] 같은 프로세스 read/write 연결과 read-only 캐시 충돌 원인 재현
- [x] 읽기 캐시에 동일 설정 충돌 폴백 추가
- [x] 회귀 테스트 추가
- [x] focused verification
- [x] 결과 기록

## Review
- Root cause:
  DuckDB는 같은 Python 프로세스에서 기본 read/write 연결이 열린 동안 같은 DB 파일을 `read_only=True`로 다시 열면 `same database file with a different configuration` 예외를 낸다.
- Changed:
  읽기 캐시는 기존처럼 `read_only=True`를 우선 사용한다.
  동일 DB/다른 설정 충돌일 때만 현재 프로세스의 기존 연결 설정에 맞춰 read/write 연결로 폴백한다.
  캐시 키에 DB 경로를 추가해 테스트/런타임에서 `WAREHOUSE_PATH`가 바뀌어도 이전 연결을 재사용하지 않게 했다.
- Verification:
  `.venv\Scripts\python -m py_compile src\data_sources\warehouse.py tests\test_warehouse_cli.py` -> passed
  `pytest -q tests/test_warehouse_cli.py -k "artifact_key_survives" --basetemp "$env:TEMP\pytest-duckdb-conflict-artifact-conda"` -> `2 passed, 16 deselected`
  `pytest -q tests/test_warehouse_cli.py --basetemp "$env:TEMP\pytest-duckdb-conflict-warehouse-cli"` -> `18 passed`
  `.venv\Scripts\python` same-process read/write conflict probe calling `warehouse.probe_dataset_mode("macro_data")` -> returned `SAMPLE` without exception

# 2026-05-16 - Sidebar Reopen Control Fix

## Goal
- 사이드바를 접은 뒤 다시 여는 버튼이 사라지는 문제를 고친다.
- Streamlit header chrome은 계속 시각적으로 숨기되 collapsed sidebar control은 보이게 한다.
- Sidebar navigation group의 material ligature가 `id_`처럼 잘려 보이지 않게 한다.

## Checklist
- [x] 원인 확인: `stHeader display:none`이 collapsed control까지 제거할 수 있음
- [x] header를 0-height transparent layer로 바꾸고 collapsed control selector를 명시적으로 복구
- [x] Deploy/toolbar chrome 재노출 방지
- [x] sidebar nav 내부 material icon ligature 숨김
- [x] `stSidebarCollapseButton`/`stBaseButton-headerNoPadding` visibility까지 복구
- [x] collapsed 상태(`aria-expanded=false`)를 클릭 가능한 rail 상태로 보정
- [x] focused verification

## Review
- Root cause:
  Streamlit의 sidebar reopen control은 header chrome 계층에 붙어 있는데, 기존 CSS가 `stHeader`를 `display:none`으로 제거해 접은 뒤 다시 펼칠 버튼도 사라질 수 있었다.
  실제 DOM에서는 접기/펼치기 버튼이 `stSidebarCollapseButton` 아래의 `stBaseButton-headerNoPadding`으로 생성되고, 이 노드가 `visibility:hidden` 상태를 가질 수 있었다.
- Changed:
  `stHeader`는 0-height transparent layer로 유지하고, `stSidebarCollapsedControl`/`collapsedControl`만 fixed button으로 보이게 했다.
  접기/펼치기 버튼 부모와 실제 `headerNoPadding` 버튼을 함께 visible로 복구했다.
  접힌 상태에서 sidebar가 완전히 offscreen으로 빠지지 않고 4.25rem rail만 남기도록 `aria-expanded=false` 상태 보정을 추가했다.
  header를 살리면서 다시 노출된 Deploy/toolbar chrome은 별도 selector로 숨겼다.
  nav group의 Streamlit material icon ligature는 렌더 실패 시 텍스트 조각으로 보이므로 sidebar nav 내부에서 숨겼다.
- Verification:
  `python -m py_compile src\ui\css.py tests\test_ui_theme.py` -> passed
  `python -m pytest -q tests/test_ui_theme.py -k "inject_css_reflects_table_tokens or inject_css_includes_new_dashboard_layout_classes" --basetemp "$env:TEMP\pytest-sidebar-reopen-theme-2"` -> `2 passed, 15 deselected`
  `python -m pytest -q tests/test_dashboard_tabs.py tests/test_ui_theme.py --basetemp "$env:TEMP\pytest-sidebar-reopen-final"` -> `56 passed`
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\sidebar-reopen-control-20260516.png --app app.py --port 8519 --debug-port 9239 --url http://127.0.0.1:8519 --width 1440 --height 1024 --timeout 120 --min-text-len 300` -> passed
  `git diff --check -- src/ui/css.py tests/test_ui_theme.py tasks/todo.md` -> passed with LF/CRLF warnings only

# 2026-05-16 - Ralph Sidebar Operations Panel Redesign

## Goal
- Deep-interview spec에 따라 사이드바를 Streamlit 기본 패널이 아니라 투자 운영 패널처럼 재구성한다.
- `expand_more` 텍스트/아이콘 겹침을 제거한다.
- 데이터 갱신 동작과 disabled 조건은 유지한다.

## Checklist
- [x] deep-interview spec/context 로드 및 Ralph 상태 전환
- [x] 데이터 상태/갱신을 상단 운영 영역으로 이동
- [x] 분석 기준/수급 해석/모델 파라미터를 보조 영역으로 재배치
- [x] Streamlit material icon/select overlap CSS 보강
- [x] focused tests 및 visual verification
- [x] architect/deslop/re-verification

## Review
- Changed:
  사이드바를 `OPERATIONS / KR 섹터 콘솔`과 `데이터 운용` 패널 중심으로 재구성했다.
  시장/매크로/수급 상태와 갱신 버튼을 사이드바 상단으로 올리고, 분석 기준/수급 해석/모델 파라미터는 보조 영역으로 낮췄다.
  `expand_more` 계열 Streamlit material icon이 텍스트로 겹치지 않도록 icon box와 select 우측 여백을 고정했다.
- Preserved:
  기존 refresh button 호출, 반환 tuple 순서, `btn_states` disabled 조건, KR-only 수급 갱신 조건은 유지했다.
- Verification:
  `python -m py_compile src\dashboard\tabs.py src\ui\css.py tests\test_dashboard_tabs.py tests\test_ui_theme.py` -> passed
  `python -m pytest -q tests/test_dashboard_tabs.py -k "sidebar_controls or sidebar_status_chip" --basetemp "$env:TEMP\pytest-sidebar-ops-tabs"` -> `3 passed, 36 deselected`
  `python -m pytest -q tests/test_ui_theme.py -k "inject_css_includes_new_dashboard_layout_classes" --basetemp "$env:TEMP\pytest-sidebar-ops-theme"` -> `1 passed, 16 deselected`
  `python -m pytest -q tests/test_dashboard_tabs.py tests/test_ui_theme.py --basetemp "$env:TEMP\pytest-sidebar-ops-regression"` -> `56 passed`
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\sidebar-ops-panel-20260516.png --app app.py --port 8517 --debug-port 9238 --url http://127.0.0.1:8517 --width 1440 --height 1024 --timeout 120 --min-text-len 300` -> passed, no visible `expand_more` overlap
  Architect review -> APPROVED / CLEAR
  Deslop pass -> no Ralph-owned masking fallback or cleanup edits needed
  Post-deslop `python -m pytest -q tests/test_dashboard_tabs.py tests/test_ui_theme.py --basetemp "$env:TEMP\pytest-sidebar-ops-post-deslop"` -> `56 passed`
  `git diff --check -- src/dashboard/tabs.py src/ui/css.py tests/test_dashboard_tabs.py tests/test_ui_theme.py tasks/todo.md` -> passed with LF/CRLF warnings only

# 2026-05-16 - Sidebar UX Pass

## Goal
- Streamlit 기본 네비게이션은 유지하면서 사이드바의 작업 흐름을 더 잘 스캔되게 만든다.
- 설정, 수급 프로필, 데이터 갱신을 명확한 그룹으로 분리한다.
- 데이터/점수 산식/라우팅은 변경하지 않는다.

## Checklist
- [x] 현재 사이드바 렌더링과 CSS 계약 확인
- [x] 사이드바 정보 구조와 상태 표시 개선
- [x] 사이드바 전용 CSS 정리
- [x] focused compile/test 검증
- [x] 결과 기록

## Review
- Changed:
  사이드바 상단에 `작업공간` 블록과 시장/매크로/수급 상태 칩을 추가했다.
  기존 `실행 환경`/divider 중심 구조를 `분석 기준`, `수급 해석`, `데이터 갱신` 작업 그룹으로 재배치했다.
  고급 설정 popover는 `모델 파라미터`로 좁히고, footer label은 별도 낮은 계층으로 내렸다.
- CSS:
  사이드바 nav와 workspace 사이를 얇은 divider로 분리했다.
  상태 칩, 섹션 라벨, footer label의 density/weight/radius를 조정해 앱 UI 톤에 맞췄다.
- Verification:
  `python -m py_compile src\dashboard\tabs.py src\ui\css.py tests\test_dashboard_tabs.py tests\test_ui_theme.py` -> passed
  `python -m pytest -q tests/test_dashboard_tabs.py -k "sidebar_controls or sidebar_status_chip" --basetemp "$env:TEMP\pytest-sidebar-ux-tabs-2"` -> `3 passed, 36 deselected`
  `python -m pytest -q tests/test_ui_theme.py -k "inject_css_includes_new_dashboard_layout_classes" --basetemp "$env:TEMP\pytest-sidebar-ux-theme"` -> `1 passed, 16 deselected`
  `python -m pytest -q tests/test_dashboard_tabs.py tests/test_ui_theme.py --basetemp "$env:TEMP\pytest-sidebar-ux-regression-2"` -> `56 passed`
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\sidebar-ux-desktop-20260516-fresh.png --app app.py --port 8516 --debug-port 9237 --url http://127.0.0.1:8516 --width 1440 --height 1024 --timeout 120 --min-text-len 300` -> passed
  `git diff --check -- src/dashboard/tabs.py src/ui/css.py tests/test_dashboard_tabs.py tests/test_ui_theme.py tasks/todo.md` -> passed with LF/CRLF warnings only.

# 2026-05-16 - Autopilot Sector Momentum Tab Decision Boards

## Goal
- 승인된 autoresearch 결과를 구현해 "섹터 모멘텀" 탭을 전체 원장 중심에서 decision board 중심으로 바꾼다.
- 기존 overview proxy/candidate helper를 재사용하고 canonical action/action_policy는 변경하지 않는다.

## Checklist
- [x] autopilot context snapshot 작성
- [x] ralplan PRD/test-spec/plan 작성
- [x] signals 탭 decision board helper 구현
- [x] signals 탭 상단 렌더링 연결
- [x] focused verification
- [x] code-review clean gate

## Review
- Plan artifacts:
  `.omx/plans/prd-sector-momentum-tab-decision-upgrade-20260516.md`
  `.omx/plans/test-spec-sector-momentum-tab-decision-upgrade-20260516.md`
  `.omx/plans/ralplan-sector-momentum-tab-decision-upgrade-20260516.md`
- Implementation:
  `render_all_signals_tab()` 상단에 `의사결정 보드`를 추가했다.
  board는 기존 overview candidate projection을 재사용하며 `신규/증액 검토`, `보유 모니터링`, `보유 축소/주의`, `변곡 감시` 4개 그룹으로 나눈다.
  held `Watch`는 기본 모니터링으로 유지하고, alert/adverse flow/negative edge/high risk scalar가 있을 때만 축소/주의로 보낸다.
- Verification:
  `python -m py_compile src\ui\panels.py src\dashboard\tabs.py tests\test_ui_components.py tests\test_dashboard_tabs.py` -> passed
  `python -m pytest -q tests/test_ui_components.py -k "sector_momentum_decision_board or overview_review_candidate" --basetemp "$env:TEMP\pytest-sector-momentum-board-ui-2"` -> `8 passed, 83 deselected`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "all_signals_tab or dashboard_page" --basetemp "$env:TEMP\pytest-sector-momentum-board-tabs-2"` -> `5 passed, 33 deselected`
  `python -m pytest -q tests/test_ui_components.py tests/test_dashboard_tabs.py --basetemp "$env:TEMP\pytest-sector-momentum-board-regression"` -> `129 passed`
  code-review cycle1 -> REQUEST CHANGES, held `N/A`가 기존 overview projection의 `N/A` 제외 조건 때문에 `보유 축소/주의`에서 누락되는 문제 발견.
  cycle2 `python -m py_compile src\ui\panels.py src\dashboard\tabs.py tests\test_ui_components.py tests\test_dashboard_tabs.py` -> passed
  cycle2 `python -m pytest -q tests/test_ui_components.py -k "sector_momentum_decision_board or overview_review_candidate" --basetemp "$env:TEMP\pytest-sector-momentum-board-ui-3"` -> `9 passed, 83 deselected`
  cycle2 `python -m pytest -q tests/test_dashboard_tabs.py -k "all_signals_tab or dashboard_page" --basetemp "$env:TEMP\pytest-sector-momentum-board-tabs-3"` -> `5 passed, 33 deselected`
  cycle2 `python -m pytest -q tests/test_ui_components.py tests/test_dashboard_tabs.py --basetemp "$env:TEMP\pytest-sector-momentum-board-regression-2"` -> `130 passed`
  cycle2 post-format `python -m pytest -q tests/test_ui_components.py tests/test_dashboard_tabs.py --basetemp "$env:TEMP\pytest-sector-momentum-board-regression-3"` -> `130 passed`
  code-review cycle2 -> APPROVE, architectural status CLEAR.

# 2026-05-16 - KR 구성종목 캐시 만료 재발 방지

## Goal
- KR 구성종목 화면이 24시간 TTL 만료만으로 빈 화면을 표시하지 않게 한다.
- 일반 렌더링은 라이브 호출을 막는 기존 정책을 유지하되, 마지막 성공 캐시가 있으면 저신뢰 캐시로 표시한다.

## Checklist
- [x] 구성종목 stale-cache fallback 구현
- [x] UI 상태 라벨 추가
- [x] 회귀 테스트 추가
- [x] focused verification

## Review
- Result:
  구성종목 캐시가 24시간 TTL을 넘었더라도 같은 Strong Buy 섹터 조합의 마지막 성공 결과가 있으면 `STALE_CACHE`로 반환한다.
  일반 화면 렌더링은 여전히 라이브 KRX 호출을 하지 않지만, 만료 캐시가 있으면 빈 화면 대신 결과를 표시한다.
  강제 갱신 중 KRX/pykrx 호출이 실패해도 마지막 성공 캐시가 있으면 `STALE_CACHE`로 fallback한다.
  UI 상태 라벨은 `만료 캐시 · 갱신 권장`으로 표시한다.
- Verification:
  `python -m py_compile src\data_sources\krx_stock_screening.py src\dashboard\tabs.py tests\test_krx_stock_screening.py` -> passed
  `python -m pytest -q tests/test_krx_stock_screening.py -k "cache_only or live_failure" --basetemp "$env:TEMP\pytest-kr-screening-stale-cache"` -> `6 passed, 10 deselected`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "screening_tab" --basetemp "$env:TEMP\pytest-dashboard-screening-stale-cache"` -> `3 passed, 35 deselected`
  `python -m pytest -q tests/test_krx_stock_screening.py --basetemp "$env:TEMP\pytest-kr-screening-all"` -> `16 passed`

# 2026-05-16 - Autoresearch Sector Momentum Tab Decision Upgrade

## Goal
- "섹터 모멘텀" 탭이 전체 신호 원장이 아니라 신규 검토, 보유 축소, 변곡 감시를 먼저 답하는 투자 의사결정 화면이 되도록 개선안을 연구한다.
- 기존 `SectorSignal`과 hybrid momentum 계약을 우선 재사용하고, 검증 전 확률 수치 포장은 피한다.

## Checklist
- [x] 현재 signals/charts 탭 구현과 `SectorSignal` 필드 확인
- [x] 기존 momentum method comparison 및 probability/turning-point 연구 문서 확인
- [x] 외부 섹터 모멘텀/로테이션 근거 확인
- [x] autoresearch mission/sandbox/report 작성
- [x] architect validation artifact 작성

## Review
- Research artifact:
  `.omx/specs/autoresearch-sector-momentum-tab/report.md`
- Working conclusion:
  현재 탭의 병목은 모멘텀 산식 부재가 아니라 의사결정 표면의 우선순위다.
  1차 개선은 새 모델 없이 기존 overview proxy helper를 재사용해 `신규/증액 검토`, `보유 모니터링`, `보유 축소/주의`, `변곡 감시` 보드를 먼저 보여주는 방향이 적절하다.
- Validation:
  Architect review cycle1 -> ITERATE, required existing overview proxy reuse and tighter held Watch semantics.
  Architect review cycle2 -> ITERATE, required held Watch to remain monitor-by-default and not reduce on `trend_ok == false` alone.
  Architect review cycle3 -> APPROVED.

# 2026-05-16 - Autopilot Turning-Point Probability Dashboard Phase 1

## Goal
- 연구 문서의 "변곡점 + 상하방 확률" 방향을 구현 가능한 1차 대시보드 개선으로 반영한다.
- calibrated probability 대신 명시적 proxy 근거를 기존 overview 후보 카드에 표시한다.
- canonical action/action_policy는 변경하지 않는다.

## Checklist
- [x] autopilot context snapshot 작성
- [x] ralplan PRD/test-spec/plan 작성
- [x] architect ITERATE 반영 및 critic 승인
- [x] Ralph 구현
- [x] focused verification
- [x] code-review clean gate

## Review
- Implementation:
  overview 검토 후보 카드에 `상방`, `하방`, `엣지`, `변곡`, `복합점수`를 고정 순서로 표시한다.
  값은 calibrated probability가 아니라 deterministic proxy이며, canonical `SectorSignal.action`과 `action_policy`는 변경하지 않았다.
  `upside_proxy`, `downside_proxy`, `edge_proxy`, `turning_point_state`, bullish/bearish evidence를 candidate projection에 추가했다.
- Verification:
  `python -m py_compile src\ui\panels.py tests\test_ui_components.py` -> passed
  `python -m pytest -q tests/test_ui_components.py -k "overview_review_candidate" --basetemp "$env:TEMP\pytest-turning-point-ui-2"` -> `6 passed, 82 deselected`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "overview or decision" --basetemp "$env:TEMP\pytest-turning-point-tabs"` -> `5 passed, 32 deselected`
  `python -m pytest -q tests/test_ui_components.py --basetemp "$env:TEMP\pytest-turning-point-ui-post-deslop"` -> `88 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "overview or decision" --basetemp "$env:TEMP\pytest-turning-point-tabs-post-deslop"` -> `5 passed, 32 deselected`
  cycle2 `python -m pytest -q tests/test_ui_components.py --basetemp "$env:TEMP\pytest-turning-point-ui-final"` -> `88 passed`
  cycle2 `python -m pytest -q tests/test_dashboard_tabs.py -k "overview or decision" --basetemp "$env:TEMP\pytest-turning-point-tabs-final"` -> `5 passed, 32 deselected`
  Ralph architect verification -> APPROVED
  Code-review cycle1 -> code-reviewer APPROVE, architect WATCH
  Code-review cycle2 -> code-reviewer APPROVE, architect CLEAR
- Artifacts:
  `.omx/context/turning-point-probability-dashboard-20260516T064820Z.md`
  `.omx/plans/prd-turning-point-probability-dashboard-20260516.md`
  `.omx/plans/test-spec-turning-point-probability-dashboard-20260516.md`
  `.omx/plans/ralplan-turning-point-probability-dashboard-20260516.md`

# 2026-05-16 - Turning-Point Probability Dashboard Research

## Goal
- 섹터 로테이션 대시보드를 "상승 확률이 높은 섹터 매수, 하락 확률이 높은 섹터 축소" 관점으로 재정의한다.
- 현재 구현의 신호 구조를 확인하고, 변곡점 감지와 상하방 확률 중심의 개선안을 정리한다.

## Checklist
- [x] 현재 regime, momentum, flow, UI decision surface 확인
- [x] 섹터 로테이션/모멘텀 외부 근거 확인
- [x] 개선 프레임워크와 구현 우선순위 문서화

## Review
- Research artifact:
  `docs/research/turning-point-probability-dashboard-2026-05-16.md`
- Verification:
  로컬 구현 파일과 기존 validation 문서를 읽고, NBER/Fidelity/SSGA/AQR 자료를 교차 확인했다.

# 2026-05-16 - US Dashboard Trade Indicator RALPLAN

## Goal
- US 대시보드에서 한국 수출 데이터 대신 의미 있는 미국 수출입 지표를 표시하는 구현 계획을 확정한다.
- KR 수출 지표/섹터 수출 UI는 회귀 없이 보존한다.
- FRED YoY transform 중복 적용과 US 화면의 한국어 수출 copy 누수를 계획 단계에서 차단한다.

## Checklist
- [x] 현재 US/KR macro alias, export derivation, UI label 접점 확인
- [x] FRED US exports/imports/trade balance 후보 지표 확인
- [x] ralplan 계획서 작성
- [x] Architect 검토 반영
- [x] Critic 승인 반영

## Review
- Plan artifact:
  `.omx/plans/ralplan-us-trade-indicators-20260516.md`
- Result:
  Option A를 실행안으로 확정했다.
  US는 FRED `BOPTEXP`/`BOPTIMP` 기반 `trade_exports_yoy`/`trade_imports_yoy`를 추가하고, 전용 `trade_indicators` payload로 표시한다.
  KR의 `export_amount -> pct_change(12)` 경로와 섹터 수출 UI는 유지한다.
  섹터 수출 UI capability는 활성 macro config에 존재하는 sector `export_series_alias` 기준으로만 산정한다.
- Verification:
  Architect review -> ITERATE, capability rule 강화 요구
  Critic review -> APPROVE after revision

# 2026-05-16 - US Dashboard Trade Indicator Ralph Implementation

## Goal
- 승인된 ralplan에 따라 US 대시보드에 FRED 기반 수출입 trade pulse를 추가한다.
- KR 수출 전년비와 섹터 수출 UI는 기존 의미로 보존한다.
- US 화면에서는 한국식 섹터 수출 copy와 double-transform 위험을 제거한다.

## Checklist
- [x] US FRED trade aliases 추가
- [x] KR export growth와 US trade indicators 분리
- [x] sector export capability를 active `export_series_alias` 기준으로 계산
- [x] US/KR hero, overview status, sector export UI gating 반영
- [x] targeted tests 추가
- [x] full regression, architect verification, deslop pass 완료

## Review
- Changed files:
  `config/macro_series_us.yml`
  `src/macro/series_utils.py`
  `src/dashboard/types.py`
  `app.py`
  `src/ui/copy.py`
  `src/ui/panels.py`
  `src/dashboard/tabs.py`
  `tests/test_market_registry.py`
  `tests/test_fred.py`
  `tests/test_macro_series_utils.py`
  `tests/test_ui_components.py`
  `tests/test_integration.py`
- Result:
  US macro config에 `trade_exports_yoy`, `trade_imports_yoy`, `trade_balance`를 추가했다.
  US trade 값은 `trade_indicators`로 전달하고, `export_growth_val`은 KR `export_amount` level series의 12개월 YoY 계산에만 사용한다.
  섹터 수출 UI는 active macro config에 존재하는 sector `export_series_alias`가 있을 때만 표시한다.
  US overview/hero는 `US Exports YoY`, `US Imports YoY`, `Trade pulse` copy를 사용하며, 섹터 수출 표/차트/sort는 capability가 없으면 숨긴다.
- Verification:
  `python -m py_compile app.py src\macro\series_utils.py src\dashboard\types.py src\dashboard\tabs.py src\ui\copy.py src\ui\panels.py tests\test_market_registry.py tests\test_fred.py tests\test_macro_series_utils.py tests\test_ui_components.py tests\test_integration.py` -> passed
  `python -m pytest -q tests/test_market_registry.py tests/test_fred.py tests/test_macro_series_utils.py tests/test_ui_components.py tests/test_dashboard_tabs.py --basetemp=%TEMP%\pytest-us-trade-post-deslop-affected` -> `134 passed`
  `python -m pytest -q --basetemp=%TEMP%\pytest-us-trade-post-deslop-full` -> `625 passed`
  Architect verification -> APPROVED
  Deslop pass -> no Ralph-owned masking fallback or cleanup edits needed; post-deslop regression passed.

# 2026-05-13 - Chart Readability Improvements

## Goal
- 섹터 상대강도와 섹터별 수출 YoY 차트에서 섹터 구분을 한눈에 하게 한다.
- 월별 데이터 포인트가 어느 월인지 X축과 hover에서 바로 보이게 한다.
- 데이터, 산식, 외부 라이브러리는 변경하지 않는다.

## Checklist
- [x] 현재 Plotly 차트 구성과 테스트 경계 확인
- [x] 선 끝 라벨, 월 눈금, hover 기준 개선
- [x] 회귀 테스트와 compile 검증
- [x] 결과 기록

## Review
- Changed files:
  `src/ui/panels.py`
  `tests/test_ui_components.py`
  `tasks/todo.md`
- Result:
  섹터 상대강도 차트와 섹터별 수출 YoY 차트에 선 끝 직접 라벨을 추가했다.
  X축은 월 단위 눈금(`%b\n%Y`, `M1`)으로 고정했고, hover는 같은 X축 위치의 series를 한 번에 비교하는 `x unified`로 바꿨다.
  상대강도 차트는 기준값 1.0, 수출 YoY 차트는 0% 기준선을 추가했다.
  수출 YoY 차트의 월 index는 timestamp로 변환해 월별 포인트 위치가 명확하게 보이도록 했다.
  끝 라벨이 가까울 때는 yshift를 적용해 겹침을 줄인다.
- Verification:
  `python -m py_compile src\ui\panels.py tests\test_ui_components.py` -> passed
  `python -m pytest -q tests/test_ui_components.py::test_build_sector_export_trend_figure_renders_monthly_series tests/test_ui_components.py::test_build_overview_trend_figure_labels_line_ends_and_month_ticks --basetemp=.pytest-tmp-chart-readability-focused-2` -> `2 passed`
  `python -m pytest -q tests/test_ui_components.py --basetemp=.pytest-tmp-chart-readability-ui-2` -> `79 passed`
  `git diff --check -- src/ui/panels.py tests/test_ui_components.py tasks/todo.md` -> no whitespace errors; LF/CRLF warnings only.

# 2026-05-13 - Sector Rotation Review Candidate First Screen

## Goal
- 기존 overview 첫 화면에서 사용자가 빠르게 검토할 섹터 1~3개와 이유를 확인하게 한다.
- 데이터, 점수 산식, 페이지, 외부 라이브러리, 이미지 에셋은 변경하지 않는다.
- 직접 매수 권유가 아닌 `검토 후보` 리서치 도구 톤을 유지한다.

## Checklist
- [x] 기존 overview 후보/섹터 frame과 decision copy 재사용 경계 확인
- [x] `검토 후보` 후보 데이터 helper와 pure renderer 추가
- [x] 모바일 top-three strip 중복 제거 또는 동일 후보 renderer로 통합
- [x] scoped CSS와 copy/test guardrail 추가
- [x] focused tests, compile, smoke/visual 검증
- [x] architect verification, deslop, post-deslop regression
- [x] 결과 기록

## Review
- Changed files:
  `src/ui/panels.py`
  `src/ui/css.py`
  `tests/test_ui_components.py`
  `tests/test_ui_theme.py`
  `tasks/todo.md`
- Result:
  overview 첫 화면에 `검토 후보` 섹션을 추가했다.
  후보는 기존 overview sector frame의 기본 `모멘텀 점수` 정렬을 재사용하고, `N/A`는 제외하며 최대 3개만 렌더링한다.
  후보 카드는 기존 decision copy/reason/invalidation을 재사용하고, renderer는 markdown 출력만 수행한다.
  모바일은 기존 top-three strip 대신 같은 candidate renderer를 compact 모드로 사용한다.
- Scope guard:
  데이터 소스, 점수 산식, 페이지 구조, 외부 라이브러리, 이미지 에셋은 변경하지 않았다.
  export 관련 diff는 이 Ralph lane 이전의 기존 worktree 변경으로 남겨두었다.
- Deslop:
  `.overview-mobile-decision-strip*`와 `.overview-decision-tile*` 죽은 CSS를 제거하고 theme selector smoke test를 새 `.overview-review-candidates` 계약으로 갱신했다.
- Verification:
  `python -m py_compile app.py src\ui\panels.py src\ui\css.py src\ui\copy.py src\dashboard\tabs.py` -> passed
  `python -m pytest -q tests/test_ui_components.py::test_build_overview_review_candidates_uses_default_momentum_order_and_limit tests/test_ui_components.py::test_render_overview_review_candidates_renders_reasons_and_guardrail_copy tests/test_ui_components.py::test_render_overview_review_candidates_empty_state tests/test_ui_components.py::test_render_overview_mobile_decision_strip_uses_review_candidate_markup --basetemp=.pytest-tmp-sector-review-candidates-focused-post` -> `4 passed`
  `python -m pytest -q tests/test_ui_components.py tests/test_ui_copy.py tests/test_ui_contrast.py tests/test_ui_theme.py --basetemp=.pytest-tmp-sector-review-candidates-ui-post2` -> `107 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py --basetemp=.pytest-tmp-sector-review-candidates-dashboard-post2` -> `37 passed`
  `git diff --check -- src/ui/panels.py src/ui/css.py tests/test_ui_components.py tests/test_ui_theme.py tasks/todo.md` -> no whitespace errors; LF/CRLF warnings only.
- Visual smoke:
  `.omx/artifacts/sector-review-candidates-desktop-post-8514.png`
  `.omx/artifacts/sector-review-candidates-mobile-post-8515.png`
  Desktop and mobile screenshots show 3 `검토 후보` cards without visible clipping or overlap.
- Architect verification:
  Approved with WATCH status.
  Candidate implementation satisfied ordering, N/A exclusion, max-3, purity, mobile semantics, and guardrails.

# 2026-05-13 - Representative ETF Mapping Overlay

## Goal
- 동적 KRX 섹터 유니버스가 `sector_map.yml`의 정적 ETF 매핑보다 넓을 때 대표 ETF가 비어 보이는 문제를 줄인다.
- 국면-섹터 배정 권한은 `config/sector_map.yml`에 남기고, 실행 참고 ETF는 생성 오버레이로 보완한다.

## Checklist
- [x] 대표 ETF 매핑 누락 원인 확인
- [x] 원본 sector map과 생성 ETF overlay 병합 경로 추가
- [x] KRX 건설/증권/정보기술 ETF overlay 추가
- [x] focused tests 및 compile 검증
- [x] 결과 기록

## Review
- Changed files:
  `src/data_sources/sector_etf_mapping.py`
  `src/dashboard/tabs.py`
  `data/curated/sector_etf_map.generated.yml`
  `tests/test_sector_etf_mapping.py`
  `tasks/todo.md`
- Result:
  `sector_map.yml`은 macro regime authority로 유지하고, 대표 ETF 실행 참고는 generated overlay를 병합해 보완한다.
  명시적 `sector_map.yml` ETF 매핑이 있으면 원본 설정이 우선한다.
  현재 overlay는 `5052 KRX 건설`, `5054 KRX 증권`, `5064 KRX 정보기술`을 채운다.
- Effective map check:
  `5052 -> 117700 KODEX 건설`
  `5054 -> 102970 KODEX 증권 / 157500 TIGER 증권`
  `5064 -> 266370 KODEX IT`
- Verification:
  `python -m py_compile src\data_sources\sector_etf_mapping.py src\dashboard\tabs.py tests\test_sector_etf_mapping.py` -> passed
  `python -m pytest -q tests/test_sector_etf_mapping.py tests/test_dashboard_tabs.py::test_render_screening_tab_renders_representative_etf_context tests/test_dashboard_tabs.py::test_render_screening_tab_initial_render_uses_cache_only_loaders tests/test_dashboard_tabs.py::test_render_screening_tab_refresh_allows_live_loaders --basetemp=.pytest-tmp-sector-etf-map` -> `7 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py tests/test_sector_etf_mapping.py --basetemp=.pytest-tmp-sector-etf-map-full` -> `41 passed`

# 2026-05-12 - Collection Completion Rate Semantics

## Goal
- 매크로/수급 데이터가 존재하는데 완료율이 0%로 보이는 문제를 고친다.
- 매크로 완료율은 alias 완주 개수가 아니라 요청 월 x enabled alias 셀 커버리지로 계산한다.
- 수급/시장 데이터는 요청 카운터가 없더라도 저장행수가 있으면 0% 대신 부분수집으로 보이게 한다.

## Checklist
- [x] 기존 완료율 산식과 0% 원인 확인
- [x] 매크로 완료율 산식 변경
- [x] 요청 카운터 없는 저장 데이터 fallback 변경
- [x] UI 주의 문구와 회귀 테스트 갱신
- [x] focused tests 및 compile 검증
- [x] 결과 기록

## Review
- Changed files:
  `src/data_sources/warehouse.py`
  `src/dashboard/tabs.py`
  `tests/test_warehouse_cli.py`
  `tests/test_dashboard_tabs.py`
  `tasks/todo.md`
- Result:
  매크로 완료율을 `완주 alias 수 / enabled alias 수`에서 `수집된 alias-month 셀 수 / 요청 alias-month 셀 수`로 변경했다.
  따라서 데이터가 존재하는 KOSIS가 0%로 찍히지 않고 현재 로컬 warehouse 기준 79.2%로 표시된다.
  ECOS도 같은 기준으로 98.6%로 표시된다.
  요청 카운터가 없는 수급 데이터는 저장행수가 있으면 완료율 0% 대신 `부분 수집 데이터 있음 (n건)`으로 표시한다.
- Local output check:
  현재 요약 표는 KOSIS `요청 범위 일부 미충족 (79.2%)`, ECOS `요청 범위 일부 미충족 (98.6%)`, 수급데이터 `부분 수집 데이터 있음 (2,871건)`으로 표시된다.
- Verification:
  `python -m py_compile src\data_sources\warehouse.py src\dashboard\tabs.py tests\test_warehouse_cli.py tests\test_dashboard_tabs.py` -> passed
  `python -m pytest -q tests/test_dashboard_tabs.py tests/test_warehouse_cli.py --basetemp=.pytest-tmp-completion-semantics-full` -> `54 passed`
  `git diff --check -- src/dashboard/tabs.py src/data_sources/warehouse.py tests/test_dashboard_tabs.py tests/test_warehouse_cli.py tasks/todo.md` -> passed with line-ending warnings only

# 2026-05-12 - Data Collection History Redesign

## Goal
- 데이터 수집 이력 화면을 최신 갱신 시점, 실제 보유 기간, 실패 이유 중심으로 재구성한다.
- 저장행수/완료율 중심의 진단 테이블은 상세 이력으로 낮추고, 운영자가 한눈에 상태를 판단하게 한다.
- 기존 warehouse 기록/캐시 경로는 유지하고 UI 표시 모델만 개선한다.

## Checklist
- [x] 기존 이력 화면과 warehouse 상태/이력 필드 확인
- [x] 데이터셋별 실제 보유 기간 조회 helper 추가
- [x] 최신 갱신 요약 테이블 추가
- [x] 실패/주의 사유를 간결한 문장으로 정리
- [x] 기존 최근 이력 테이블을 상세 로그로 유지
- [x] focused tests 및 compile 검증
- [x] 결과와 남은 리스크 기록

## Review
- Changed files:
  `src/data_sources/warehouse.py`
  `src/dashboard/tabs.py`
  `tests/test_warehouse_cli.py`
  `tests/test_dashboard_tabs.py`
  `tasks/todo.md`
- Result:
  데이터 수집 이력 화면의 첫 표를 최신 갱신 요약으로 재구성했다.
  이제 `마지막 갱신`, `보유기간`, `최근 요청`, `실패/주의`, `provider`, `저장행수`를 한 줄에서 확인한다.
  매크로데이터는 ECOS/KOSIS provider별 최신 행으로 분리해 완료율 0.0%/9.1% 같은 차이가 한눈에 보이게 했다.
  기존 상세 이력은 `최근 수집 실행 로그`로 이름을 바꿔 아래에 유지했다.
- Local output check:
  현재 warehouse 기준 요약은 시장데이터 2016-03-08~2026-05-11, KOSIS 2016-03~2026-04, ECOS 2016-03~2026-05, 수급데이터 2025-12-11~2026-04-21로 표시된다.
- Verification:
  `python -m py_compile src\dashboard\tabs.py src\data_sources\warehouse.py tests\test_dashboard_tabs.py tests\test_warehouse_cli.py` -> passed
  `python -m pytest -q tests/test_dashboard_tabs.py tests/test_warehouse_cli.py --basetemp=.pytest-tmp-monitoring-redesign-full2` -> `52 passed`
  `git diff --check -- src/dashboard/tabs.py src/data_sources/warehouse.py tests/test_dashboard_tabs.py tests/test_warehouse_cli.py tasks/todo.md` -> passed with line-ending warnings only
- Remaining risks:
  실제 provider API 실패 원인은 warehouse에 저장된 `failed_codes`/`abort_reason` 범위 안에서만 표시된다.

# 2026-05-12 - Auto Export Visibility Fix

## Goal
- 자동차 수출 데이터가 캐시에 있어도 화면에서 독립적으로 인식되지 않는 문제를 고친다.
- `KOSPI200 경기소비재`에 연결된 `export_auto`가 자동차 수출 기준임을 UI에서 명확히 보여준다.
- 월별 수출 차트는 현재 visible signal에 없는 수출 품목도 빠뜨리지 않는다.

## Checklist
- [x] 기존 수출 UI 표시 경계 확인
- [x] 섹터 테이블에 수출 기준 라벨 표시
- [x] 월별 수출 차트에 모든 수출 시계열 포함
- [x] 자동차 수출 표시 회귀 테스트 추가
- [x] focused tests 및 compile 검증
- [x] 결과와 남은 리스크 기록

## Review
- Changed files:
  `src/ui/panels.py`
  `src/ui/css.py`
  `tests/test_ui_components.py`
  `tasks/todo.md`
  `tasks/lessons.md`
- Fix:
  자동차 수출은 기존처럼 `KOSPI200 경기소비재`의 export series로 유지하되, 섹터 테이블의 섹터명 아래에 `수출 기준: 자동차 수출` 보조 라벨을 표시한다.
  월별 수출 YoY 차트는 현재 signal에 있는 섹터를 먼저 그린 뒤, signal에 없는 export history도 뒤에 추가해 자동차 수출 라인이 빠지지 않게 했다.
  차트 trace 이름은 `자동차 수출`, `반도체 수출`처럼 품목 기준으로 표시한다.
- Verification:
  `python -m py_compile src\ui\panels.py src\ui\css.py tests\test_ui_components.py` -> passed
  `python -m pytest -q tests/test_ui_components.py --basetemp=.pytest-tmp-auto-export-visibility-ui` -> `73 passed`
  `git diff --check -- src/ui/panels.py src/ui/css.py tests/test_ui_components.py tasks/todo.md` -> passed with line-ending warnings only

# 2026-05-12 - Autopilot Sector Export Trends Dashboard

## Goal
- 한국 전체 수출 전년비를 넘어 주요 섹터별 수출 동향을 일반 대시보드에서 확인할 수 있게 한다.
- 기존 ECOS macro warehouse/cache 경로를 재사용한다.
- 섹터 액션 산식은 바꾸지 않고 보조 진단 지표로만 표시한다.

## Checklist
- [x] Autopilot context/PRD/test-spec 산출물 작성
- [x] ECOS 품목별 수출금액지수 series alias 추가
- [x] `sector_map.yml` 섹터별 export alias 매핑 추가
- [x] 앱에서 섹터별 12개월 수출 YoY 계산
- [x] 일반 대시보드 섹터 테이블에 `수출 YoY` 열 추가
- [x] 일반 대시보드에 섹터별 월별 수출 YoY 추이 차트 추가
- [x] macro warehouse 캐시 force refresh로 새 alias 적재 확인
- [x] focused UI/macro tests 및 compile/diff 검증
- [x] code-review gate 기록

## Artifacts
- Context: `.omx/context/sector-export-trends-dashboard-20260512T120441Z.md`
- PRD: `.omx/plans/prd-sector-export-trends-dashboard.md`
- Test spec: `.omx/plans/test-spec-sector-export-trends-dashboard.md`

## Review
- Changed files:
  `config/macro_series.yml`
  `config/sector_map.yml`
  `src/data_sources/ecos.py`
  `app.py`
  `src/dashboard/types.py`
  `src/ui/panels.py`
  `tests/test_ui_components.py`
  plus prior aggregate export display files from the preceding export task.
- Result:
  주요 섹터 수출금액지수 aliases를 추가했다:
  `export_it`, `export_semiconductor`, `export_chemicals`, `export_steel`, `export_auto`, `export_machinery`, `export_pharma`.
  일반 대시보드 섹터 테이블에 `수출 YoY`가 표시되고, 정렬 기준에도 `수출 YoY`가 추가된다.
  우측 패널에는 최근 24개월 섹터별 `수출 YoY 월별 추이` 라인 차트가 추가된다.
- Cache refresh evidence:
  `sync_macro_warehouse(..., reason="manual_sector_export_trends_refresh", force=True)` -> `LIVE`
  latest period: `2026-03`
  alias rows: each selected export alias `119`
  latest YoY sample:
  `export_semiconductor=155.2638%`, `export_it=121.642%`, `export_chemicals=7.5128%`, `export_steel=1.0587%`, `export_auto=1.1612%`, `export_machinery=-0.7837%`, `export_pharma=0.9169%`
- Verification:
  `python -m py_compile app.py src\ui\panels.py src\dashboard\types.py src\data_sources\ecos.py tests\test_ui_components.py tests\test_dashboard_tabs.py` -> passed
  `python -m pytest -q tests/test_ui_components.py tests/test_dashboard_tabs.py --basetemp=.pytest-tmp-sector-export-ui` -> `107 passed`
  `python -m pytest -q tests/test_ui_components.py tests/test_dashboard_tabs.py --basetemp=.pytest-tmp-monthly-export-ui` -> `108 passed`
  `python -m pytest -q tests/test_ecos_kosis_api_handling.py tests/test_macro_sync.py --basetemp=.pytest-tmp-sector-export-macro` -> `20 passed`
  `python -m pytest -q tests/test_ecos_kosis_api_handling.py tests/test_macro_sync.py --basetemp=.pytest-tmp-monthly-export-macro` -> `20 passed`
  `git diff --check` -> passed with line-ending warnings only
- Code review:
  Recommendation `APPROVE`; architectural status `CLEAR`.
  No new provider dependency was introduced.
  Scoring/action semantics were not changed.

# 2026-05-12 - Autoresearch Korean Export Dashboard Signal

## Goal
- 한국 수출 전년비 데이터를 기존 매크로 데이터 파이프라인에 포함한다.
- 일반 대시보드 화면에서 의사결정에 바로 보이도록 노출한다.
- 새 데이터 소스는 기존 KOSIS OpenAPI/KOSIS warehouse 경로를 재사용해 캐시 정책을 깨지 않는다.

## Checklist
- [x] 기존 KOSIS `export_growth` 설정과 매크로 캐시 경로 확인
- [x] 실제 호출 가능한 ECOS 수출금액 series로 전환
- [x] 최신 수출 전년비 값을 대시보드 표시 모델에 연결
- [x] UI copy와 컴포넌트 테스트 갱신
- [x] autoresearch 산출물 작성
- [x] focused 테스트와 정적 검증 실행

## Review
- Changed files:
  `config/macro_series.yml`
  `src/data_sources/ecos.py`
  `src/data_sources/kosis.py`
  `app.py`
  `src/dashboard/types.py`
  `src/dashboard/tabs.py`
  `src/ui/copy.py`
  `src/ui/panels.py`
  `tests/test_dashboard_tabs.py`
  `tests/test_ui_components.py`
  `tasks/todo.md`
- Output artifact:
  `.omx/specs/autoresearch-korean-export-dashboard/report.md`
- Completion artifact:
  `.omx/specs/autoresearch-korean-export-dashboard/result.json`
- Result:
  기존 KOSIS `export_growth` 후보는 실제 호출에서 통계표 오류가 나서 비활성 상태로 유지했다.
  대신 한국은행 ECOS `901Y118/T002` 수출금액을 `export_amount`로 매크로 캐시에 추가하고, 앱 표시 단계에서 12개월 전년비를 계산해 일반 대시보드 시장/조회 카드와 의사결정 히어로에 연결했다.
  ECOS API key 또는 최신 캐시가 없으면 기존 macro fallback 정책에 따라 값은 `N/A` 또는 미표시될 수 있다.
- Cache refresh:
  `sync_macro_warehouse(..., market="KR", reason="manual_export_amount_refresh", force=True)` -> `LIVE`
  warehouse `export_amount` rows: `119`
  latest period: `2026-03`
  latest YoY: `49.19061991718805%`
- Verification:
  `python -m py_compile app.py src\dashboard\tabs.py src\ui\panels.py src\ui\copy.py src\dashboard\types.py src\data_sources\ecos.py src\data_sources\kosis.py tests\test_dashboard_tabs.py tests\test_ui_components.py` -> passed
  `python -m pytest -q tests/test_dashboard_tabs.py::test_render_decision_first_sections_orders_main_canvas tests/test_ui_components.py::test_render_decision_hero_renders_regime_and_provisional_badge` -> `2 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py::test_render_decision_first_sections_orders_main_canvas tests/test_ui_components.py::test_render_decision_hero_renders_regime_and_provisional_badge tests/test_ui_components.py::test_overview_market_cards_include_export_growth_status --basetemp=.pytest-tmp-export-smoke` -> `3 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py tests/test_ui_components.py --basetemp=.pytest-tmp-export-ui` -> `105 passed`
  `python -m pytest -q tests/test_ecos_kosis_api_handling.py tests/test_macro_sync.py --basetemp=.pytest-tmp-export-macro` -> `20 passed`
  `git diff --check` -> passed with line-ending warnings only
- Note:
  First broad UI/macro test attempt ran in parallel against the same `.pytest-tmp` DuckDB path and failed during Windows temp cleanup.
  Re-running the same suites sequentially with isolated `--basetemp` directories passed.

# 2026-05-12 - Autoresearch Investor Flow Importance

## Goal
- 섹터 로테이션 탐색에서 투자자 수급을 핵심 신호로 볼지, 보조 신호로 볼지 판단한다.
- 학술/시장구조 근거와 현재 repo 구현 근거를 분리해 기록한다.

## Checklist
- [x] repo의 기존 수급 overlay 설계 확인
- [x] 섹터/업종 모멘텀 근거 확인
- [x] ETF/펀드 수급의 가격 압력 및 반전 근거 확인
- [x] 결론과 운용 가드레일 작성
- [x] autoresearch completion artifact 작성

## Review
- Output artifact:
  `.omx/specs/autoresearch-investor-flow-sector-rotation/report.md`
- Project docs copy:
  `docs/research/investor-flow-sector-rotation-2026-05-12.md`
- Completion artifact:
  `.omx/specs/autoresearch-investor-flow-sector-rotation/result.json`
- Conclusion:
  투자자 수급은 섹터 로테이션에서 중요하지만 1차 결정 변수가 아니라 확인/리스크 조정 변수로 두는 편이 타당하다.
  기본 랭킹은 상대강도, 추세, 변동성, 매크로 국면으로 만들고 수급은 conviction 조정, 과열/반전 경고, 진입 타이밍 보정에 쓴다.
  특히 KR 수급은 데이터 소스가 비공식/운영상 취약하므로 최종 액션을 단독으로 뒤집는 신호가 되면 안 된다.

# 2026-05-11 - Incremental Data Refresh

## Goal
- 사이드바 데이터 갱신이 DB의 기존 수집 상태를 기준으로 증분 범위만 수집하게 한다.
- 시장데이터/매크로데이터는 전체 설정기간 재요청을 피한다.
- 투자자수급은 기존 operational cursor 기반 증분 resolver를 유지해 KRX 크롤링 폭주를 막는다.

## Checklist
- [x] DB 스키마와 수집 상태 테이블 확인
- [x] 사이드바 갱신 버튼 경로 확인
- [x] Autopilot context/PRD/test-spec 산출물 작성
- [x] 시장데이터 증분 시작일 계산 추가
- [x] 매크로데이터는 `macro_sync`가 증분/보정 정책을 소유하도록 runtime 전체 윈도우 전달 유지
- [x] 투자자수급 resolver 위임 계약 회귀 테스트 추가
- [x] focused py_compile 및 pytest 검증
- [x] 코드 리뷰 결과 기록

## Plan Artifacts
- `.omx/context/incremental-data-refresh-20260510T150430Z.md`
- `.omx/plans/prd-incremental-data-refresh.md`
- `.omx/plans/test-spec-incremental-data-refresh.md`

## Review
- Changed files:
  `src/dashboard/runtime.py`
  `tests/test_dashboard_runtime.py`
  `.omx/context/incremental-data-refresh-20260510T150430Z.md`
  `.omx/plans/prd-incremental-data-refresh.md`
  `.omx/plans/test-spec-incremental-data-refresh.md`
  `tasks/todo.md`
- Fix:
  시장데이터 갱신은 요청 코드 전체의 warehouse 최신일을 읽고, 모든 코드가 이력을 가진 경우 가장 오래된 최신일 다음 날부터만 runner에 전달한다.
  이미 최신이면 live runner를 호출하지 않고, 기존 transient market preview도 지운다.
  매크로데이터는 runtime에서 임의로 줄이지 않고 기존 120개월 윈도우를 `sync_macro_warehouse()`에 전달한다. provider-series drift, gap, alias별 증분 fetch는 `macro_sync`가 계속 판단한다.
  투자자수급은 `start_date_str`를 넘기지 않아 기존 operational complete cursor / failed-day repair resolver가 KRX 수집 범위를 결정한다.
- Verification:
  `python -m py_compile src\dashboard\runtime.py tests\test_dashboard_runtime.py` -> passed
  `python -m pytest -q tests/test_dashboard_runtime.py` -> `29 passed`
  `python -m pytest -q tests/test_dashboard_runtime.py tests/test_macro_sync.py tests/test_krx_investor_flow_data_source.py::test_run_manual_investor_flow_refresh_returns_cached_when_already_current tests/test_integration.py::TestIntegration::test_run_manual_price_refresh_caps_openapi_range_to_recent_window tests/test_warehouse_cli.py::test_sync_warehouse_cli_uses_incremental_market_start` -> `37 passed`
  `git diff --check -- <scoped files>` -> passed with line-ending warnings only
- Code review:
  Initial review blocked the macro runtime-shortening approach because it could bypass provider-series drift repair.
  Follow-up removed macro runtime shortening and updated tests/specs.
  Second review requested clearing stale market transient preview on already-current skip.
  Final code-review recommendation `APPROVE`; architectural status `CLEAR`.
- Remaining risks:
  매크로 refresh 버튼은 provider drift 안전성을 위해 orchestration range를 계속 120개월로 넘긴다. 실제 provider 호출량 감소는 기존 `sync_provider_macro()`의 alias별 cache/coverage 판단에 의존한다.

# 2026-05-11 - Slow Investor Flow Day Collector

## Goal
- KRX 수급 endpoint를 바로 재시도하지 않고, 다음 실행 때 차단 위험을 낮추는 하루치 저속 수집기를 추가한다.
- 기본 실행은 dry-run이어야 하며, `--execute` 없이는 KRX 네트워크 요청을 하지 않는다.
- 수집은 ticker 단위로 재개 가능해야 하고, `ACCESS_DENIED` 감지 시 즉시 중단해야 한다.

## Checklist
- [x] 기존 수급 collector/warehouse 저장 경계 확인
- [x] 저속 하루 수집 CLI 추가
- [x] dry-run, sector 필터, ticker limit, resume state 구현
- [x] access-denied 즉시 중단 및 partial state 보존 구현
- [x] 단위 테스트 추가
- [x] py_compile 및 focused pytest 검증
- [x] 결과/리스크 기록

## Review
- Changed files:
  `scripts/collect_investor_flow_day_slow.py`
  `tests/test_collect_investor_flow_day_slow.py`
  `tasks/todo.md`
- Implementation:
  새 CLI는 기본 dry-run이라 `--execute` 없이는 KRX 요청을 하지 않는다.
  cached sector constituent snapshot만 사용해 구성종목 live lookup을 피한다.
  ticker 단위로 raw rows를 `data/runtime/investor_flow_slow/.../raw.parquet`에 spool하고 `state.json`으로 완료/실패 ticker를 보존한다.
  `ACCESS_DENIED`로 분류되는 실패는 즉시 중단하며 실패 상태를 저장한다.
  전체 섹터/전체 ticker가 완주된 경우에만 기존 operational warehouse writer를 호출한다. `--sectors export`, `--sector-codes`, `--max-tickers` 같은 부분 수집은 cursor를 갱신하지 않는다.
- Verification:
  `python -m py_compile scripts\collect_investor_flow_day_slow.py tests\test_collect_investor_flow_day_slow.py` -> passed
  `python -m pytest -q tests/test_collect_investor_flow_day_slow.py` -> `4 passed`
  `python scripts\collect_investor_flow_day_slow.py --date 20260422 --sectors export --max-tickers 2` -> dry-run only, `planned_tickers_this_run=2`, `processed_requests=0`
  `git diff --check -- scripts/collect_investor_flow_day_slow.py tests/test_collect_investor_flow_day_slow.py tasks/todo.md` -> passed with line-ending warning for `tasks/todo.md`
- Remaining risks:
  이 변경은 차단 회피가 아니라 요청량/속도/재개성을 개선하는 실행 경로다.
  실제 `--execute`는 현재 cooldown 이후에만 사용해야 하며, KRX가 계속 403을 반환하면 즉시 중단된다.

# 2026-05-06 - KRX Investor Flow Persistent Cooldown

## Goal
- KRX access-denied cooldown을 프로세스 전역 변수에서 로컬 runtime JSON 상태로 확장한다.
- Streamlit 재시작/다중 프로세스에서도 cooldown이 유지되어 KRX 재타격을 막는다.

## Checklist
- [x] Ralph context/PRD/test-spec 산출물 작성
- [x] 현재 cooldown helper 경계 확인
- [x] persistent cooldown 파일 저장/읽기 구현
- [x] runtime 상태 파일 gitignore 처리
- [x] cross-process 회귀 테스트 추가
- [x] py_compile 및 focused pytest 검증
- [x] architect verification 및 deslop 후 재검증
- [x] 결과와 남은 리스크 기록

## Plan Artifacts
- `.omx/context/krx-investor-flow-persistent-cooldown-20260506T111538Z.md`
- `.omx/plans/prd-krx-investor-flow-persistent-cooldown.md`
- `.omx/plans/test-spec-krx-investor-flow-persistent-cooldown.md`

## Review
- Changed files:
  `.gitignore`
  `src/data_sources/krx_investor_flow.py`
  `tests/test_krx_investor_flow_data_source.py`
  `.omx/context/krx-investor-flow-persistent-cooldown-20260506T111538Z.md`
  `.omx/plans/prd-krx-investor-flow-persistent-cooldown.md`
  `.omx/plans/test-spec-krx-investor-flow-persistent-cooldown.md`
  `tasks/todo.md`
- Simplifications made:
  Cooldown state now persists to `data/runtime/krx_investor_flow_access_denied_cooldown.json`.
  The runtime state file is ignored via `.gitignore`.
  State writes use a pid-scoped temp file plus atomic replace.
  Cooldown reads tolerate missing, corrupt, inactive, or expired JSON by clearing/ignoring the state.
  Every cooldown check reloads persisted state and prefers a newer persisted deadline over stale memory, so an old process cannot erase another process's newer cooldown.
- Verification:
  `python -m py_compile src\data_sources\krx_investor_flow.py tests\test_krx_investor_flow_data_source.py` -> passed
  `python -m pytest -q tests/test_krx_investor_flow_data_source.py::test_access_denied_cooldown_persists_to_runtime_file tests/test_krx_investor_flow_data_source.py::test_access_denied_cooldown_survives_memory_reset_for_manual_refresh tests/test_krx_investor_flow_data_source.py::test_access_denied_cooldown_clears_expired_runtime_file tests/test_krx_investor_flow_data_source.py::test_access_denied_cooldown_ignores_corrupt_runtime_file tests/test_krx_investor_flow_data_source.py::test_access_denied_cooldown_ignores_inactive_runtime_file` -> `5 passed`
  `python -m pytest -q tests/test_krx_investor_flow_data_source.py::test_access_denied_cooldown_prefers_newer_runtime_file_over_stale_memory tests/test_krx_investor_flow_data_source.py::test_access_denied_cooldown_clears_expired_runtime_file tests/test_krx_investor_flow_data_source.py::test_access_denied_cooldown_survives_memory_reset_for_manual_refresh` -> `3 passed`
  `python -m pytest -q tests/test_krx_investor_flow_data_source.py tests/test_dashboard_runtime.py::test_build_investor_flow_refresh_notice_surfaces_access_denied --basetemp "$env:TEMP\sector-rotation-pytest-krx-flow-persistent-cooldown-final"` -> `63 passed`
- Code review:
  Initial architect review rejected stale-memory behavior because an old process could clear a newer persisted cooldown.
  Follow-up made `_get_access_denied_cooldown()` re-read persisted state before expiry clearing and prefer newer file deadlines.
  Final architect review: `APPROVE`.
- Deslop:
  Scope stayed limited to Ralph-owned files.
  Cleanup tightened inactive-state handling and pid-scoped temp writes.
- Remaining risks:
  Cooldown sharing is local-filesystem only, not cross-machine.
  If the app runs in an environment where `data/runtime` is not writable, the in-memory cooldown still works but persistence is best-effort.

# 2026-05-06 - KRX Investor Flow Access-Denied Circuit Breaker

## Goal
- KRX 투자자수급 endpoint가 access denied를 반환하면 일정 시간 live 수집을 멈추고 캐시 fallback만 사용한다.
- 반복 요청/세션 reset 폭주를 막고, 회귀 테스트로 short-circuit 동작을 고정한다.

## Checklist
- [x] Ralph context/PRD/test-spec 산출물 작성
- [x] 현재 수급 refresh/retry 경로 확인
- [x] access-denied cooldown 회로 차단 구현
- [x] focused 회귀 테스트 추가
- [x] py_compile 및 focused pytest 검증
- [x] architect verification 및 deslop 후 재검증
- [x] 결과와 남은 리스크 기록

## Plan Artifacts
- `.omx/context/krx-investor-flow-access-denied-circuit-breaker-20260506T104437Z.md`
- `.omx/plans/prd-krx-investor-flow-access-denied-circuit-breaker.md`
- `.omx/plans/test-spec-krx-investor-flow-access-denied-circuit-breaker.md`

## Review
- Changed files:
  `src/data_sources/krx_investor_flow.py`
  `tests/test_krx_investor_flow_data_source.py`
  `.omx/context/krx-investor-flow-access-denied-circuit-breaker-20260506T104437Z.md`
  `.omx/plans/prd-krx-investor-flow-access-denied-circuit-breaker.md`
  `.omx/plans/test-spec-krx-investor-flow-access-denied-circuit-breaker.md`
  `tasks/todo.md`
- Simplifications made:
  KRX access-denied failure now activates a 6-hour process-local cooldown.
  During cooldown, manual investor-flow refresh returns warehouse/cache fallback without resolving the default refresh window or calling pykrx/KRX live collection.
  During cooldown, trading-day calendar resolution uses weekday fallback before pykrx warmup, protecting backfill/direct-collector paths from pre-collector KRX probes.
  Direct `collect_sector_investor_flow()` calls now short-circuit before socket checks or pykrx setup while cooldown is active.
  The cooldown fallback reuses one helper for explicit and default-window paths, preserving `ACCESS_DENIED` notice classification and `processed_requests=0`.
  Non-access-denied endpoint failures still use the existing degraded-cache path without activating cooldown.
- Verification:
  `python -m py_compile src\data_sources\krx_investor_flow.py tests\test_krx_investor_flow_data_source.py` -> passed
  `python -m pytest -q tests/test_krx_investor_flow_data_source.py::test_requested_trading_days_uses_weekday_fallback_during_access_denied_cooldown tests/test_krx_investor_flow_data_source.py::test_collect_sector_investor_flow_short_circuits_during_access_denied_cooldown tests/test_krx_investor_flow_data_source.py::test_run_manual_investor_flow_refresh_short_circuits_default_window_during_cooldown` -> `3 passed`
  `python -m pytest -q tests/test_krx_investor_flow_data_source.py tests/test_dashboard_runtime.py::test_build_investor_flow_refresh_notice_surfaces_access_denied --basetemp "$env:TEMP\sector-rotation-pytest-krx-flow-cooldown-final"` -> `57 passed`
  `python -m pytest -q tests/test_dashboard_runtime.py::test_build_investor_flow_refresh_notice_surfaces_access_denied --basetemp "$env:TEMP\sector-rotation-pytest-dashboard-runtime-cooldown"` -> `1 passed`
  `git diff --check -- <Ralph-owned files>` -> passed with line-ending warnings only
  Post-deslop `python -m py_compile src\data_sources\krx_investor_flow.py tests\test_krx_investor_flow_data_source.py` -> passed
  Post-deslop `python -m pytest -q tests/test_krx_investor_flow_data_source.py tests/test_dashboard_runtime.py::test_build_investor_flow_refresh_notice_surfaces_access_denied --basetemp "$env:TEMP\sector-rotation-pytest-krx-flow-cooldown-post-deslop-final"` -> `57 passed`
- Code review:
  Initial architect review rejected the first version because cooldown ran after default refresh-window resolution, which could still touch pykrx calendar.
  Follow-up moved the cooldown gate before default window resolution and added a regression that fails if the resolver runs during cooldown.
  Second architect review rejected direct backfill coverage because trading-day resolution could still probe pykrx before collector entry.
  Follow-up made trading-day resolution use weekday fallback during cooldown and added a regression that fails if pykrx transport warms.
  Final architect review: `APPROVE`.
- Deslop:
  Scope stayed limited to Ralph-owned files.
  No extra cleanup edit was needed after extracting the shared cooldown fallback helper.
- Remaining risks:
  Cooldown is process-local and resets on Streamlit process restart.
  The 6-hour duration is conservative but not synchronized across multiple app processes.
  During cooldown, KR trading-day truth falls back to weekdays; this is intentionally less precise than KRX calendar lookup to avoid touching KRX while blocked.

# 2026-05-05 - KRX Access Denied Degraded Refresh

## Goal
- KRX Access Denied 상황에서 앱이 캐시 fallback summary를 유지하되 반복 경고와 예상 가능한 stack trace를 줄인다.

## Checklist
- [x] 오류 로그와 기존 KRX access-denied 처리 경로 확인
- [x] Autopilot context/PRD/test-spec 산출물 작성
- [x] investor-flow expected endpoint failure logging 축소
- [x] OpenAPI repeated access-denied batch warning dedupe
- [x] focused 회귀 테스트 추가/갱신
- [x] py_compile 및 focused pytest 검증
- [x] 코드 리뷰 결과 기록

## Plan Artifacts
- `.omx/context/krx-access-denied-degraded-refresh-20260505T110323Z.md`
- `.omx/plans/prd-krx-access-denied-degraded-refresh.md`
- `.omx/plans/test-spec-krx-access-denied-degraded-refresh.md`

## Notes
- KRX upstream 접근 자체를 우회하거나 해결하지 않는다.
- 목표는 예상 가능한 provider 차단을 명확한 degraded state로 기록하고 운영 로그를 작게 만드는 것이다.

## Review
- Changed files:
  `src/data_sources/krx_investor_flow.py`
  `src/data_sources/krx_openapi.py`
  `src/dashboard/data.py`
  `tests/test_krx_investor_flow_data_source.py`
  `tests/test_krx_openapi.py`
  `.omx/context/krx-access-denied-degraded-refresh-20260505T110323Z.md`
  `.omx/plans/prd-krx-access-denied-degraded-refresh.md`
  `.omx/plans/test-spec-krx-access-denied-degraded-refresh.md`
  `tasks/todo.md`
- Simplifications made:
  예상 가능한 KRX access-denied/non-JSON 수급 실패는 `logger.exception` 대신 warning으로 기록한다.
  실패 detail은 기존처럼 `failed_codes.refresh`와 ingest failure summary에 보존한다.
  OpenAPI batch abort warning은 같은 access-denied detail에 대해 프로세스당 1회만 출력한다.
  pykrx JSON retry/login 실패 warning도 같은 detail 기준 프로세스당 1회만 출력한다.
  KR 공식 이름 discovery fallback 실패 warning도 같은 detail 기준 프로세스당 1회만 출력한다.
  Full pytest에서 드러난 기존 US benchmark proxy 회귀도 작은 label fallback으로 정리했다.
- Verification:
  `python -m py_compile src\data_sources\krx_investor_flow.py src\data_sources\krx_openapi.py tests\test_krx_investor_flow_data_source.py tests\test_krx_openapi.py` -> passed
  `python -m pytest -q tests/test_krx_investor_flow_data_source.py::test_run_manual_investor_flow_refresh_logs_expected_access_denied_without_stacktrace tests/test_krx_investor_flow_data_source.py::test_run_manual_investor_flow_refresh_classifies_access_denied tests/test_krx_openapi.py::test_fetch_index_ohlcv_openapi_batch_logs_repeated_access_denied_once` -> `3 passed`
  `python -m pytest -q tests/test_krx_openapi.py tests/test_pykrx_compat.py` -> `35 passed`
  `python -m pytest -q tests/test_krx_investor_flow_data_source.py tests/test_dashboard_runtime.py` -> `77 passed`
  `python -m py_compile src\dashboard\data.py` -> passed
  `python -m pytest -q tests/test_us_signal_pipeline.py::test_cached_signals_supports_us_market` -> `1 passed`
  `python -m pytest -q` -> `568 passed`
  `python -m py_compile src\data_sources\pykrx_compat.py src\dashboard\data.py tests\test_pykrx_compat.py tests\test_dashboard_data.py` -> passed
  `python -m pytest -q tests/test_pykrx_compat.py::test_request_krx_data_logs_repeated_access_denied_retry_once tests/test_dashboard_data.py::test_kr_active_index_name_lookup_logs_official_discovery_failure_once --basetemp "$env:TEMP\sector-rotation-pytest"` -> `2 passed`
  `python -m pytest -q tests/test_pykrx_compat.py tests/test_dashboard_data.py --basetemp "$env:TEMP\sector-rotation-pytest"` -> `29 passed`
  `python -m pytest -q tests/test_krx_investor_flow_data_source.py tests/test_dashboard_runtime.py --basetemp "$env:TEMP\sector-rotation-pytest-2"` -> `77 passed`
  `python -m pytest -q --basetemp "$env:TEMP\sector-rotation-pytest-full"` -> `570 passed`
- Code review:
  Recommendation `APPROVE`; architectural status `CLEAR`.
  KRX scope is limited to logging/deduplication and regression tests; cache fallback contracts are unchanged.
  US follow-up is limited to resolving a benchmark proxy by configured benchmark label when the configured code is absent.
- Remaining risks:
  KRX may still block OpenAPI or unofficial investor-flow endpoints from this network/session.
  Live data success still depends on valid KRX access and upstream endpoint availability.

# 2026-05-05 - KR Investor Flow UX Simplification

## Goal
- KR 투자자 수급 화면에서 사용자가 요청한 두 정보만 남긴다.
- 섹터별 최신 수급 금액은 조/억 단위로 읽히게 표시한다.
- 기간별 수급 추이 그래프를 함께 보여준다.

## Checklist
- [x] 현재 KR 수급 탭 렌더링 경로 확인
- [x] 최신 섹터별 수급 금액 표로 단순화
- [x] 기간별 추이 그래프 추가
- [x] 기존 σ/신호 변화/설명성 표 노출 제거
- [x] focused 테스트와 컴파일 검증
- [x] 결과와 남은 리스크 기록

## Review
- Changed files:
  `src/dashboard/tabs.py`
  `tests/test_dashboard_tabs.py`
  `tasks/todo.md`
- Simplifications made:
  KR 투자자 수급 탭에서 σ 설명, 신호 변화 표, raw cue 표를 제거했다.
  최신 기준일의 섹터별 외국인/기관/개인/합계 순매수 금액만 표로 남겼다.
  금액은 원시 정수 대신 조/억/원 단위 문자열로 표시한다.
  기간별 섹터 순매수 금액 추이 그래프를 추가했다.
- Verification:
  `python -m py_compile src\dashboard\tabs.py tests\test_dashboard_tabs.py` -> passed
  `python -m pytest -q tests/test_dashboard_tabs.py --basetemp "$env:TEMP\sector-rotation-pytest"` -> `31 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py::test_build_kr_sector_flow_trend_figure_accepts_trade_date_index_name tests/test_dashboard_tabs.py --basetemp "$env:TEMP\sector-rotation-pytest"` -> `32 passed`
  `python -m pytest -q tests/test_ui_components.py::test_build_investor_flow_snapshot_rows_pivots_latest_snapshot tests/test_ui_components.py::test_build_investor_flow_snapshot_rows_keeps_raw_fields_and_adds_cues --basetemp "$env:TEMP\sector-rotation-pytest"` -> `2 passed`
  `http://localhost:8501/_stcore/health` -> `ok`
- Follow-up fix:
  실제 Streamlit 화면에서 `trade_date`가 인덱스 이름이자 컬럼 이름인 DataFrame이 들어와 기간별 그래프 groupby가 모호해졌다.
  그래프 빌더는 이제 인덱스 날짜를 내부 전용 `flow_date` 컬럼으로 복사해 groupby하고, 같은 형태를 회귀 테스트로 고정했다.
- Follow-up UX:
  기간별 그래프를 섹터 합계 라인 비교에서 선택 섹터 내 외국인/기관/개인 3개 라인 비교로 변경했다.
  섹터 선택 옵션은 최신 기준일 순매수 절대 합계가 큰 순서로 정렬한다.
  `python -m pytest -q tests/test_dashboard_tabs.py::test_build_kr_sector_flow_trend_figure_accepts_trade_date_index_name tests/test_dashboard_tabs.py::test_get_kr_flow_sector_options_sorts_by_latest_abs_total tests/test_dashboard_tabs.py --basetemp "$env:TEMP\sector-rotation-pytest"` -> `33 passed`
- Follow-up UX:
  섹터 선택 없이 전체 섹터를 한 번에 보도록, 기간별 그래프를 섹터별 small multiples 형태로 변경했다.
  각 섹터 패널 안에서 외국인/기관/개인 3개 라인을 같은 색상으로 반복 표시한다.
  `python -m pytest -q tests/test_dashboard_tabs.py::test_build_kr_sector_flow_trend_figure_accepts_trade_date_index_name tests/test_dashboard_tabs.py::test_render_investor_flow_tab_does_not_render_sector_selectbox tests/test_dashboard_tabs.py --basetemp "$env:TEMP\sector-rotation-pytest"` -> `34 passed`
- Remaining risks:
  Browser screenshot QA는 수행하지 않았다. 현재 로컬 Streamlit 서버는 이미 `http://localhost:8501`에서 실행 중이다.

# 2026-05-05 - Collection History UX Simplification

## Goal
- 데이터 수집 이력 화면을 시장/매크로/수급 데이터별 샘플 점검 중심으로 단순화한다.
- 각 데이터셋은 수집일시 내림차순 기준 최신 10건만 보여준다.
- 수집현황과 오류 원인 파악에 필요한 열만 남긴다.

## Checklist
- [x] 현재 이력 조회/렌더링 경로 확인
- [x] Autopilot context/PRD/test-spec 산출물 작성
- [x] warehouse 이력 조회에 provider/오류 요약 필드 포함
- [x] 모니터링 화면의 혼합 최근 15건 표를 데이터셋별 샘플 표로 단순화
- [x] focused 회귀 테스트 추가/갱신
- [x] py_compile 및 focused pytest 검증
- [x] 코드 리뷰 결과 기록

## Plan Artifacts
- `.omx/context/collection-history-ux-simplification-20260505T070000Z.md`
- `.omx/plans/prd-collection-history-ux-simplification.md`
- `.omx/plans/test-spec-collection-history-ux-simplification.md`

## Review
- Changed files:
  `src/data_sources/warehouse.py`
  `src/dashboard/tabs.py`
  `tests/test_warehouse_cli.py`
  `tests/test_dashboard_tabs.py`
  `.omx/context/collection-history-ux-simplification-20260505T070000Z.md`
  `.omx/plans/prd-collection-history-ux-simplification.md`
  `.omx/plans/test-spec-collection-history-ux-simplification.md`
  `tasks/todo.md`
- Simplifications made:
  기존 혼합 "최근 15건" 표를 제거하고, 시장데이터/매크로데이터/수급데이터별 최신 10건 샘플 표로 분리했다.
  운영 현황과 오류 점검도 수급 전용 블록에서 데이터셋별 공통 테이블로 바꿨다.
  각 데이터셋은 warehouse 쿼리에서 수집일시 내림차순 최신 10건만 가져오며, dashboard는 재샘플링 없이 표시만 한다.
  화면 열은 수집일시, 요청범위, 상태, 커버리지, 중단, 오류요약, 완료율, provider, 저장행수로 제한했다.
  예상요청/처리요청/이유 같은 노이즈 열은 제거하고, 오류 원인 파악에 필요한 failed days/codes와 abort reason은 `오류요약`으로 압축했다.
- Verification:
  `python -m py_compile src\data_sources\warehouse.py src\dashboard\tabs.py tests\test_warehouse_cli.py tests\test_dashboard_tabs.py` -> passed
  `python -m pytest -q --basetemp "$env:TEMP\sector-rotation-pytest" tests/test_dashboard_tabs.py tests/test_warehouse_cli.py` -> `43 passed`
  `git diff --check -- ...` -> passed with line-ending warnings only.
- Code review:
  Code-reviewer recommendation `APPROVE`; 0 findings.
  Architect initial status `WATCH` for split sampling ownership; follow-up moved sample ownership into warehouse via `sample_bucket`.
  Final architectural status `CLEAR`.
- Remaining risks:
  Visual browser screenshot validation was not run in this pass; behavior is covered by focused Streamlit renderer tests.
  Full repository pytest was not rerun because this task touched a narrow UI/query surface and the repo already had unrelated broader-test risk recorded earlier.

# 2026-05-05 - Investor Flow KRX Access Denied Refresh Fix

## Goal
- 투자자수급 갱신 버튼 실패 원인을 KRX trading-value endpoint의 403 Access Denied HTML 응답으로 분류한다.
- 실패 시 캐시 fallback은 유지하되, 사용자 알림은 KRX 접근차단/인증 문제와 조치사항을 명확히 보여준다.
- KRX 세션 상태가 access denied 이후에도 불필요하게 굳어져 인증 재시도를 막지 않도록 점검한다.

## Checklist
- [x] 수급 갱신 raw trading-value 호출부와 실패 summary 경로 확인
- [x] Access Denied 실패를 `ACCESS_DENIED` 계열로 분류
- [x] 투자자수급 갱신 notice에 KRX_ID/KRX_PW 또는 KRX 정책 차단 안내 노출
- [x] 회귀 테스트 추가
- [x] 관련 테스트 실행

## Review
- Root cause:
  KRX `getJsonData.cmd` trading-value endpoint returned 403 Access Denied HTML instead of JSON.
  The collector correctly stopped before pykrx wrapper spam, but the manual refresh summary stored the raw exception under `refresh`, so the dashboard notice treated it as a generic cached fallback.
- Changed files:
  `src/data_sources/pykrx_compat.py`
  `src/data_sources/krx_investor_flow.py`
  `src/dashboard/data.py`
  `tests/test_pykrx_compat.py`
  `tests/test_krx_investor_flow_data_source.py`
  `tests/test_dashboard_runtime.py`
- Simplifications:
  Access-denied refresh failures now normalize to `ACCESS_DENIED: ...` once, and the UI branches on that stable prefix instead of parsing the full builder exception.
  Endpoint-level access denial no longer permanently suppresses a later authenticated login attempt; only login-page access denial does.
- Verification:
  `python -m py_compile src\data_sources\pykrx_compat.py src\data_sources\krx_investor_flow.py src\dashboard\data.py tests\test_pykrx_compat.py tests\test_krx_investor_flow_data_source.py tests\test_dashboard_runtime.py` -> passed
  `python -m pytest -q tests/test_pykrx_compat.py tests/test_krx_investor_flow_data_source.py::test_run_manual_investor_flow_refresh_classifies_access_denied tests/test_dashboard_runtime.py::test_build_investor_flow_refresh_notice_surfaces_access_denied --basetemp "$env:TEMP\sector-rotation-pytest"` -> `17 passed`
  `python -m pytest -q tests/test_krx_investor_flow_data_source.py tests/test_dashboard_runtime.py tests/test_pykrx_compat.py --basetemp "$env:TEMP\sector-rotation-pytest"` -> `91 passed`
- Remaining risks:
  Live 수급 갱신은 여전히 KRX 비공식 endpoint 정책과 실제 계정 세션에 의존한다.
  `KRX_ID` / `KRX_PW` 설정 후에도 403이 계속되면 코드 문제가 아니라 KRX 측 endpoint 차단 상태로 봐야 한다.
- Follow-up after live log:
  User provided a live Streamlit log showing KRX login success followed by trading-value `status=403`.
  Exact app payload (`MDCSTAT02302/02303`, Samsung Electronics ISIN, sell/buy/net legs) succeeds after resetting and re-authenticating the shared KRX session.
  Updated `request_krx_data()` so endpoint Access Denied also triggers one session reset/relogin retry before failing.
  `python -m py_compile src\data_sources\pykrx_compat.py tests\test_pykrx_compat.py` -> passed
  `python -m pytest -q tests/test_pykrx_compat.py tests/test_krx_investor_flow_data_source.py::test_run_manual_investor_flow_refresh_classifies_access_denied tests/test_dashboard_runtime.py::test_build_investor_flow_refresh_notice_surfaces_access_denied --basetemp "$env:TEMP\sector-rotation-pytest"` -> `17 passed`
  Live payload probe for `005930` on `20260407` returned `200` and 1 row for general/detail sell, buy, and net after authenticated session reset.


# 2026-05-05 - Market Data KRX Empty Response Guard Review

## Goal
- 시장 데이터 수집 경로에 수급 수집과 유사한 빈/비JSON KRX 응답 반복 문제가 있는지 점검한다.
- OpenAPI와 PYKRX provider 경로를 분리해 취약 지점을 확인한다.
- 필요한 경우 반복 호출을 조기 중단하고 회귀 테스트로 고정한다.

## Checklist
- [x] OpenAPI 시장 데이터 응답 파싱/오류 분류 경로 확인
- [x] PYKRX 시장 데이터 refresh 경로의 반복 실패 가능성 확인
- [x] PYKRX 연속 endpoint payload 실패 circuit breaker 추가
- [x] 컴파일 및 focused 테스트 검증
- [x] 결과와 남은 리스크 기록

## Notes
- OpenAPI 경로는 이미 비JSON/Access Denied를 `KRXOpenAPIResponseError` 계열로 분류하고 `snapshot_failures`/`failed_days`에 보존한다.
- 보강 대상은 OpenAPI 키가 없거나 `KRX_PROVIDER=PYKRX`인 fallback 경로다.

## Review
- Finding:
  Manual market refresh normally goes through KRX OpenAPI when `KRX_OPENAPI_KEY` exists. That path already catches `resp.json()` failures, classifies Access Denied separately, retries boundedly, and records per-day failures in `snapshot_failures`/`failed_days`.
- Gap:
  The PYKRX fallback path could still attempt every requested index code when KRX returned the same unusable payload repeatedly. It did not retry deterministic JSON failures inside `_fetch_chunk`, but `_refresh_pykrx_raw_cache()` still moved on to the next code.
- Fix:
  Added a PYKRX endpoint-payload classifier and a 3-consecutive-failure circuit breaker in `_refresh_pykrx_raw_cache()`.
  Successful fetches reset the counter, so partial success remains supported.
- Changed files:
  `src/data_sources/krx_indices.py`
  `tests/test_integration.py`
  `tasks/todo.md`
- Verification:
  `python -m py_compile src\data_sources\krx_indices.py tests\test_integration.py` -> passed
  `python -m pytest -q tests/test_integration.py::TestIntegration::test_pykrx_refresh_circuit_breaks_repeated_empty_json_failures tests/test_integration.py::TestIntegration::test_partial_success_returns_cached_status` -> `2 passed`
  `python -m pytest -q tests/test_integration.py::TestIntegration::test_warm_sector_price_cache_refetches_raw_cache_with_empty_close_rows tests/test_integration.py::TestIntegration::test_warm_sector_price_cache_refetches_internal_empty_close_rows tests/test_integration.py::TestIntegration::test_load_sector_prices_raises_access_denied_instead_of_cached_fallback tests/test_integration.py::TestIntegration::test_openapi_auth_failure_falls_back_to_cache tests/test_integration.py::TestIntegration::test_api_failure_falls_back_to_cache` -> `5 passed`
  `python -m pytest -q tests/test_integration.py tests/test_krx_indices.py tests/test_data_status.py` -> `56 passed`
- Remaining risks:
  PYKRX live success still depends on KRX/pykrx returning usable OHLCV frames.
  OpenAPI remains the preferred market-data provider when `KRX_OPENAPI_KEY` is configured.

# 2026-05-05 - Investor Flow KRX Empty JSON Guard

## Goal
- 수급 데이터 수집 버튼에서 반복되는 `Expecting value: line 1 column 1 (char 0)` 원인을 특정한다.
- KRX/pykrx 비JSON 또는 빈 응답을 종목별 반복 로그로 방치하지 않는다.
- 실패 원인을 수집 summary/UI notice/warehouse 실패 기록까지 보존한다.

## Checklist
- [x] 수급 버튼 호출 경로와 pykrx 호출 래퍼 확인
- [x] 빈/비JSON 응답 재현 형태를 테스트로 고정
- [x] KRX 응답 이상을 분류하고 조기 중단 또는 명확한 실패 detail로 전파
- [x] focused 테스트와 컴파일 검증
- [x] 결과와 남은 리스크 기록

## Notes
- 새 의존성은 추가하지 않는다.
- 기존 캐시 fallback 및 partial preview 동작은 유지한다.

## Review
- Root cause:
  KRX trading-value endpoint sometimes returns an empty or non-JSON body after login. The raw KRX path retried once, but the investor-flow collector could then fall through to pykrx wrapper fallback. pykrx prints `Error occurred in get_market_trading_value_and_volume_on_ticker_by_date: Expecting value...` internally and returns empty frames, so the same endpoint failure was repeated for every tracked ticker.
- Changed files:
  `src/data_sources/pykrx_compat.py`
  `src/data_sources/krx_investor_flow.py`
  `tests/test_pykrx_compat.py`
  `tests/test_krx_investor_flow_data_source.py`
  `tasks/todo.md`
- Simplifications made:
  Empty/non-JSON KRX JSON responses now use a dedicated `KRXInvalidPayloadError`.
  Investor-flow trading-value collection now treats deterministic KRX payload failures as whole-refresh endpoint failures instead of trying pykrx wrapper fallback.
  A circuit breaker stops after three consecutive fully empty ticker results, covering the case where pykrx swallowed the JSON error and returned empty frames.
- Verification:
  `python -m py_compile src\data_sources\pykrx_compat.py src\data_sources\krx_investor_flow.py tests\test_pykrx_compat.py tests\test_krx_investor_flow_data_source.py` -> passed
  `python -m pytest -q tests/test_pykrx_compat.py tests/test_krx_investor_flow_data_source.py` -> `62 passed`
  `python -m pytest -q tests/test_krx_constituents.py tests/test_dashboard_runtime.py tests/test_dashboard_data.py` -> `52 passed`
- Remaining risks:
  Live success still depends on KRX returning usable JSON for the unofficial trading-value endpoint.
  When KRX itself blocks or changes this endpoint, the app now fails fast with a concise refresh failure instead of flooding logs.

# 2026-05-05 - Unified Collection History

## Goal
- 데이터 수집 이력 페이지에서 시장데이터, 매크로데이터, 수급데이터 수집 이력을 함께 관리한다.
- 기존 투자자 수급 운영 현황, 커버리지, 오류 점검 화면 동작은 유지한다.
- 새 저장소나 의존성 없이 기존 `ingest_runs` 공통 테이블을 재사용한다.

## Checklist
- [x] 현재 이력 조회/렌더링 경로 확인
- [x] 공통 수집 이력 조회 함수 추가
- [x] 데이터 수집 이력 테이블에 데이터셋 구분 표시
- [x] 수급 전용 runtime fallback 유지
- [x] 회귀 테스트 추가 및 검증
- [x] 결과 기록

## Notes
- 화면의 운영 현황/커버리지/오류 점검은 현재 수급 데이터 중심 스냅샷으로 남긴다.
- "수집 이력" 섹션만 세 데이터셋 공통 이력으로 확장한다.

## Review
- Changed files:
  `src/data_sources/warehouse.py`
  `src/dashboard/tabs.py`
  `tests/test_warehouse_cli.py`
  `tests/test_dashboard_tabs.py`
  `tasks/todo.md`
- Simplifications made:
  Existing `ingest_runs` is now queried through a shared `read_collection_run_history()` helper.
  The old `read_investor_flow_run_history()` remains as a wrapper, so existing investor-flow callers keep their old shape.
  The monitoring page keeps the current 수급 status/coverage/error sections and expands only the history table.
- Verification:
  `python -m py_compile src\data_sources\warehouse.py src\dashboard\tabs.py tests\test_warehouse_cli.py tests\test_dashboard_tabs.py` -> passed
  `python -m pytest -q tests/test_warehouse_cli.py::test_read_collection_run_history_returns_market_macro_and_flow_runs tests/test_dashboard_tabs.py::test_render_monitoring_tab_shows_unified_collection_history tests/test_dashboard_tabs.py::test_render_monitoring_tab_uses_runtime_flow_snapshot_when_warehouse_history_is_empty` -> `3 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py tests/test_warehouse_cli.py tests/test_warehouse_investor_flow.py` -> `46 passed`
  `python -m pytest -q` -> `551 passed, 1 failed`
- Remaining risks:
  Full pytest still has an unrelated standalone failure in `tests/test_us_signal_pipeline.py::test_cached_signals_supports_us_market`: the US benchmark price frame is treated as missing, producing `N/A` instead of `Watch`.
  The working tree already contains many unrelated modified/deleted files; this change did not revert or normalize them.

# 2026-05-05 - Page Header Title Routing

## Goal
- 상단 헤더 제목이 모든 페이지에서 `섹터 로테이션 리서치`로 고정되지 않게 한다.
- 현재 선택된 시장과 페이지에 맞는 제목을 표시한다.
- 기존 페이지 이동, 데이터 로딩, 본문 렌더링 동작은 유지한다.

## Checklist
- [x] 네비게이션 상태와 헤더 렌더링 위치 확인
- [x] 페이지 제목 해석 헬퍼 추가
- [x] 앱 헤더에 현재 페이지 제목 연결
- [x] focused 테스트와 컴파일 검증
- [x] 결과 기록

## Review
- Changed files:
  `app.py`
  `src/dashboard/tabs.py`
  `tests/test_dashboard_tabs.py`
  `tasks/todo.md`
- Simplifications made:
  Page labels already used by Streamlit navigation are reused for the visible header title.
  No new routing state, dependency, or UI component was added.
- Verification:
  `python -m py_compile app.py src\dashboard\tabs.py tests\test_dashboard_tabs.py` -> passed
  `python -m pytest -q tests\test_dashboard_tabs.py` -> `28 passed`
  `python -m pytest -q tests/test_ui_components.py::test_render_page_header_renders_shell_markup tests/test_ui_components.py::test_render_page_header_avoids_forbidden_brokerage_or_live_claims` -> `2 passed`
- Remaining risks:
  The browser tab title from `st.set_page_config` remains the app-level title; this change targets the in-page top header.

# 2026-05-05 - Constituents Loading Performance

## Goal
- "구성종목" 페이지 진입 시 느린 KRX/pykrx 라이브 호출을 자동 실행하지 않는다.
- 기존 캐시가 있으면 즉시 표시하고, 캐시가 없으면 수동 `데이터 갱신`을 안내한다.
- 명시적 갱신 버튼의 라이브 수집 동작은 유지한다.

## Checklist
- [x] 현재 구성종목 렌더링 및 데이터 로딩 경로 확인
- [x] PRD/test-spec 산출물 작성
- [x] cache-only loader 옵션 구현
- [x] 구성종목 페이지 초기 렌더에서 cache-only 사용
- [x] 회귀 테스트 추가
- [x] py_compile 및 focused pytest 검증
- [x] 코드 리뷰 결과 기록

## Plan Artifacts
- `.omx/context/constituents-loading-performance-20260505T034737Z.md`
- `.omx/plans/prd-constituents-loading-performance.md`
- `.omx/plans/test-spec-constituents-loading-performance.md`

## Review
- Changed files:
  `src/data_sources/krx_stock_screening.py`
  `src/dashboard/tabs.py`
  `tests/test_krx_stock_screening.py`
  `tests/test_dashboard_tabs.py`
  `.omx/context/constituents-loading-performance-20260505T034737Z.md`
  `.omx/plans/prd-constituents-loading-performance.md`
  `.omx/plans/test-spec-constituents-loading-performance.md`
  `tasks/todo.md`
- Simplifications made:
  Initial 구성종목 render now uses existing pickle caches only.
  The expensive KRX/pykrx live path remains behind the existing `데이터 갱신` button.
  No dependency, background worker, or cache format change was added.
- Verification:
  `python -m py_compile src\data_sources\krx_stock_screening.py src\dashboard\tabs.py tests\test_krx_stock_screening.py tests\test_dashboard_tabs.py` -> passed
  `python -m pytest -q tests\test_krx_stock_screening.py tests\test_dashboard_tabs.py` -> `38 passed`
  `python -m pytest -q tests\test_dashboard_runtime.py tests\test_dashboard_data.py tests\test_krx_constituents.py` -> `51 passed`
- Code review:
  Recommendation `APPROVE`; architectural status `CLEAR`.
  No unresolved blocker found in the cache-only boundary, refresh path, or regression coverage.
- Remaining risks:
  First-time users with no cache must press `데이터 갱신` once to populate 구성종목 data.
  Existing cache keys still follow the prior sector-list based contract and do not include all settings.

# 2026-05-05 - UI Scale Rebalance

## Goal
- 웹에서 작게 느껴지는 Streamlit 대시보드의 폰트 크기와 컴포넌트 비율을 적정 수준으로 키운다.
- 기존 정보 구조, 데이터 로직, 네비게이션 동작은 유지한다.
- 한글 UI에서 과밀하거나 잘리는 요소가 생기지 않도록 검증한다.

## Checklist
- [x] 스타일 진입점과 기존 토큰 구조 확인
- [x] 전역 타이포그래피/레이아웃 토큰 상향
- [x] 주요 컴포넌트(버튼, 탭, 카드, 표, 모바일 헤더) 스케일 보정
- [x] 관련 테스트 기대값 갱신
- [x] py_compile, UI/theme 테스트, Streamlit 시각 검증 실행
- [x] 결과와 남은 리스크 기록

## Notes
- 이번 작업은 비율 조정이 목적이므로 새 의존성이나 데이터 경로 변경은 하지 않는다.
- 너무 큰 리디자인 대신, 기존 `config/theme.py` 토큰과 `src/ui/css.py` CSS 계층에서 조정한다.

## Review
- Changed files:
  `config/theme.py`
  `src/ui/css.py`
  `tests/test_ui_theme.py`
  `tasks/todo.md`
- Simplifications made:
  Light theme typography now uses a larger, steadier dashboard scale instead of the previous compact text scale.
  Buttons, tabs, sidebar navigation, status strips, cards, tables, chips, and mobile page headers now share the enlarged scale.
  Existing Streamlit routing, data loaders, dashboard panels, and dependencies were left unchanged.
- Verification:
  `.venv\Scripts\python.exe -m py_compile config\theme.py src\ui\css.py tests\test_ui_theme.py` -> passed
  `python -m pytest -q tests\test_ui_theme.py tests\test_ui_contrast.py tests\test_ui_components.py` -> `92 passed`
  `python -m pytest -q tests\test_dashboard_tabs.py tests\test_dashboard_runtime.py` -> `50 passed`
  `python -m py_compile app.py config\theme.py src\ui\css.py src\ui\panels.py src\dashboard\tabs.py tests\test_ui_theme.py` -> passed
  `python scripts\capture_visual_eval.py` -> 8 screenshots regenerated, no horizontal overflow, no Page not found, no `position_mode` warning.
- Visual evidence:
  `.omx/artifacts/visual-eval/overview-desktop.png`
  `.omx/artifacts/visual-eval/overview-mobile.png`
  `.omx/artifacts/visual-eval/signals-desktop.png`
  `.omx/artifacts/visual-eval/layout-report.json`
- Remaining risks:
  The layout report still flags many generic Streamlit internal elements as overflowing/offscreen, especially on mobile, but document-level horizontal overflow is false across all captured routes.
  The local `.venv` does not include `pytest`; tests were run with the available `python` environment that has the project test dependencies.

# 2026-05-10 - Macro Alias Completion Rate

## Goal
- 데이터 수집 이력의 매크로데이터 완료율을 요청 카운터가 아니라 alias 단위 커버리지로 표시한다.
- 시장/수급 데이터의 기존 요청 처리율 표시는 유지한다.

## Checklist
- [x] 매크로 alias 완료율 계산 위치 확인
- [x] warehouse 이력 조회에 매크로 전용 완료율 적용
- [x] 모니터링 UI 테스트 갱신
- [x] py_compile 및 focused pytest 검증

## Review
- Changed files:
  `src/data_sources/warehouse.py`
  `tests/test_warehouse_cli.py`
  `tasks/todo.md`
- Fix:
  `macro_data` 수집 이력은 `predicted_requests / processed_requests` 기본값 대신 provider별 enabled alias의 월별 커버리지 비율을 `completion_pct`로 계산한다.
  시장데이터와 수급데이터의 기존 요청 처리율 계산은 유지했다.
- Verification:
  `python -m py_compile src\data_sources\warehouse.py tests\test_warehouse_cli.py src\dashboard\tabs.py` -> passed
  `python -m pytest -q tests/test_warehouse_cli.py::test_read_collection_run_history_uses_macro_alias_completion_rate tests/test_warehouse_cli.py::test_read_collection_run_history_returns_market_macro_and_flow_runs tests/test_dashboard_tabs.py::test_render_monitoring_tab_shows_dataset_sample_history` -> `3 passed`
  로컬 warehouse 확인: 2026-05-10 KR 매크로 이력은 ECOS `33.3%`, KOSIS `0.0%`로 표시된다.
- Remaining risks:
  매크로 provider별 완료율은 alias가 요청 범위의 모든 월을 가진 경우만 완료로 본다. 최신월 공표 지연이 있는 alias는 부분 완료로 남는다.

# 2026-05-10 - Manual Refresh Collection History Filter

## Goal
- 데이터 수집 이력 화면에는 사용자/운영 수집 이력인 `manual_refresh`만 표시한다.
- 화면 분석용 자동 매크로 로드 이력은 warehouse에 남기되 모니터링 샘플에서는 제외한다.

## Checklist
- [x] 수집 이력 조회 함수에 reason 필터 추가
- [x] 모니터링 화면 조회를 `manual_refresh`로 제한
- [x] 회귀 테스트 추가
- [x] py_compile 및 focused pytest 검증

## Review
- Changed files:
  `src/data_sources/warehouse.py`
  `src/dashboard/tabs.py`
  `tests/test_warehouse_cli.py`
  `tests/test_dashboard_tabs.py`
  `tasks/todo.md`
- Fix:
  `read_collection_run_history()`에 `reasons` 필터를 추가했다.
  데이터 수집 이력 화면의 cached monitoring data는 `reasons=("manual_refresh",)`만 조회한다.
- Verification:
  `python -m py_compile src\data_sources\warehouse.py src\dashboard\tabs.py tests\test_warehouse_cli.py tests\test_dashboard_tabs.py` -> passed
  `python -m pytest -q tests/test_warehouse_cli.py::test_read_collection_run_history_filters_reasons_before_sampling tests/test_warehouse_cli.py::test_read_collection_run_history_can_sample_latest_ten_per_dataset tests/test_dashboard_tabs.py::test_cached_monitoring_data_reads_manual_refresh_history tests/test_dashboard_tabs.py::test_render_monitoring_tab_shows_dataset_sample_history` -> `4 passed`
  로컬 warehouse 확인: 매크로 이력 샘플에서 `load_ecos_macro`/`load_kosis_macro`가 빠지고 `manual_refresh` 범위만 표시된다.

# 2026-05-05 - Market Cache Close Coalesce

## Goal
- 앱 실행 시 표시되는 market data warehouse-cache fallback 경고의 원인을 고친다.
- `20260504`처럼 실제 OpenAPI/raw cache 데이터가 있는 날짜는 warehouse 변환에서 누락되지 않게 한다.
- `20260501`처럼 OpenAPI가 실제로 빈 행을 반환하는 비거래일은 데이터 없음으로 구분한다.

## Checklist
- [x] 실패 날짜와 실패 종목 원인 조사
- [x] Ralph context/PRD/test-spec 산출물 작성
- [x] mixed raw close 컬럼 회귀 테스트 추가
- [x] row-wise close coalesce 구현
- [x] targeted pytest 및 py_compile 검증
- [x] read-only raw cache 드라이런 검증
- [x] deslop/self-review 및 최종 결과 기록

## Plan Artifacts
- `.omx/context/market-cache-close-coalesce-20260505T055025Z.md`
- `.omx/plans/prd-market-cache-close-coalesce.md`
- `.omx/plans/test-spec-market-cache-close-coalesce.md`

## Review
- Changed files:
  `src/data_sources/krx_indices.py`
  `tests/test_integration.py`
  `.omx/context/market-cache-close-coalesce-20260505T055025Z.md`
  `.omx/plans/prd-market-cache-close-coalesce.md`
  `.omx/plans/test-spec-market-cache-close-coalesce.md`
  `tasks/todo.md`
- Root cause:
  Long raw cache windows contained many historical `종가` values and recent OpenAPI `close` values.
  The old code selected one close column by total non-null count, so latest `close` rows such as `20260504` were ignored.
- Fix:
  Raw cache close values are now coalesced row by row across close candidates.
  `_valid_close_raw_cache`, `_build_sector_frame`, and `_raw_cache_signature` share that same close series.
  Rows with no usable close are excluded before sector-price validation.
- Verification:
  `python -m py_compile src\data_sources\krx_indices.py tests\test_integration.py` -> passed
  `python -m pytest tests/test_integration.py::TestIntegration::test_raw_cache_coalesces_recent_close_over_historical_korean_close -q` -> `1 passed`
  `python -m pytest tests/test_integration.py::TestIntegration::test_warm_sector_price_cache_refetches_raw_cache_with_empty_close_rows tests/test_integration.py::TestIntegration::test_warm_sector_price_cache_refetches_internal_empty_close_rows tests/test_integration.py::TestIntegration::test_build_sector_frame_prefers_valid_close_over_empty_korean_close tests/test_integration.py::TestIntegration::test_compute_missing_ranges_does_not_expand_future_only_cache -q` -> `4 passed`
  `python -m pytest tests/test_data_status.py -q` -> `16 passed`
  `python -m pytest tests/test_integration.py tests/test_data_status.py -q` -> `51 passed`
  Read-only raw cache dry run -> active KR codes `18`, codes with `2026-05-04` row `18`, missing latest `[]`.
  OpenAPI snapshot check -> `20260501` KOSPI/KRX rows `0/0`; `20260504` KOSPI/KRX rows `50/34`.
  Local warehouse warm after releasing Streamlit lock -> status `CACHED`, `coverage_complete=True`, `failed_codes={}`, `failed_days=[]`, watermark `20260504`.
  Warehouse verification for `20260504` -> active KR codes `18`, rows `18`, missing codes `[]`.
- Deslop/self-review:
  Scope stayed limited to the raw close conversion path and one focused regression.
  No new dependency, schema change, or warehouse write was added.
- Remaining risks:
  The local warehouse is now refreshed through `20260504`.
  Generic business-day missing-range output still lists KRX holidays such as `20260501`; this fix only prevents real trading-day latest rows from being dropped.

# 2026-05-05 - Data Work Buttons

## Goal
- `데이터 작업` 영역의 시장/매크로/투자자수급/전체 재계산 버튼 동작을 점검한다.
- 동작이 보이지 않거나 실패 피드백이 약한 버튼을 개선한다.
- 기존 데이터 갱신 및 캐시 무효화 경로는 유지한다.

## Checklist
- [x] 버튼 렌더링과 `app.py` 핸들러 연결 확인
- [x] 시장/매크로/투자자수급 갱신 런타임 테스트 실행
- [x] 전체 재계산의 피드백 누락 결함 식별
- [x] PRD/test-spec 산출물 작성
- [x] 전체 재계산 진행 이벤트 및 완료 notice 구현
- [x] rerun 이후 notice 표시 보존 구현
- [x] py_compile 및 focused pytest 검증
- [x] 코드 리뷰 결과 기록

## Plan Artifacts
- `.omx/context/data-work-buttons-20260505T061603Z.md`
- `.omx/plans/prd-data-work-buttons.md`
- `.omx/plans/test-spec-data-work-buttons.md`

## Review
- Changed files:
  `app.py`
  `src/dashboard/runtime.py`
  `tests/test_dashboard_runtime.py`
  `.omx/context/data-work-buttons-20260505T061603Z.md`
  `.omx/plans/prd-data-work-buttons.md`
  `.omx/plans/test-spec-data-work-buttons.md`
  `tasks/todo.md`
- Simplifications made:
  시장/매크로/투자자수급 갱신 경로는 기존 구조를 유지했다.
  `전체 재계산`만 기존 progress/toast 패턴에 맞춰 확장했다.
  Streamlit rerun을 바꾸지 않고 `st.session_state`에 notice만 임시 저장했다.
- Verification:
  `python -m pytest -q tests/test_dashboard_runtime.py tests/test_dashboard_tabs.py tests/test_data_status.py` -> `68 passed` before implementation, confirming the existing three refresh paths.
  `python -m py_compile app.py src\dashboard\runtime.py tests\test_dashboard_runtime.py` -> passed
  `python -m pytest -q tests/test_dashboard_runtime.py tests/test_dashboard_tabs.py tests/test_data_status.py` -> `69 passed`
- Code review:
  Initial review returned `COMMENT` / architectural `WATCH` because the recompute notice overstated that signals were already recalculated and the static test did not prove ordering.
  Follow-up changed the notice to say signals will recalculate on the next load and strengthened the test to assert `pop < store < rerun < toast`.
  Final recommendation `APPROVE`; architectural status `CLEAR`.
- Remaining risks:
  External provider live refresh success still depends on API/provider availability and credentials.

# 2026-05-10 - Market Refresh History Label Check

## Goal
- 사이드바 `시장데이터 갱신` 버튼이 실제 시장 데이터 갱신 경로를 호출하는지 확인한다.
- 데이터 수집 이력 화면에서 `매크로데이터`로 표시되는 원인이 버튼 매칭인지 화면/캐시 표시인지 구분한다.

## Checklist
- [x] 사이드바 버튼 반환값과 `app.py` 핸들러 연결 확인
- [x] 시장/매크로 갱신 함수가 기록하는 warehouse dataset 값 확인
- [x] 데이터 수집 이력 화면의 조회/캐시/라벨 매핑 확인
- [x] 필요한 경우 회귀 테스트와 수정 적용
- [x] py_compile 및 focused pytest 검증

## Review
- Root cause:
  `시장데이터 갱신` 버튼은 `run_market_refresh()`를 호출하고, KR 경로는 `dataset="market_prices"`로 `ingest_runs`에 기록한다.
  로컬 warehouse 최신 이력도 `2026-05-10 22:37:09+09:00 market_prices manual_refresh OPENAPI`로 확인됐다.
  따라서 버튼 기능 매칭은 정상이다.
- Fix:
  수동 갱신 후 데이터 수집 이력 화면의 60초 `st.cache_data` 캐시가 남지 않도록 monitoring cache clear 경로를 추가했다.
  시장/매크로/수급/all 캐시 무효화 시 이력 화면 캐시도 같이 비운다.
- Changed files:
  `src/dashboard/runtime.py`
  `src/dashboard/tabs.py`
  `tests/test_dashboard_runtime.py`
  `tasks/todo.md`
- Verification:
  `python -m py_compile src\dashboard\runtime.py src\dashboard\tabs.py tests\test_dashboard_runtime.py` -> passed
  `python -m pytest -q tests/test_dashboard_runtime.py::test_invalidate_dashboard_caches_scopes tests/test_dashboard_runtime.py::test_run_market_refresh_returns_notice_and_invalidates tests/test_dashboard_runtime.py::test_run_macro_refresh_returns_notice_and_invalidates tests/test_dashboard_tabs.py::test_render_monitoring_tab_shows_dataset_sample_history` -> `4 passed`
  `python -m pytest -q tests/test_dashboard_runtime.py` -> `27 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py::test_render_monitoring_tab_splits_sector_and_ticker_failures tests/test_dashboard_tabs.py::test_render_monitoring_tab_separates_other_collection_errors tests/test_dashboard_tabs.py::test_render_monitoring_tab_keeps_warm_status_when_runtime_status_is_omitted tests/test_dashboard_tabs.py::test_render_monitoring_tab_uses_runtime_flow_snapshot_when_warehouse_history_is_empty tests/test_dashboard_tabs.py::test_render_monitoring_tab_shows_dataset_sample_history` -> `5 passed`
- Note:
  One attempted pytest command used removed/renamed test ids and ran no tests; it was replaced with the actual current test ids above.

# 2026-05-13 - KR Constituents Refresh Spinner Check

## Goal
- KR 구성종목 화면의 `데이터 갱신` 버튼이 계속 spinner 상태로 남는 원인을 점검한다.
- 실제 네트워크 대기 결함인지, 단순 진행률 표시 부족인지 구분한다.
- 무한 대기를 막는 최소 수정과 회귀 테스트를 적용한다.

## Checklist
- [x] 구성종목 버튼 렌더링과 live loader 연결 확인
- [x] pykrx/KRX transport timeout 경로 확인
- [x] timeout 결함 수정
- [x] focused pytest 검증
- [x] 결과 기록

## Review
- Root cause:
  KR 구성종목 화면의 `데이터 갱신`은 `load_screened_stocks()`에서 pykrx wrapper 경로를 통해 구성종목과 종목별 OHLCV를 조회한다.
  그런데 `ensure_pykrx_transport_compat()`가 pykrx `Post.read`/`Get.read`를 공유 `requests.Session`으로 교체하면서 `timeout`을 넘기지 않았다.
  KRX/pykrx 요청 하나가 응답을 멈추면 Streamlit spinner가 계속 도는 상태로 보일 수 있었다.
- Fix:
  pykrx 공유 세션 read 경로에 `PYKRX_SHARED_SESSION_TIMEOUT = 15`를 적용했다.
  raw KRX `request_krx_data()` 경로처럼 wrapper 호출도 무한 대기하지 않고 실패/예외 경로로 빠질 수 있게 했다.
- Changed files:
  `src/data_sources/pykrx_compat.py`
  `tests/test_pykrx_compat.py`
  `tasks/todo.md`
- Verification:
  `python -m py_compile src\data_sources\pykrx_compat.py tests\test_pykrx_compat.py` -> passed
  `python -m pytest -q tests/test_pykrx_compat.py --basetemp .pytest-tmp-kr-spinner` -> `17 passed`
  `python -m pytest -q tests/test_krx_stock_screening.py --basetemp .pytest-tmp-kr-screening` -> `11 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "render_screening_tab" --basetemp .pytest-tmp-kr-tabs` -> `3 passed, 34 deselected`
  `git diff --check -- src/data_sources/pykrx_compat.py tests/test_pykrx_compat.py` -> passed with line-ending warnings only
- Note:
  First pytest attempt without `--basetemp` failed before tests because the existing `.pytest-tmp` directory was locked by Windows/OneDrive permissions.

# 2026-05-13 - KR Constituents Stock OHLCV Cache

## Goal
- 구성종목 화면의 `데이터 갱신`이 개별 종목 가격을 매번 pykrx로 전량 재조회하지 않게 한다.
- 개별 주식 OHLCV를 warehouse에 저장하고, 스크리닝은 DB cache-hit를 먼저 사용한다.
- 스크리닝 가격 구간을 최근 120일에서 최근 1년으로 늘려 200일 이평선 계산이 가능하게 한다.

## Checklist
- [x] 현재 구성종목 스크리닝 캐시/DB 구조 확인
- [x] 개별 주식 OHLCV warehouse table/read/upsert 추가
- [x] 스크리닝 live path를 DB 우선 + 누락분 보충으로 변경
- [x] 1년 가격 구간과 200DMA 계산 계약 추가
- [x] focused pytest 및 compile 검증
- [x] 결과 기록

## Review
- Root cause:
  구성종목 스크리닝은 결과 pickle cache만 있었고, 종목별 OHLCV는 매 refresh 때 pykrx에서 직접 조회했다.
  따라서 개별 종목 가격 데이터가 warehouse에 이미 있어도 스크리닝 조회 병목을 줄이는 구조가 아니었다.
- Fix:
  `fact_kr_stock_ohlcv_daily` warehouse table과 read/upsert API를 추가했다.
  스크리닝 종목별 가격 로더는 warehouse hit를 먼저 사용하고, 200일 이동평균 계산에 필요한 최소 200개 종가와 최신성 조건이 부족할 때만 pykrx live 조회 후 DB에 저장한다.
  가격 조회 구간은 최근 120일에서 최근 365일로 변경했다.
  구성종목 화면에 `200DMA↑` 체크 컬럼도 추가해 200일선 상회 여부를 볼 수 있게 했다.
- Changed files:
  `src/data_sources/warehouse.py`
  `src/data_sources/krx_stock_screening.py`
  `src/dashboard/tabs.py`
  `tests/test_warehouse_investor_flow.py`
  `tests/test_krx_stock_screening.py`
  `tasks/todo.md`
- Verification:
  `python -m py_compile src\data_sources\warehouse.py src\data_sources\krx_stock_screening.py src\dashboard\tabs.py tests\test_warehouse_investor_flow.py tests\test_krx_stock_screening.py` -> passed
  `python -m pytest -q tests/test_warehouse_investor_flow.py tests/test_krx_stock_screening.py --basetemp %TEMP%\pytest-sector-stock-ohlcv-*` -> `20 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "render_screening_tab" --basetemp %TEMP%\pytest-sector-tabs-*` -> `3 passed, 34 deselected`
  `python -m pytest -q tests/test_warehouse_cli.py::test_ensure_warehouse_schema_is_idempotent tests/test_warehouse_cli.py::test_read_dataset_status_skips_schema_write_when_read_schema_ready tests/test_warehouse_multimarket.py::test_market_and_macro_rows_are_scoped_by_market --basetemp %TEMP%\pytest-sector-warehouse-core-*` -> `3 passed`
  `git diff --check -- src/data_sources/warehouse.py src/data_sources/krx_stock_screening.py src/dashboard/tabs.py tests/test_warehouse_investor_flow.py tests/test_krx_stock_screening.py tasks/todo.md` -> passed with line-ending warnings only
- Note:
  One pytest attempt under the OneDrive repo path failed because OneDrive held a temporary DuckDB file lock.
  Re-running the same focused tests under `%TEMP%` passed.

# 2026-05-13 - KR Constituents Refresh Progress

## Goal
- 구성종목 화면에서 `데이터 갱신`을 눌렀을 때 전체 갱신 대상 종목 수와 현재 처리 중인 종목 순번을 표시한다.
- 초기 화면의 cache-only 로딩은 불필요한 진행률 UI를 띄우지 않는다.
- 스크리닝 루프 진행 상태를 테스트로 고정한다.

## Checklist
- [x] 현재 구성종목 refresh UI와 스크리닝 루프 연결 확인
- [x] 스크리닝 loader에 progress callback 계약 추가
- [x] Streamlit progress/status 표시 연결
- [x] focused pytest 및 compile 검증
- [x] 결과 기록

## Review
- Fix:
  `load_screened_stocks()`와 `_fetch_and_score()`에 progress callback 계약을 추가했다.
  구성종목 갱신 대상 종목을 먼저 산정한 뒤, 각 종목 처리 직전에 `current`, `total`, `ticker`, `sector_name` 이벤트를 보낸다.
  화면에서는 `데이터 갱신`을 누른 live refresh일 때만 progress bar와 상태 문구를 표시한다.
  표시 문구는 `갱신 대상 총 N종목 중 M번째 처리 중: 종목 · 섹터` 형태다.
- Changed files:
  `src/data_sources/krx_stock_screening.py`
  `src/dashboard/tabs.py`
  `tests/test_krx_stock_screening.py`
  `tests/test_dashboard_tabs.py`
  `tasks/todo.md`
- Verification:
  `python -m py_compile src\data_sources\krx_stock_screening.py src\dashboard\tabs.py tests\test_krx_stock_screening.py tests\test_dashboard_tabs.py` -> passed
  `python -m pytest -q tests/test_krx_stock_screening.py --basetemp %TEMP%\pytest-sector-progress-screening-*` -> `14 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "render_screening_tab" --basetemp %TEMP%\pytest-sector-progress-tabs-*` -> `3 passed, 34 deselected`
  `python -m pytest -q tests/test_warehouse_investor_flow.py tests/test_krx_stock_screening.py --basetemp %TEMP%\pytest-sector-progress-stock-ohlcv-*` -> `21 passed`
  `git diff --check -- src/data_sources/krx_stock_screening.py src/dashboard/tabs.py tests/test_krx_stock_screening.py tests/test_dashboard_tabs.py tasks/todo.md` -> passed with line-ending warnings only
# 2026-05-13 - Composite Review Candidates Consensus Plan

## Goal
- 모멘텀 단독 `검토 후보`를 복합점수 기반 후보 랭킹으로 바꾸기 위한 실행 계획을 확정한다.
- canonical KR action 정책은 그대로 두고, overview 후보 selection/projection만 1차 범위로 제한한다.

## Checklist
- [x] deep-interview spec 확인
- [x] 현재 후보 생성, KR action, flow, UI test 근거 확인
- [x] Planner 초안 작성
- [x] Architect 검토 반영
- [x] Critic reject 피드백 반영
- [x] Architect 재검토 반영
- [x] Critic 최종 승인
- [x] PRD/test-spec/ralplan 산출물 저장

## Review
- Decision:
  1차 구현은 `SectorSignal.action`, `SectorSignal.action_policy`, `compute_kr_action()`을 바꾸지 않는다.
  대신 overview `검토 후보`에 candidate-only projection을 두고 `COMPOSITE_REVIEW_CANDIDATE` 정책 메타데이터와 composite score를 표시한다.
- Artifacts:
  `.omx/plans/prd-composite-review-candidates-20260513T134500Z.md`
  `.omx/plans/test-spec-composite-review-candidates-20260513T134500Z.md`
  `.omx/plans/ralplan-composite-review-candidates-20260513T134500Z.md`
- Verification:
  Artifact existence and final critic approval confirmed.

# 2026-05-13 - Composite Review Candidates Ralph Implementation

## Goal
- overview `검토 후보`를 모멘텀 단독 순서가 아니라 composite review score 순서로 산출한다.
- canonical `SectorSignal.action`, `action_policy`, `compute_kr_action()` 계약은 변경하지 않는다.
- 후보 카드에 복합점수, 유형, 모멘텀/매크로/수급 기여도와 약점 신호를 노출한다.

## Checklist
- [x] Ralph context snapshot 생성
- [x] 기존 dirty worktree에서 관련 파일 범위 확인
- [x] composite candidate projection helper 추가
- [x] overview 후보 builder를 composite score 정렬로 변경
- [x] 후보 renderer copy와 metric 노출 갱신
- [x] numeric formula, non-finite guard, policy preservation test 추가
- [x] focused compile/test 검증
- [x] architect verification 승인
- [x] deslop pass 및 post-deslop regression

## Review
- Changed files:
  `src/ui/panels.py`
  `tests/test_ui_components.py`
  `tasks/todo.md`
- Result:
  `_build_overview_review_candidates()`가 eligible signal 전체를 candidate projection으로 점수화한 뒤 `candidate_score desc`, `momentum_score desc`, `sector_name asc` 순서로 정렬하고 `limit`을 적용한다.
  candidate DTO는 `COMPOSITE_REVIEW_CANDIDATE`, composite/component scores, availability, warnings, canonical `action`/`action_policy` metadata를 포함한다.
  `SectorSignal`과 KR action 함수/계약은 건드리지 않았다.
- Deslop:
  Ralph-owned scope에서 masking fallback은 발견하지 못했다.
  non-finite guard는 `math.isfinite()`로 고정했고 `inf`/`-inf` regression을 추가했다.
- Verification:
  `python -m py_compile src\ui\panels.py tests\test_ui_components.py` -> passed
  `python -m pytest -q tests/test_ui_components.py -k "overview_review_candidate" --basetemp "$env:TEMP\pytest-composite-review-ui"` -> `5 passed, 73 deselected`
  `python -m pytest -q tests/test_signals.py tests/test_signal_pipeline_integration.py -k "kr_action or flow or sector_fit or composite" --basetemp "$env:TEMP\pytest-composite-review-signals"` -> `3 passed, 17 deselected`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "overview or decision" --basetemp "$env:TEMP\pytest-composite-review-tabs"` -> `5 passed, 32 deselected`
  `git diff --check -- src/ui/panels.py tests/test_ui_components.py tasks/todo.md` -> passed with line-ending warnings only
  Architect verification retry -> APPROVED
# 2026-05-15 - KOFIA FreeSIS Dashboard Research

## Goal
- 금투협 FreeSIS 통계 중 섹터 로테이션 대시보드에 반영할 가치가 높은 지표를 선별한다.
- 현재 대시보드의 가격/수급/매크로 구조와 충돌하지 않는 적용 위치를 정한다.
- 구현 전 리스크와 collector spike 범위를 분리한다.

## Checklist
- [x] FreeSIS 메인 메뉴와 serviceId 후보 추출
- [x] FreeSIS metadata API 호출 방식 확인
- [x] 현재 dashboard/data source 구조 확인
- [x] 대시보드 반영 우선순위와 산출물 작성
- [x] autoresearch completion artifact 작성

## Review
- Output:
  `.omx/specs/autoresearch-kofia-freesis/report.md`
- Result:
  FreeSIS는 섹터별 가격/수급이 아니라 KR 시장 유동성, 레버리지, 금리, 펀드 플로우 컨텍스트 보강용으로 반영하는 것이 적합하다.
  1차 P0 후보는 `증시자금추이`, `신용공여 잔고 추이`, `최종호가수익률`이다.
  P1 후보는 `일자별 CMA현황`, `기간자금유출입`이다.
  초기 구현에서는 sector score 산식에 반영하지 않고 overview/context/monitoring에만 표시한다.
- Verification:
  FreeSIS 메인 HTML에서 serviceId 61개 추출.
  `/meta/getSrvData.do` JSON POST 성공.
  주요 후보 8개 서비스의 `dsGridSQL`, `dsSearch`, `dsListAppDt` 확인.
  현재 프로젝트의 KR 수급/매크로/warehouse 구조와 배치 경계 확인.

# 2026-05-16 - US Trade Sector Lens Autopilot

## Goal
- 미국 총량 수출입 데이터를 섹터별 직접 지표로 오해하지 않게 한다.
- US overview에 총량 수출입과 섹터별 교역 노출도를 결합한 보조 lens를 표시한다.
- 섹터 점수/액션 산식은 변경하지 않는다.

## Checklist
- [x] Autopilot context snapshot 생성
- [x] PRD/test-spec 작성
- [x] US sector map에 `trade_exposure`/`trade_proxy_label` 추가
- [x] 총량 trade proxy lens helper 구현
- [x] US overview에 `미국 수출입 섹터 렌즈` 패널 추가
- [x] 직접 섹터 데이터가 아닌 `총량 proxy` guardrail copy 추가
- [x] macro/UI targeted tests 통과

## Review
- Artifacts:
  `.omx/context/us-trade-sector-lens-dashboard-20260516T054000Z.md`
  `.omx/plans/prd-us-trade-sector-lens-dashboard.md`
  `.omx/plans/test-spec-us-trade-sector-lens-dashboard.md`
- Verification:
  `python -m py_compile app.py src\macro\series_utils.py src\dashboard\types.py src\ui\panels.py tests\test_macro_series_utils.py tests\test_ui_components.py` -> passed
  `python -m pytest -q tests/test_macro_series_utils.py tests/test_ui_components.py` -> `92 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py` -> `37 passed`
# 2026-05-16 - Dashboard Natural Korean UI Pass

## Goal
- 대시보드에서 AI 생성물처럼 보이는 과한 pill, 굵은 한글, 과장된 hover/색 대비를 줄인다.
- 한글 UI는 Pretendard 중심으로 자연스럽게 보이도록 weight, line-height, letter-spacing을 재조정한다.
- 데이터, 점수 산식, 화면 구조는 유지하고 theme/CSS 체감만 좁게 개선한다.

## Checklist
- [x] 현재 theme/CSS와 기존 디자인 계약 확인
- [x] image-taste-frontend 레퍼런스 생성 및 디자인 방향 추출
- [x] 한글 폰트/토큰/카드/배지 CSS 개선
- [x] theme/contrast/UI focused tests 실행
- [x] 결과 기록

## Reference Extraction
- Direction:
  조용한 프리미엄 금융 운영 UI.
  밝은 off-white 배경, 얇은 border, 낮은 그림자, 6-8px radius, 자연스러운 한글 weight를 기준으로 한다.
- Avoid:
  과한 uppercase, 큰 pill radius, 과도하게 굵은 카드 제목/숫자, hover lift, AI식 푸른 glow/gradient.
- Typography:
  한글 본문/라벨은 Pretendard Local 우선.
  한글은 자간 0, display도 과한 음수 자간을 피한다.
  제목은 650-700, 본문은 450-500, caption/badge는 620-650 중심으로 낮춘다.

## Review
- Changed files:
  `.streamlit/config.toml`
  `config/theme.py`
  `src/ui/css.py`
  `src/ui/panels.py`
  `tests/test_ui_theme.py`
  `tests/test_ui_components.py`
  `tasks/todo.md`
- Result:
  한글 UI font stack에 `SUIT`, `Spoqa Han Sans Neo` fallback을 추가했다.
  light palette를 조금 더 조용한 finance tone으로 낮추고 muted contrast는 WCAG AA를 유지하도록 보정했다.
  heading/body/badge/button weight를 낮추고, heading 음수 자간, uppercase 라벨, 큰 pill radius, hover lift를 줄였다.
  상단 `market context`와 `WARNING` 배지는 각각 `시장 컨텍스트`, `주의`로 바꿨다.
- Verification:
  `python -m py_compile src\ui\panels.py tests\test_ui_components.py config\theme.py src\ui\css.py` -> passed
  `python -m pytest -q tests/test_ui_components.py --basetemp "$env:TEMP\pytest-natural-ko-components-2"` -> `87 passed`
  `python -m pytest -q tests/test_ui_theme.py tests/test_ui_contrast.py --basetemp "$env:TEMP\pytest-natural-ko-theme-contrast-3"` -> `23 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py tests/test_ui_copy.py tests/test_dashboard_runtime.py --basetemp "$env:TEMP\pytest-natural-ko-tabs-copy-runtime-2"` -> `74 passed`
  `python scripts\capture_streamlit_screenshot.py --output .omx\artifacts\dashboard-natural-ko-desktop-post.png --port 8513 --debug-port 9233 --url http://127.0.0.1:8513 --width 1440 --height 1024 --timeout 120 --min-text-len 300` -> passed
  `git diff --check -- .streamlit/config.toml config/theme.py src/ui/css.py src/ui/panels.py tests/test_ui_theme.py tests/test_ui_components.py tasks/todo.md` -> passed with line-ending warnings only
# 2026-05-16 - KOSPI Overview Daily Return Label Check

## Goal
- KOSPI overview card percentage가 실제 일간 수익률인지 확인한다.
- 표시값이 잘못 계산되거나 기간 기준이 불명확하면 좁게 수정한다.
- 기존 sector/macro 산식은 변경하지 않는다.

## Checklist
- [x] KOSPI 카드 계산 경로와 최신 캐시 데이터 대조
- [x] 일간 수익률 기준을 고정하는 regression test 추가
- [x] 필요한 UI 표시/계산 수정
- [x] targeted tests 실행
- [x] 결과 기록

## Review
- Finding:
  Overview market card percent is already daily return: latest close / previous observed close - 1.
  Local cache KOSPI latest two closes are 2026-05-11 `7822.24` and 2026-05-12 `7643.15`, which equals `-2.29%`.
- Changed:
  Market card percent now renders as `1D ±x.xx%`.
  Date-indexed series are sorted before selecting latest/previous values.
- Verification:
  `python -m py_compile src\ui\panels.py tests\test_ui_components.py` -> passed
  `python -m pytest -q tests/test_ui_components.py -k "overview_market_cards" --basetemp "$env:TEMP\pytest-kospi-overview-daily"` -> `4 passed, 85 deselected`
  `git diff --check -- src\ui\panels.py tests\test_ui_components.py tasks\todo.md` -> passed with line-ending warnings only
# 2026-05-16 - Overview Sector Table Export YoY De-duplication

## Goal
- 섹터 모멘텀 표에서 `수출 YoY` 숫자 컬럼을 제거한다.
- 같은 페이지의 `섹터별 수출 YoY 월별 추이` 차트는 유지한다.
- 수출 기준 라벨은 차트 해석 보조 정보로 유지하되, 표 정렬 기준에서는 수출 YoY를 제거한다.

## Checklist
- [x] overview sector table 데이터/렌더링 경로 확인
- [x] 표의 `수출 YoY` 컬럼과 정렬 옵션 제거
- [x] 기존 수출 추이 차트 회귀 방지 테스트 조정
- [x] targeted tests 실행
- [x] 결과 기록

## Review
- Changed:
  `src/ui/panels.py`에서 overview 섹터 표의 `수출 YoY` 값 컬럼과 `수출 YoY` 정렬 옵션을 제거했다.
  수출 기준 보조 라벨과 `섹터별 수출 YoY 월별 추이` 차트는 유지했다.
- Tests:
  `python -m py_compile src\ui\panels.py tests\test_ui_components.py` -> passed
  `python -m pytest -q tests/test_ui_components.py -k "overview_sector_frame or render_overview_sector_table or sector_export_trend" --basetemp "$env:TEMP\pytest-export-yoy-dedupe"` -> `5 passed, 87 deselected`
# 2026-05-16 - AI Slop Cleaner Pass

## Goal
- 프로젝트에 남은 AI slop을 좁은 범위로 제거한다.
- 기존 user/worktree 변경은 되돌리지 않는다.
- 동작은 focused regression으로 먼저 잠그고, 작은 중복/침묵 fallback 정리만 수행한다.

## Checklist
- [x] 기존 lessons, dirty worktree, slop 후보 확인
- [x] focused baseline test로 behavior lock
- [x] cleanup plan 작성
- [x] overview/sector momentum 카드 렌더링 중복 제거
- [x] cache read fallback 침묵성 줄이기
- [x] focused verification 및 diff check

## Cleanup Plan
- Scope:
  `src/ui/panels.py`, `src/data_sources/krx_stock_screening.py`, 관련 focused tests.
- Fallback findings:
  cache read 실패가 `None`으로 조용히 축소되는 경로가 있다. cache는 선택적 boundary라 동작은 보존하되 debug log를 남기는 grounded fail-safe fallback으로 정리한다.
- Smells:
  overview 후보 카드와 sector momentum board 카드가 같은 HTML 조립을 반복한다.
  cache read 예외가 evidence 없이 사라진다.
- Order:
  1. 카드 HTML 공통 헬퍼로 duplication 제거.
  2. cache read 예외에 debug evidence 추가.
  3. focused tests, compile, diff check 실행.

## Review
- Changed:
  `src/ui/panels.py`에서 overview 검토 후보 카드와 섹터 모멘텀 보드 카드 HTML 조립을 `_render_review_candidate_card()`로 통합했다.
  `src/data_sources/krx_stock_screening.py`에서 screening/ETF context cache read 예외를 완전 침묵시키지 않고 debug log로 evidence를 남기게 했다.
- Fallback classification:
  cache read 실패 후 `None` 반환은 선택적 cache boundary의 grounded fail-safe fallback으로 유지했다.
  masking fallback slop은 발견하지 않았다.
- Verification:
  `python -m py_compile src\ui\panels.py src\data_sources\krx_stock_screening.py tests\test_ui_components.py tests\test_krx_stock_screening.py` -> passed
  `python -m pytest -q tests/test_ui_components.py -k "overview_review_candidate or sector_momentum_decision_board" --basetemp "$env:TEMP\pytest-ai-slop-ui"` -> `9 passed, 83 deselected`
  `python -m pytest -q tests/test_krx_stock_screening.py -k "representative_etf_context or cache_only or live_failure" --basetemp "$env:TEMP\pytest-ai-slop-screening"` -> `6 passed, 10 deselected`
  `python -m pytest -q tests/test_ui_components.py tests/test_krx_stock_screening.py tests/test_sector_etf_mapping.py --basetemp "$env:TEMP\pytest-ai-slop-final"` -> `112 passed`
  `git diff --check -- src/ui/panels.py src/data_sources/krx_stock_screening.py tasks/todo.md` -> passed with LF/CRLF warnings only
# Overview 매수/매도 검토 후보 동시 표시

## 목표

- overview 대시보드의 검토 후보 영역에서 매수 검토 후보와 매도 검토 후보를 함께 볼 수 있게 한다.

## 계획

- [x] 현재 overview 후보 산출/렌더링 경로 확인
- [x] Autopilot context/PRD/test-spec/ralplan 산출물 작성
- [x] 기존 composite projection을 보존한 매수/매도 그룹 계층 추가
- [x] overview 호출부를 그룹 렌더링으로 전환
- [x] 그룹 산출 및 렌더링 회귀 테스트 추가
- [x] py_compile 및 targeted pytest 검증

## 리뷰

- 변경 파일:
  - `src/ui/panels.py`
  - `tests/test_ui_components.py`
  - `.omx/context/overview-buy-sell-review-candidates-20260516T225051Z.md`
  - `.omx/plans/prd-overview-buy-sell-review-candidates-20260516.md`
  - `.omx/plans/test-spec-overview-buy-sell-review-candidates-20260516.md`
  - `.omx/plans/ralplan-overview-buy-sell-review-candidates-20260516.md`
- 검증:
  - `python -m py_compile src\ui\panels.py tests\test_ui_components.py` -> passed
  - `python -m pytest -q tests/test_ui_components.py -k "overview_review_candidate" --basetemp "$env:TEMP\pytest-overview-buy-sell-candidates"` -> `8 passed, 87 deselected`
  - `python -m pytest -q tests/test_dashboard_tabs.py -k "overview or decision" --basetemp "$env:TEMP\pytest-overview-buy-sell-tabs"` -> `6 passed, 33 deselected`
  - WATCH 보완 후 `python -m pytest -q tests/test_ui_components.py -k "overview_review_candidate" --basetemp "$env:TEMP\pytest-overview-buy-sell-candidates-2"` -> `9 passed, 87 deselected`
  - WATCH 보완 후 `python -m pytest -q tests/test_dashboard_tabs.py -k "overview or decision" --basetemp "$env:TEMP\pytest-overview-buy-sell-tabs-2"` -> `6 passed, 33 deselected`
  - Follow-up code review -> `APPROVE`
  - Follow-up architecture review -> `CLEAR`
- 남은 리스크:
  - 전체 테스트는 실행하지 않았다. 현재 worktree에 기존 변경이 많아 이번 변경 범위는 targeted 검증으로 제한했다.
  - `omx state write`는 기존 `ralph/autoresearch/ralplan` 활성 상태와 충돌해 Autopilot 상태 파일 갱신이 차단됐다.
# 2026-05-16 - Autoresearch Thematic Sector Mapping

## Goal
- 전력, 조선, 원자력, 로봇, 방산, 우주항공, 화장품 같은 테마형 섹터 구분이 외부에서 이미 쓰이는지 확인한다.
- 각 구분에 대해 주가정보를 가져올 수 있는 현실적인 데이터 경로를 판정한다.

## Checklist
- [x] 외부 분류/테마 지수 체계 조사
- [x] 국내 시장에서 해당 테마별 매핑 가능성 판정
- [x] 주가 데이터 수집 가능 경로 판정
- [x] autoresearch 산출물과 completion artifact 작성

## Review
- Research artifact:
  `.omx/specs/autoresearch-thematic-sector-mapping/report.md`
- Conclusion:
  요청한 구분은 하나의 공통 표준 섹터 체계라기보다 `WICS/GICS/FICS 산업분류`와 `FnGuide/iSelect/DeepSearch 테마지수`를 결합해 매핑하는 것이 맞다.
  조선, 방산/우주항공, 화장품은 표준 산업분류 anchor가 강하고, 원자력/로봇/AI전력은 테마지수/ETF proxy 성격이 강하다.
- Price feasibility:
  `pykrx.stock.get_market_ohlcv("20260512", "20260516", code)`로 대표 ETF 14개 코드의 최근 OHLCV 반환을 확인했다.
  KRX index finder는 125개 공식 rows를 반환했지만 요청 테마명 직접 매칭은 없었다.
- Completion artifact:
  `.omx/specs/autoresearch-thematic-sector-mapping/result.json`

# 2026-05-16 - Ralplan Thematic Sector Lens Implementation

## Goal
- autoresearch 결과를 구현 가능한 계획으로 확정한다.
- 기존 macro-regime 섹터 권위를 보존하면서 별도 theme lens를 추가하는 실행 범위를 정한다.
- 정상 렌더 cache-only와 명시 refresh live-fetch 경계를 계획 단계에서 고정한다.

## Checklist
- [x] autoresearch 상태 정리 및 context snapshot 작성
- [x] PRD 작성
- [x] test spec 작성
- [x] ralplan 작성
- [x] Architect ITERATE 반영
- [x] Architect 승인
- [x] Critic 승인

## Review
- Plan artifacts:
  `.omx/context/thematic-sector-lens-20260516T141849Z.md`
  `.omx/plans/prd-thematic-sector-lens-20260516.md`
  `.omx/plans/test-spec-thematic-sector-lens-20260516.md`
  `.omx/plans/ralplan-thematic-sector-lens-20260516.md`
- Decision:
  `config/sector_map.yml`은 canonical macro-regime authority로 유지한다.
  전력/조선/원자력/로봇/방산/우주항공/화장품은 별도 KR theme lens로 추가한다.
  phase 1 가격 데이터는 representative ETF OHLCV proxy를 사용한다.
- Key guardrails:
  normal render는 warehouse `read_stock_ohlcv()` 기반 cache-only.
  live pykrx ETF OHLCV fetch는 panel-local 또는 별도 runtime refresh에서만 수행한다.
  cache invalidation은 `config/theme_lens.yml` artifact + warehouse artifact token 기준으로 명시한다.
- Validation:
  Architect cycle1 -> ITERATE: refresh/cache boundary under-specified.
  Architect cycle2 -> APPROVE.
  Critic -> APPROVE.

# 2026-05-16 - Ralph Thematic Sector Lens Implementation

## Goal
- 전력, 조선, 원자력, 로봇, 방산, 우주항공, 화장품을 별도 KR theme lens로 구현한다.
- 기존 `config/sector_map.yml` macro-regime authority와 canonical sector action을 변경하지 않는다.
- 정상 렌더는 cache-only, 명시적 테마 ETF 갱신만 live pykrx fetch를 수행하도록 분리한다.

## Checklist
- [x] `config/theme_lens.yml` 추가
- [x] cache-only loader와 explicit refresh loader 구현
- [x] dashboard runtime payload와 artifact token 연결
- [x] KR signals 화면에 reference-only theme lens panel 추가
- [x] normal render no-pykrx, explicit refresh/upsert, string ETF code, KR/US UI boundary 테스트 추가
- [x] deslop pass 및 post-deslop 회귀 검증
- [x] Architect 최종 승인

## Review
- Changed scope:
  `config/theme_lens.yml`
  `src/data_sources/theme_lens.py`
  `src/dashboard/data.py`
  `src/dashboard/tabs.py`
  `src/dashboard/types.py`
  `src/ui/panels.py`
  `app.py`
  `tests/test_theme_lens.py`
  `tests/test_dashboard_tabs.py`
  `tests/test_ui_components.py`
- Validation:
  `python -m py_compile ...` -> passed
  `python -m pytest -q tests/test_theme_lens.py tests/test_dashboard_tabs.py tests/test_ui_components.py tests/test_dashboard_data.py tests/test_dashboard_runtime.py tests/test_warehouse_investor_flow.py -k "theme_lens or all_signals_tab or routes_filter_state or dashboard or stock_ohlcv" --basetemp "$env:TEMP\pytest-theme-lens-post-deslop"` -> `91 passed, 101 deselected`
  `git diff --check -- <Ralph-owned files>` -> no whitespace errors; CRLF warnings only
  Architect final verification -> APPROVED
- Residual risk:
  `config/sector_map.yml` has unrelated pre-existing dirty changes and was not treated as Ralph-owned scope.
  Full repository pytest was not run because the worktree contains broad unrelated changes; verification was scoped to affected theme/dashboard/UI/warehouse boundaries.

# 2026-05-17 - Ralplan Theme Lens Sector Group Taxonomy

## Goal
- `테마렌즈`를 기존 sector_map replacement가 아니라 별도 KR theme taxonomy authority layer로 계획한다.
- 1차 범위는 taxonomy config, loader, validation tests로 제한한다.
- 운용사/ETF 상품과 benchmark/comparison theme index를 우선 source authority로 둔다.

## Checklist
- [x] deep-interview spec 확인
- [x] 현행 `theme_lens.yml`, loader, UI/cache boundary 확인
- [x] PRD 작성
- [x] test spec 작성
- [x] ralplan 작성
- [x] Architect ITERATE 반영
- [x] Architect 승인
- [x] Critic 승인

## Review
- Plan artifacts:
  `.omx/specs/deep-interview-theme-lens-sector-group-taxonomy.md`
  `.omx/plans/prd-theme-lens-sector-group-taxonomy-20260517.md`
  `.omx/plans/test-spec-theme-lens-sector-group-taxonomy-20260517.md`
  `.omx/plans/ralplan-theme-lens-sector-group-taxonomy-20260517.md`
- Decision:
  `config/sector_map.yml`은 교체하지 않는다.
  새 taxonomy는 `config/theme_taxonomy.yml` + `src/data_sources/theme_taxonomy.py`로 분리한다.
  기존 7개 theme_id는 adapter continuity를 위해 그대로 보존한다.
  `authority_priority`, `primary_authority.basis_type`, conflict-priority, secondary anchor non-authority를 테스트로 고정한다.
- Validation:
  Architect cycle1 -> ITERATE: ID continuity / authority priority / basis enum / conflict tests 필요.
  Architect cycle2 -> APPROVE.
  Critic -> APPROVE.
  `git diff --check -- .omx/plans/... .omx/specs/... .omx/interviews/... .omx/context/...` -> passed.

# 2026-05-17 - Ralph Theme Taxonomy Implementation

## Goal
- 승인된 `ralplan-theme-lens-sector-group-taxonomy-20260517.md`를 구현한다.
- `config/sector_map.yml`은 건드리지 않고 별도 KR theme taxonomy authority layer를 추가한다.
- 기존 theme lens 가격 proxy/refresh 경계는 유지한다.

## Checklist
- [x] Ralph context snapshot 작성
- [x] `config/theme_taxonomy.yml` 추가
- [x] `src/data_sources/theme_taxonomy.py` static loader/validator 추가
- [x] `tests/test_theme_taxonomy.py` 추가
- [x] taxonomy focused tests 실행
- [x] 기존 theme lens 회귀 테스트 실행
- [x] Architect verification 승인
- [x] Ralph deslop pass 수행
- [x] post-deslop 회귀 테스트 실행

## Review
- Changed scope:
  `config/theme_taxonomy.yml`
  `src/data_sources/theme_taxonomy.py`
  `tests/test_theme_taxonomy.py`
  `.omx/context/theme-lens-sector-group-taxonomy-implementation-20260517T052046Z.md`
- Decision:
  새 taxonomy는 `sector_map.yml` 및 기존 `theme_lens.py`와 분리했다.
  기존 7개 theme_id를 유지하고, 추가 ETF-product-backed theme로 `semiconductors`를 추가했다.
  `authority_priority`, `basis_type`, source evidence, product code string, secondary anchor non-override를 validator/test로 고정했다.
- Validation:
  `python -m py_compile src\data_sources\theme_taxonomy.py tests\test_theme_taxonomy.py` -> passed
  `python -m pytest -q tests/test_theme_taxonomy.py --basetemp "$env:TEMP\pytest-theme-taxonomy"` -> `11 passed`
  `python -m pytest -q tests/test_theme_lens.py --basetemp "$env:TEMP\pytest-theme-lens-regression"` -> `8 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "theme_lens or dashboard_page" --basetemp "$env:TEMP\pytest-theme-taxonomy-tabs"` -> `5 passed, 36 deselected`
  Architect verification -> APPROVE
  Deslop pass -> deterministic product authority basis ordering cleanup only
  post-deslop `py_compile` -> passed
  post-deslop `tests/test_theme_taxonomy.py` -> `11 passed`
  post-deslop `tests/test_theme_lens.py` -> `8 passed`
  post-deslop `tests/test_dashboard_tabs.py -k "theme_lens or dashboard_page"` -> `5 passed, 36 deselected`
  `git diff --check -- <Ralph-owned files>` -> passed with CRLF warning only for `tasks/todo.md`

# 2026-05-17 - Ralplan Theme Taxonomy Official Source Expansion

## Goal
- `theme_taxonomy.yml` 확장을 운용사 공식 URL 중심으로 계획한다.
- K-ETF는 보조 URL로만 쓰고, 단독 최종 권위로 쓰지 않는다.
- `sector_map.yml` 교체 금지와 기존 taxonomy/theme lens 분리를 유지한다.

## Checklist
- [x] 기존 taxonomy config/loader와 autoresearch 결과 확인
- [x] 운용사 공식 URL/K-ETF 보조 URL 근거 정책 확인
- [x] context snapshot 작성
- [x] PRD 작성
- [x] test spec 작성
- [x] ralplan 초안 작성
- [x] Architect 검토 반영
- [x] Critic 승인 반영

## Review
- Plan artifacts:
  `.omx/context/theme-taxonomy-official-source-expansion-20260517T055004Z.md`
  `.omx/plans/prd-theme-taxonomy-official-source-expansion-20260517.md`
  `.omx/plans/test-spec-theme-taxonomy-official-source-expansion-20260517.md`
  `.omx/plans/ralplan-theme-taxonomy-official-source-expansion-20260517.md`
- Architect cycle1 -> ITERATE:
  legacy K-ETF primary URL 예외가 validator 정책과 충돌하므로 accepted entry는 기존/신규 모두 공식 primary URL을 요구하도록 수정했다.
  `config/sector_map.yml` diff guard와 `k-etf.com` primary URL 차단 테스트를 추가했다.
- Architect cycle2 -> ITERATE:
  PRD risk section의 "기존 항목은 opportunistic" 표현이 strict migration과 충돌해 제거했다.
- Architect cycle3 -> ITERATE:
  기존 accepted theme_id 안정성과 "제외 가능" 문구가 충돌해, 기존 ID는 반드시 공식 근거로 backfill하거나 blocker 보고로 중단하도록 고정했다.
- Architect cycle4 -> APPROVE:
  공식 primary URL 정책, 기존 ID blocker rule, `sector_map.yml` diff guard, K-ETF supporting-only 테스트가 승인됐다.
- Critic -> APPROVE:
  Option A 방향, acceptance criteria, verification, 기존 seed blocker rule 모두 승인됐다.
  비차단 표현 정리로 PRD의 "Accepted source roles"를 "Allowed source roles"로 바꿨다.

# 2026-05-17 - Ralph Theme Taxonomy Official Source Expansion

## Goal
- 승인된 ralplan대로 `theme_taxonomy.yml`을 공식-primary URL 기반 schema로 확장한다.
- K-ETF는 보조 URL로만 유지한다.
- 기존 `sector_map.yml`과 `theme_lens.yml` 경계는 변경하지 않는다.

## Checklist
- [x] Ralph context/plan/test spec 확인
- [x] loader에 `source_role`/`supporting_urls` dataclass 및 validation 추가
- [x] accepted primary `k-etf.com` URL 차단 추가
- [x] 기존 seed theme ID 공식-primary URL로 migration
- [x] first expansion batch 12개 theme 추가
- [x] taxonomy unit tests 확장
- [x] focused compile/test 실행
- [x] theme lens/dashboard focused regression 실행
- [ ] architect verification
- [x] deslop pass
- [x] post-deslop regression

## Review
- Changed scope:
  `config/theme_taxonomy.yml`
  `src/data_sources/theme_taxonomy.py`
  `tests/test_theme_taxonomy.py`
  `.omx/context/theme-taxonomy-official-source-evidence-20260517.md`
- Implementation:
  accepted entry는 `source_role`과 non-K-ETF primary `source_url`을 요구한다.
  K-ETF URL은 `supporting_urls`의 `AGGREGATOR_REFERENCE`로만 허용한다.
  taxonomy는 20개 theme ID를 로드한다.
- Verification so far:
  `python -m py_compile src\data_sources\theme_taxonomy.py tests\test_theme_taxonomy.py` -> passed
  `python -m pytest -q tests/test_theme_taxonomy.py --basetemp "$env:TEMP\pytest-theme-taxonomy-official-source-initial"` -> `16 passed`
  loader smoke check -> 20 theme IDs loaded, `primary_ketf []`
  `python -m pytest -q tests/test_theme_lens.py --basetemp "$env:TEMP\pytest-theme-lens-source-regression"` -> `8 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "theme_lens or dashboard_page" --basetemp "$env:TEMP\pytest-theme-taxonomy-source-tabs"` -> `5 passed, 36 deselected`
  `git diff --exit-code -- config\sector_map.yml` -> passed
  `git diff --check -- <Ralph-owned files>` -> passed with `tasks/todo.md` CRLF warning only
- Architect verification -> APPROVE
- Deslop pass:
  fallback-like 검색 결과 없음.
  미사용 `PRIMARY_SOURCE_ROLES` 상수만 제거했다.
- Post-deslop verification:
  `python -m py_compile src\data_sources\theme_taxonomy.py tests\test_theme_taxonomy.py` -> passed
  `python -m pytest -q tests/test_theme_taxonomy.py --basetemp "$env:TEMP\pytest-theme-taxonomy-official-source-post-deslop"` -> `16 passed`
  `python -m pytest -q tests/test_theme_lens.py --basetemp "$env:TEMP\pytest-theme-lens-source-post-deslop"` -> `8 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "theme_lens or dashboard_page" --basetemp "$env:TEMP\pytest-theme-taxonomy-source-tabs-post-deslop"` -> `5 passed, 36 deselected`
  final loader smoke -> 20 theme IDs loaded, accepted primary K-ETF URLs `[]`
  final `git diff --exit-code -- config\sector_map.yml` -> passed
  final `git diff --check -- <Ralph-owned files>` -> passed with `tasks/todo.md` CRLF warning only

# 2026-05-17 - Autopilot Theme Taxonomy Dual Axis Improvement

## Goal
- `한국 주식시장 테마 분류의 적절성 평가와 개선안_260517.md` 권고를 반영한다.
- 기존 ETF/product-backed `themes`는 유지하고, master taxonomy용 기본산업 + 교차테마 축을 추가한다.
- `sector_map.yml`은 변경하지 않는다.

## Checklist
- [x] source evaluation 문서 읽기
- [x] Autopilot context snapshot 작성
- [x] ralplan PRD/test-spec/plan 작성
- [x] Architect ITERATE 반영: `K-뷰티` cross-theme와 `axis_id/tag_id` 주소 규칙 추가
- [x] Critic ITERATE 반영: alias/history 검증 추가
- [x] loader dataclass/parser 확장
- [x] `classification_axes` config 추가
- [x] taxonomy tests 확장
- [x] focused verification
- [x] Ralph architect verification
- [x] Ralph deslop + post-regression
- [x] code-review clean gate

## Review
- Plan artifacts:
  `.omx/context/theme-taxonomy-dual-axis-improvement-20260517T074418Z.md`
  `.omx/plans/prd-theme-taxonomy-dual-axis-improvement-20260517.md`
  `.omx/plans/test-spec-theme-taxonomy-dual-axis-improvement-20260517.md`
  `.omx/plans/ralplan-theme-taxonomy-dual-axis-improvement-20260517.md`
- Implementation evidence:
  `.omx/context/theme-taxonomy-dual-axis-implementation-evidence-20260517.md`
- Verification so far:
  `python -m py_compile src\data_sources\theme_taxonomy.py tests\test_theme_taxonomy.py` -> passed
  `python -m pytest -q tests/test_theme_taxonomy.py --basetemp "$env:TEMP\pytest-theme-taxonomy-dual-axis-initial"` -> `23 passed`
  `python -m pytest -q tests/test_theme_lens.py --basetemp "$env:TEMP\pytest-theme-lens-dual-axis"` -> `8 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "theme_lens or dashboard_page" --basetemp "$env:TEMP\pytest-theme-taxonomy-dual-axis-tabs"` -> `5 passed, 36 deselected`
  loader smoke -> 20 themes, 7 base axes, 5 cross axes, `k_beauty` aliases preserved
  `git diff --exit-code -- config\sector_map.yml` -> passed
- Ralph architect verification -> APPROVE
- Deslop:
  fallback-like search -> no findings.
  behavior lock `tests/test_theme_taxonomy.py` -> `23 passed`.
  no cleanup edits needed.
- Post-deslop verification:
  `python -m py_compile src\data_sources\theme_taxonomy.py tests\test_theme_taxonomy.py` -> passed
  `python -m pytest -q tests/test_theme_taxonomy.py --basetemp "$env:TEMP\pytest-theme-taxonomy-dual-axis-post-deslop"` -> `23 passed`
  `python -m pytest -q tests/test_theme_lens.py --basetemp "$env:TEMP\pytest-theme-lens-dual-axis-post-deslop"` -> `8 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "theme_lens or dashboard_page" --basetemp "$env:TEMP\pytest-theme-taxonomy-dual-axis-tabs-post-deslop"` -> `5 passed, 36 deselected`
  final `git diff --exit-code -- config\sector_map.yml` -> passed
  final `git diff --check -- <Autopilot-owned files>` -> passed with `tasks/todo.md` CRLF warning only
- Code-review -> APPROVE / CLEAR:
  findings none.
  reviewer evidence included 0 scoped diagnostics, compile pass, taxonomy/theme_lens/dashboard focused tests pass, loader smoke pass, `sector_map.yml` guard pass, scoped diff check pass.

# 2026-05-17 - Autopilot Theme Taxonomy Warehouse Layer

## Goal
- `theme_taxonomy` 기준 실행 매핑을 수집 파이프라인에 포함한다.
- `theme_taxonomy`를 기존 KRX 수집 레이어와 같은 warehouse 상태/이력 화면에서 확인한다.
- `sector_map.yml`은 변경하지 않는다.

## Checklist
- [x] 기존 taxonomy/dashboard/warehouse 접점 확인
- [x] `theme_taxonomy` warehouse 수집 상태 기록 구현
- [x] sync/bootstrap 수집 코드 집합에 taxonomy 실행 매핑 반영
- [x] 품질 대시보드 데이터셋 표시에 `theme_taxonomy` 추가
- [x] 회귀 테스트와 smoke 검증
- [x] Architect ITERATE 반영: mapping coverage completeness 검증
- [x] Code-review REQUEST CHANGES 반영: CLI success gate와 monitoring reasons 보강
- [x] 최종 code-review clean gate

## Review
- Implementation:
  `src/data_sources/theme_taxonomy_sync.py`를 추가해 `theme_taxonomy` 실행 매핑을 warehouse ingest dataset으로 기록한다.
  `theme_taxonomy`는 `market_prices`, `macro_data`, `investor_flow`와 같은 `COLLECTION_HISTORY_DATASETS` 경로에 포함된다.
  `scripts/sync_warehouse.py`와 `scripts/bootstrap_warehouse.py`는 taxonomy 실행 인덱스 코드를 KRX 수집 코드 집합에 병합한다.
  품질 대시보드는 `테마분류`를 같은 상태/이력 테이블에서 표시한다.
- Verification so far:
  `python -m py_compile src\data_sources\theme_taxonomy_sync.py src\data_sources\warehouse.py src\dashboard\tabs.py scripts\sync_warehouse.py scripts\bootstrap_warehouse.py tests\test_theme_taxonomy.py tests\test_theme_taxonomy_sync.py tests\test_dashboard_tabs.py tests\test_warehouse_cli.py` -> passed
  `python -m pytest -q tests/test_theme_taxonomy.py tests/test_theme_taxonomy_adapter.py tests/test_theme_taxonomy_sync.py tests/test_warehouse_cli.py tests/test_dashboard_tabs.py -k "theme_taxonomy or collection or monitoring or warehouse_cli or cached_monitoring" --basetemp "$env:TEMP\pytest-theme-taxonomy-warehouse"` -> `71 passed, 32 deselected`
  `python -m pytest -q tests/test_theme_taxonomy.py tests/test_theme_taxonomy_adapter.py tests/test_theme_taxonomy_sync.py tests/test_warehouse_cli.py tests/test_dashboard_tabs.py --basetemp "$env:TEMP\pytest-theme-taxonomy-warehouse-full"` -> `103 passed`
  `python -m pytest -q tests/test_dashboard_state.py tests/test_theme_lens.py tests/test_dashboard_runtime.py --basetemp "$env:TEMP\pytest-theme-taxonomy-runtime"` -> `49 passed`
  Architect initial review -> ITERATE: 부분 runtime mapping 누락 시 녹색 처리 가능.
  coverage fix 후 `python -m pytest -q tests/test_theme_taxonomy_sync.py --basetemp "$env:TEMP\pytest-theme-taxonomy-sync-coverage"` -> `3 passed`
  coverage fix 후 `python -m pytest -q tests/test_theme_taxonomy.py tests/test_theme_taxonomy_adapter.py tests/test_theme_taxonomy_sync.py tests/test_warehouse_cli.py tests/test_dashboard_tabs.py --basetemp "$env:TEMP\pytest-theme-taxonomy-warehouse-full"` -> `104 passed`
  coverage fix 후 `python -m pytest -q tests/test_dashboard_state.py tests/test_theme_lens.py tests/test_dashboard_runtime.py tests/test_dashboard_data.py tests/test_krx_indices.py tests/test_warehouse_multimarket.py --basetemp "$env:TEMP\pytest-theme-taxonomy-runtime-extra"` -> `70 passed`
  Architect re-review -> APPROVE / CLEAR
  Code-review initial -> REQUEST CHANGES: CLI success gate must require taxonomy coverage; quality history must include sync/bootstrap reasons.
  review fix smoke -> `python -m pytest -q tests/test_warehouse_cli.py::test_bootstrap_warehouse_cli_fails_when_theme_taxonomy_incomplete tests/test_warehouse_cli.py::test_sync_warehouse_cli_fails_when_theme_taxonomy_incomplete tests/test_dashboard_tabs.py::test_cached_monitoring_data_reads_manual_refresh_history --basetemp "$env:TEMP\pytest-theme-taxonomy-review-fixes"` -> `3 passed`
  review fix full targeted -> `python -m pytest -q tests/test_theme_taxonomy.py tests/test_theme_taxonomy_adapter.py tests/test_theme_taxonomy_sync.py tests/test_warehouse_cli.py tests/test_dashboard_tabs.py --basetemp "$env:TEMP\pytest-theme-taxonomy-warehouse-full"` -> `106 passed`
  review fix extra regression -> `python -m pytest -q tests/test_dashboard_state.py tests/test_theme_lens.py tests/test_dashboard_runtime.py tests/test_dashboard_data.py tests/test_krx_indices.py tests/test_warehouse_multimarket.py --basetemp "$env:TEMP\pytest-theme-taxonomy-runtime-extra"` -> `70 passed`
  `git diff --exit-code -- config\sector_map.yml` -> passed
  `git diff --check -- <Autopilot-owned files>` -> passed with CRLF warnings only
  `rg -n "[ \t]$" <Autopilot-owned code/test files>` -> no trailing whitespace
  Final code-review -> APPROVE / CLEAR

# 2026-05-17 - Autopilot Theme Taxonomy Feedback Improvements

## Goal
- `피드백_한국 주식시장 테마 분류의 적절성 평가와 개선안_260517.md`의 구현 피드백을 taxonomy 레이어에 반영한다.
- `theme_mappings`, verification freshness, missing base/cross axes, value-up overlay, network rename을 추가한다.
- `sector_map.yml`은 변경하지 않는다.

## Checklist
- [x] Autopilot ralplan artifacts 작성
- [x] Architect ITERATE 반영: all-theme mapping coverage, mapping cardinality, stale-date behavior, preservation tests
- [x] Architect ITERATE 반영: 부동산/리츠 base axis 추가
- [x] Architect ITERATE 반영: `network_infrastructure` aliases 및 value-up semantics 테스트 강화
- [x] loader에 verification/theme_mappings dataclass와 validation 추가
- [x] taxonomy YAML에 verification, 20개 mappings, 확장 axes/tags 추가
- [x] taxonomy tests 확장
- [x] focused regression 실행
- [x] code-review clean gate

## Review
- Plan artifacts:
  `.omx/context/theme-taxonomy-feedback-improvements-20260517T082423Z.md`
  `.omx/plans/prd-theme-taxonomy-feedback-improvements-20260517.md`
  `.omx/plans/test-spec-theme-taxonomy-feedback-improvements-20260517.md`
  `.omx/plans/ralplan-theme-taxonomy-feedback-improvements-20260517.md`
- Verification so far:
  `python -m py_compile src\data_sources\theme_taxonomy.py tests\test_theme_taxonomy.py` -> passed
  loader smoke -> 20 themes, 20 mappings, 11 base axes, 7 cross axes, `verified`
  `python -m pytest -q tests/test_theme_taxonomy.py --basetemp "$env:TEMP\pytest-theme-taxonomy-feedback"` -> `36 passed`
  `python -m pytest -q tests/test_theme_lens.py --basetemp "$env:TEMP\pytest-theme-lens-feedback"` -> `8 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "theme_lens or dashboard_page" --basetemp "$env:TEMP\pytest-theme-taxonomy-feedback-tabs"` -> `5 passed, 36 deselected`
  `git diff --exit-code -- config\sector_map.yml` -> passed
  `git diff --check -- <tracked scoped files>` -> passed with `tasks/*.md` CRLF warnings only
  `rg -n "[ \t]$" <scoped files>` -> no trailing whitespace
  code-review -> `APPROVE / CLEAR`

# 2026-05-17 - Overview Review Cards Theme Taxonomy Projection

## Goal
- 첫 화면의 `매수 검토 후보` / `매도 검토 후보` 카드도 theme_taxonomy 기준 표시명을 사용한다.
- 기존 KRX 섹터명은 내부 신호/런타임 추적용으로 보존한다.

## Checklist
- [x] overview review candidate 렌더링 경로 확인
- [x] taxonomy display label/subtext projection 구현
- [x] 회귀 테스트 실행
- [x] 실행 중인 대시보드 상태 확인

## Review
- Implementation:
  overview 검토 후보 카드, 섹터 표, 3M 히트맵이 `Theme Taxonomy` 선택 시 `theme_labels`를 우선 표시한다.
  기존 KRX 섹터명은 `sector_name`/`원섹터`로 보존하고 화면에는 `런타임:` 보조 문구로 남긴다.
- Verification:
  `python -m py_compile src\ui\panels.py tests\test_ui_components.py` -> passed
  `python -m pytest -q tests/test_ui_components.py --basetemp "$env:TEMP\pytest-theme-taxonomy-ui-components-full"` -> `100 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py tests/test_theme_taxonomy_adapter.py --basetemp "$env:TEMP\pytest-theme-taxonomy-tabs-adapter-full"` -> `45 passed`
  `git diff --check -- src\ui\panels.py tests\test_ui_components.py tasks\todo.md tasks\lessons.md` -> passed with CRLF warnings only
  `Invoke-WebRequest http://localhost:8502` -> `HTTP 200 OK`

# 2026-05-17 - Autopilot Theme Taxonomy Sector Price Collection Diagnostic

## Goal
- `theme_taxonomy`/`theme_lens` 기준 섹터·테마 가격 데이터가 후보 산출까지 연결되는지 점검한다.
- `로봇`, `우주항공/UAM` 같은 테마 proxy가 캐시에 있을 때 매수 검토 후보 산출 대상에서 누락되지 않게 한다.

## Checklist
- [x] 기존 taxonomy, theme lens, KR signal universe 경로 확인
- [x] 진단 증거와 ralplan/test spec 작성
- [x] cache-only theme ETF proxy를 signal 입력으로 변환
- [x] KR signal payload에 theme proxy universe 병합
- [x] overview taxonomy 표시 fallback 보강
- [x] 회귀 테스트와 diff guard 실행
- [x] code-review clean gate

## Review
- Diagnosis:
  active KR 후보 universe는 `dim_index`의 canonical KRX sector rows를 사용한다.
  기존 `theme_taxonomy` warehouse 수집 코드는 11개 legacy mapping code만 다루며, `로봇`/`우주항공`은 broad sector index가 아니라 `theme_lens` 대표 ETF proxy로만 존재한다.
  `445290 KODEX K-로봇액티브`는 live proxy 기준 2026-05-15에 1M 약 `+30.38%`, 3M 약 `+20.52%`였지만, DuckDB write lock 때문에 warehouse 저장이 막혔다.
  현재 정상 렌더는 cache-only theme lens를 참고 패널로만 표시하고 후보 산출에는 넣지 않는다.
- Implementation:
  `src/data_sources/theme_lens.py`에 cache-only `load_theme_proxy_signal_inputs()`를 추가해 대표 ETF OHLCV를 signal-compatible price frame과 universe row로 변환한다.
  `src/dashboard/data.py`는 KR 신호 계산 전에 cached theme proxy rows를 canonical KRX universe 뒤에 병합한다.
  `src/ui/panels.py`는 static taxonomy context가 없어도 `taxonomy_kind=THEME`, `taxonomy_label`을 후보 표시 payload로 사용할 수 있다.
- Verification:
  `python -m py_compile src\data_sources\theme_lens.py src\dashboard\data.py src\ui\panels.py tests\test_theme_lens.py tests\test_dashboard_data.py tests\test_ui_components.py` -> passed
  `python -m pytest -q tests/test_theme_lens.py -k "theme_proxy_signal_inputs" --basetemp "$env:TEMP\pytest-theme-proxy-signal-inputs"` -> `2 passed, 8 deselected`
  `python -m pytest -q tests/test_dashboard_data.py -k "cached_signals_merges_cached_theme_proxy_inputs" --basetemp "$env:TEMP\pytest-theme-proxy-dashboard-data"` -> `1 passed, 13 deselected`
  `python -m pytest -q tests/test_ui_components.py -k "theme_signal_metadata" --basetemp "$env:TEMP\pytest-theme-proxy-ui"` -> `1 passed, 104 deselected`
  `python -m pytest -q tests/test_signal_pipeline_integration.py -k "theme_proxy_universe_row" --basetemp "$env:TEMP\pytest-theme-proxy-signal-pipeline"` -> `1 passed, 13 deselected`
  `python -m pytest -q tests/test_theme_lens.py tests/test_dashboard_data.py tests/test_dashboard_runtime.py tests/test_signal_pipeline_integration.py tests/test_ui_components.py --basetemp "$env:TEMP\pytest-theme-proxy-focused-full"` -> `172 passed`
  actual local cache check -> `theme_proxy_signal_status UNAVAILABLE`, because representative ETF OHLCV cache is still missing.
  active KR canonical universe check -> 17 sector rows, no theme proxy rows while ETF cache is unavailable.
  `git diff --exit-code -- config\sector_map.yml` -> passed
  `git diff --check -- <scoped files>` -> passed with CRLF warnings only
  `rg -n "[ \t]$" <scoped code/test files>` -> no trailing whitespace
  code-review -> APPROVE / CLEAR
- Post-lock-clear processing:
  Streamlit `app.py` 프로세스 `27120`, `13392`, `23144` 종료 후 writer lock 해소 확인.
  `sync_theme_taxonomy_warehouse(reason="manual_retry_after_lock_clear")` -> `LIVE`, 11 rows, coverage complete.
  `refresh_theme_lens_etf_ohlcv(asof_date="20260515", lookback_days=420)` -> `LIVE`, fetched 15, refreshed 15, failed 0, rows 3452.
  cache-only readback -> `theme_lens_cache_status CACHED`, 7 theme rows.
  signal input readback -> `proxy_signal_status CACHED`, 1967 price rows, 7 proxy universe rows.
  `_cached_signals(..., "20260515")` smoke -> 24 signals total, theme signals included.
  theme Strong Buy after persistence: `전력/AI전력인프라`, `원자력`, `우주항공/UAM`.
  `로봇` persisted and included, but current hybrid momentum action is `Hold` because 6M/12M excluding recent 21 trading days has negative relative raw momentum despite strong recent 1M/3M returns.

# 2026-05-17 - Recent Momentum Inclusion for Tactical Theme Rotation

## Goal
- KR tactical sector/theme candidates should react to recent theme momentum by default.
- Keep ex-recent momentum as an explicit experiment/risk filter, not the production default.

## Checklist
- [x] `momentum_skip_recent_days` default changed from 21 to 0
- [x] zero-skip momentum window calculation fixed and tested
- [x] dashboard copy updated from ex-1M wording to recent-included wording
- [x] current momentum-method comparison artifacts regenerated
- [x] real cached KR signal smoke confirms theme proxies are included
- [x] focused and full regression tests passed

## Review
- Rationale:
  `12-1` style momentum can reduce short-term reversal risk, but it delays recognition of fast KR theme rotations.
  For the dashboard's buy-review candidate surface, the current price should be part of the default 6M/12M relative momentum calculation.
- Verification:
  `python -m py_compile src\indicators\momentum.py src\signals\matrix.py src\dashboard\data.py src\dashboard\tabs.py tests\test_momentum.py tests\test_signal_pipeline_integration.py` -> passed
  `python -m pytest -q tests/test_momentum.py tests/test_signal_pipeline_integration.py --basetemp "$env:TEMP\pytest-recent-momentum-core-2"` -> `23 passed`
  cached KR smoke as of `20260515` -> theme signals include `로봇` as `Strong Buy`; `우주항공/UAM` remains included but is currently `Hold` under the configured ETF proxy signal.
  `python -m pytest -q tests/test_theme_lens.py tests/test_dashboard_data.py tests/test_dashboard_runtime.py tests/test_ui_components.py --basetemp "$env:TEMP\pytest-recent-momentum-related-2"` -> `158 passed`
  `python -m pytest -q tests/test_dashboard_tabs.py -k "theme_lens or dashboard_page" --basetemp "$env:TEMP\pytest-recent-momentum-tabs"` -> `5 passed, 37 deselected`
  `python -m pytest -q tests/test_integration.py::TestIntegration::test_read_warm_status_returns_sanitized_summary --basetemp "$env:TEMP\pytest-warm-status-row-count"` -> `1 passed`
  `python -m pytest -q --basetemp "$env:TEMP\pytest-recent-momentum-full-2"` -> `717 passed`
  momentum method comparison as of `2026-05-15` -> whipsaws legacy `96`, hybrid `15`, reduction `84.4%`, rank-stability median rho `0.91`, gate `approved`

# 2026-05-18 - Dashboard Theme Visibility and Refresh History Fix

## Goal
- 대시보드 조회 기준일이 warehouse 최신 시장데이터보다 뒤처지지 않게 한다.
- 시장데이터 갱신 버튼을 눌렀을 때 이미 최신인 no-op 경로도 갱신 이력에 남긴다.
- 로봇, 전력, 원자력, 우주항공 같은 theme_lens ETF proxy 신호를 첫 화면에서 바로 보이게 한다.

## Checklist
- [x] 중복 Streamlit 프로세스 정리
- [x] OPENAPI calendar 2026-05-14 / warehouse benchmark 2026-05-15 불일치 재현
- [x] dashboard market end date를 warehouse 최신 벤치마크일로 보정
- [x] no-op market refresh ingest history 기록 추가
- [x] collection history no-op 완료율 100% 표시 보정
- [x] overview 후보 그룹을 action 우선으로 분류
- [x] overview `테마 proxy 모니터` 추가
- [x] focused tests와 실제 warehouse smoke 실행

## Review
- Diagnosis:
  `get_last_business_day(provider=OPENAPI, benchmark_code=1001)`는 2026-05-18 00시대에 `2026-05-14`를 반환했지만, warehouse benchmark `1001`은 이미 `20260515`까지 보유했다.
  첫 화면 후보 카드는 그룹당 3개 제한이라 테마가 신호 원장에 있어도 카드에서 밀렸고, `Strong Buy`가 보조 edge 음수이면 sell group으로 먼저 갈 수 있었다.
  시장데이터가 이미 최신이면 `run_market_refresh()`가 runner를 생략하고 이력을 기록하지 않았다.
- Verification:
  `python -m py_compile src\dashboard\data.py src\dashboard\runtime.py src\dashboard\tabs.py src\ui\panels.py src\data_sources\warehouse.py tests\test_dashboard_data.py tests\test_dashboard_runtime.py tests\test_dashboard_tabs.py tests\test_ui_components.py tests\test_warehouse_cli.py` -> passed
  `python -m pytest -q tests/test_dashboard_data.py -k "resolve_market_end_date or cached_signals_merges_cached_theme_proxy_inputs" --basetemp "$env:TEMP\pytest-dashboard-date-theme-2"` -> `3 passed, 13 deselected`
  `python -m pytest -q tests/test_dashboard_runtime.py -k "run_market_refresh" --basetemp "$env:TEMP\pytest-market-refresh-history-2"` -> `4 passed, 25 deselected`
  `python -m pytest -q tests/test_warehouse_cli.py -k "collection_run_history" --basetemp "$env:TEMP\pytest-collection-history"` -> `7 passed, 14 deselected`
  `python -m pytest -q tests/test_dashboard_tabs.py::test_cached_monitoring_data_reads_manual_refresh_history --basetemp "$env:TEMP\pytest-monitoring-reasons-2"` -> `1 passed`
  `python -m pytest -q tests/test_ui_components.py -k "overview_review_candidate_group or overview_theme_proxy_frame or render_overview_review_candidates" --basetemp "$env:TEMP\pytest-overview-candidate-ui"` -> `8 passed, 98 deselected`
  `python -m pytest -q tests/test_dashboard_data.py tests/test_dashboard_runtime.py tests/test_dashboard_tabs.py tests/test_ui_components.py tests/test_warehouse_cli.py tests/test_theme_lens.py tests/test_signal_pipeline_integration.py --basetemp "$env:TEMP\pytest-dashboard-theme-refresh-related"` -> `238 passed`
  `python -m pytest -q --basetemp "$env:TEMP\pytest-dashboard-theme-refresh-full-2"` -> `721 passed`
  actual local date smoke -> dashboard market end date `2026-05-15`; 18/18 KR market universe codes latest `20260515`.
  actual no-op market refresh smoke -> notice `Market data already current`; latest history row `market_prices manual_refresh 20260515~20260515 CACHED row_count 109088 completion_pct 100.0`.
  actual theme overview smoke -> theme proxy monitor lists `전력/AI전력인프라`, `원자력`, `우주항공/UAM`, `로봇`, `방산`, `조선`, `화장품/K-뷰티`; buy candidates with desktop limit 6 include `전력/AI전력인프라`, `원자력`, `로봇`.
  cleanup -> removed two agent-created zero-row no-op smoke history rows; latest market history row remains row_count `109088`.
  Streamlit restart -> single conda process PID `16648`, `http://localhost:8501` returned HTTP 200.
