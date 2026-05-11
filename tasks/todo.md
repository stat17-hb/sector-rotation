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
