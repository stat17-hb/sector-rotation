# Lessons Learned

## 2026-05-17
- Pattern: 사용자가 섹션 간격이 여전히 타이트하다고 재지적했는데, 이전 수정은 CSS 토큰 확대 위주라 실제 Streamlit DOM에서 붙어 보이는 구간을 충분히 해소하지 못했다.
- Rule: 프론트엔드 spacing 피드백은 토큰 변경만으로 끝내지 말고, 실제 렌더 DOM gap을 측정해 major section gap과 internal gap을 분리해서 검증한다.
- Rule: Streamlit `st.container(border=True)` 안에 여러 작업 흐름이 섞이면 CSS margin만으로 시각적 분리가 약하다. 의미가 다른 흐름은 컨테이너 구조부터 분리한 뒤 spacing token을 적용한다.
- Pattern: 카드형 `상위/하위 변화` heatmap은 이미 원장/차트에서 읽히는 정보를 반복해 화면 밀도만 높였다.
- Rule: 대시보드 보조 시각화는 의사결정에 새 정보를 추가하지 못하면 유지하지 않는다. 제거 요청이 나오면 렌더러, 호출부, CSS, 테스트 계약을 함께 정리한다.

## 2026-05-12
- Pattern: User asked for export data, and the first implementation exposed only aggregate Korean export YoY while the intended decision question required sector-level export trends.
- Rule: When the user asks for a data signal "by sector", verify the output granularity before implementation. Aggregate macro indicators are not sufficient unless the user explicitly accepts aggregate-only scope.
- Rule: For dashboard data additions, state whether the implemented signal is aggregate, mapped-by-sector, or instrument-level, and add a UI/test contract that proves the intended granularity is visible.
- Pattern: 자동차 수출 데이터 was loaded but hidden behind the broader `KOSPI200 경기소비재` sector label and the chart prioritized only currently visible signals.
- Rule: When a sector uses a proxy/export-item series, show the proxy label in the UI and test that non-visible but configured export series still render in supporting charts.

## 2026-04-26
- Pattern: Chrome `--headless --screenshot` captured only the Streamlit skeleton even though `_stcore/health` returned `ok`.
- Rule: For Streamlit visual evidence, do not treat HTTP 200 or health `ok` as proof that the app has hydrated. Wait in the browser for `data-test-connection-state="CONNECTED"`, `data-test-script-state` not in `initial/running`, nontrivial body text, and absence of `[data-testid="stAppSkeleton"]`.
- Pattern: Chrome remote debugging can list extension background pages before the real app page.
- Rule: CDP screenshot tooling must select a target with `type == "page"` and a URL matching the intended app URL; never attach to the first `/json` target blindly.
- Pattern: Streamlit can serve the shell with HTTP 200 for an invalid page path while rendering a `Page not found` modal.
- Rule: Route smoke for Streamlit navigation must inspect rendered DOM/text for `Page not found`, not only HTTP status.
- Pattern: Dashboard showed a calendar/provider target date as the data 기준일 while cached market prices were older than that target.
- Rule: For market-data freshness UI, compute the displayed 기준일 from the actual loaded price frame, preferably the benchmark row's max trade date; show target/query dates separately when they differ.

## 2026-02-22
- Pattern: User asked to reconstruct a high-level evaluation into actionable execution.
- Rule: When feedback asks for "restructure", produce a concrete phased checklist with gates, verification commands, and review template in `tasks/todo.md`.
- Rule: If task-management conventions exist in `AGENTS.md`, persist plan artifacts in `tasks/` instead of only replying in chat.
- Pattern: Source-of-truth plan (`plan.md`) was revised and task checklist drifted.
- Rule: Before implementation, diff `plan.md` vs `tasks/todo.md` on contracts, cache behavior, return types, and test matrix; then sync `todo.md` to the latest plan.
- Pattern: User confirmed IDE view is normal while CLI output looked garbled.
- Rule: Treat mojibake seen in constrained PowerShell output as a display-encoding issue first; do not claim file corruption unless byte-level or editor-level evidence confirms it.

## 2026-02-23
- Pattern: User explicitly requested implementation of an already-agreed plan.
- Rule: When the user says "implement this plan", start coding immediately; do not re-propose alternatives unless a blocker is discovered.
- Rule: For approved plans, record execution checklist + verification evidence in `tasks/todo.md` as part of implementation (not as a follow-up suggestion).
- Pattern: `pre-commit` local hook used `language: system` with `entry: python ...`, causing Windows commit failures (`exit code 9009`) when commit-time PATH lacked `python`.
- Rule: For Python-based local pre-commit hooks, prefer `language: python` to avoid PATH-dependent failures across IDE/CLI commit environments.
- Pattern: User reported `run_streamlit.bat` showed no output after simplifying launcher commands.
- Rule: In Windows batch scripts, keep `call` when invoking `conda activate` (batch-to-batch); otherwise control flow may not return to run the Streamlit command.
- Rule: For local launcher scripts, preserve `%*` argument forwarding and include explicit activation-failure messages so double-click runs do not fail silently.
- Pattern: `conda activate` failed in CMD with `Run 'conda init' before 'conda activate'` even though conda was installed.
- Rule: For CMD launchers, prefer resolving `conda.bat` (via `CONDA_EXE` or known `condabin` paths) and call it directly; do not assume `conda init cmd.exe` was run.

## 2026-02-24
- Pattern: Light theme looked low-contrast despite acceptable token-level WCAG checks.
- Rule: For theme toggles, verify readability at component level (native Streamlit widgets, sidebar controls, markdown body text), not only token contrast pairs.
- Rule: Avoid applying `text_muted` to global markdown paragraphs; reserve muted color for captions/labels and keep body copy at base text color.
- Pattern: Light background + dark dataframe default skin created visual mismatch and poor readability.
- Rule: For Streamlit tables, apply theme at component level (`pandas.Styler`) first and keep CSS selectors only as fallback; do not rely on root CSS tokens alone.
- Pattern: Streamlit top chrome (`stHeader`/`stDecoration`) and Glide Data Grid can ignore app background tokens and remain dark.
- Rule: In light mode, explicitly style `stHeader`, `stDecoration`, and Glide Data Grid CSS variables; token updates alone are insufficient.
- Pattern: FX shock behavior was documented but runtime signal path still passed `fx_change_pct=0.0`, disabling the alert in practice.
- Rule: For risk/input wiring changes, validate end-to-end propagation in integration tests (`caller -> build_signal_table -> scoring`) instead of relying only on helper unit tests.
- Rule: Do not hardcode neutral sentinel values (`0.0`) for optional market inputs; pass computed runtime values and use `NaN` when data is unavailable.

## 2026-03-02
- Pattern: `app.py` UI 문자열이 모지바케로 저장되어 한글이 깨지고, `st.warning(..., icon=...)`에 깨진 문자열이 들어가 `invalid emoji` 예외가 발생함.
- Rule: UI 텍스트/주석이 많은 파일은 수정 후 `python -m py_compile <file>`와 `rg "�|\\?곗|\\?꾩|\\?좏"` 같은 깨짐 패턴 점검을 최소 1회 실행한다.
- Rule: Streamlit `icon=` 인자는 항상 단일 유효 이모지(`⚠️`, `✅` 등)만 사용하고, 일반 문자열/깨진 문자열은 절대 넣지 않는다.
- Rule: VS Code/IDE 및 저장 설정을 UTF-8(권장: UTF-8 without BOM)으로 고정하고, 인코딩 전환 저장(CP949/EUC-KR)을 금지한다.

## 2026-03-08
- Pattern: User reported that the cycle timeline visually collapsed into near-invisible lines, so regime colors were technically present in code but not legible in the actual chart.
- Rule: For time-interval visuals, verify that the rendering primitive matches the semantics of a duration; do not use point-like bars for date spans when the goal is a visible band/timeline.
- Rule: When a user calls out “can’t distinguish periods,” inspect both palette contrast and chart geometry before treating it as a theme-only bug.
- Pattern: A real-network `bootstrap_warehouse.py` run failed immediately with `_duckdb.IOException` because a local `streamlit run app.py` process still held a write lock on `data/warehouse.duckdb`.
- Rule: Before any warehouse bootstrap/backfill/sync that writes to DuckDB, check for repo-local `python -m streamlit run app.py` or other writer processes and stop them first; otherwise treat file-lock errors as environment/process conflicts before changing code.
- Pattern: `git push` to GitHub failed because a regenerated `data/warehouse.duckdb` blob in the latest commit exceeded the 100 MB file limit.
- Rule: Before committing or pushing, inspect staged/generated artifacts for size-sensitive paths (`data/`, `backups/`, `logs/`) and keep local DuckDB files untracked via `.gitignore` plus `git rm --cached` when they were already tracked.

## 2026-04-07
- Pattern: pykrx를 데이터 소스로 재선택하려는 상황에서, pykrx가 KRX 서버 응답 구조 변경으로 실제로 broken 상태일 수 있음.
- Rule: `KRX_PROVIDER=PYKRX`로 복귀하기 전에 반드시 pykrx 실시간 호출이 정상인지 먼저 확인한다.
  - 확인 방법: `python -c "from pykrx import stock; df = stock.get_index_ohlcv('20260101', '20260107', '1001'); print(df)"` 실행 후 비어 있지 않은 DataFrame이 반환되는지 검증.
  - pykrx GitHub issues (<https://github.com/sharebook-kr/pykrx/issues>) 에서 open 이슈 및 최근 릴리즈 확인.
  - 빈 DataFrame, `'시장'`/`'지수명'` KeyError, `IndexTicker singleton had empty df` 경고 중 하나라도 발생하면 pykrx가 아직 broken 상태이므로 OPENAPI를 유지한다.
- Context: 2026-02-27 KRX 서버 변경으로 pykrx `get_index_ohlcv`가 `'지수명'` 컬럼 누락 문제를 일으켰고, 1.2.4(최신)에서도 미수정(issue #276). 이 프로젝트는 KRX OpenAPI 경로로 전환해 해결함.

## 2026-03-30
- Pattern: Streamlit app load hit a DuckDB write-lock error in US macro loading because `sync_provider_macro()` always called `upsert_macro_dimension()` before checking whether cached warehouse macro data already satisfied the request.
- Rule: In warehouse-backed sync paths, never open a write connection on a cache-hit fast path; perform completeness checks using read-only calls first and only switch to write mode after a live refresh is actually required.
- Rule: When fixing same-process DuckDB lock issues, add a regression test that proves the cached read path avoids both live fetches and dimension upserts.

## 2026-04-09
- Pattern: A KRX data-path change passed focused integration tests but still broke unrelated shared UI tests in CI because the final verification skipped full `pytest -q`.
- Rule: Before pushing any change that touches shared UI exports or modules re-exported through compatibility barrels (for example `src/ui/components.py` -> `src/ui/tables.py`), always run full `pytest -q`, not only targeted tests.
- Rule: When tests describe a renderer as using "native dataframe", keep the `st.dataframe` payload as a plain `DataFrame`; do not switch to `pandas.Styler` unless the test contract is intentionally updated in the same change.

## 2026-04-11
- Pattern: A new KRX investor-flow failure looked at first like another response-key/schema drift, but the stronger root cause was 2026 KRX Data Marketplace login gating that returned non-JSON HTML to unauthenticated requests.
- Rule: When KRX raw endpoints start failing with `JSONDecodeError` / `Expecting value`, treat login/auth gating as a first-class hypothesis before iterating on payload keys or date candidates.
- Rule: For KRX non-JSON responses, always capture and classify the body as `AUTH_REQUIRED`, `ACCESS_DENIED`, `NON_JSON_EMPTY`, or `NON_JSON_RESPONSE`; never reduce it to a generic "empty constituent list".
- Rule: If a third-party source becomes login-gated, add optional authenticated-session support and a precise blocker message in the same change; do not present it as a completed data fix while credentials are still unsupported.

## 2026-04-18
- Pattern: CI pytest failed with `ModuleNotFoundError: No module named 'openpyxl'` while all tests passed locally — `openpyxl` was installed locally as a transitive dependency but was never listed in `requirements.txt`.
- Rule: When source code uses a library via `engine="openpyxl"` or similar string-based plugin references (pandas, SQLAlchemy, etc.), always add that library as an explicit dependency in `requirements.txt` — implicit transitive availability is not guaranteed in clean CI environments.
- Rule: After adding any new `import`, `engine=`, or plugin reference, run `pip install --no-deps -r requirements.txt` in a clean venv to verify all dependencies are declared.
- Pattern: CI warning about Node.js 20 deprecation for `actions/checkout@v4` and `actions/setup-python@v5`.
- Rule: Keep GitHub Actions pinned to the latest major version (`@v6` as of 2026-04); review action versions when CI deprecation warnings appear.
- Pattern: A progress feature was implemented, but the user still could not see it when pressing the sidebar refresh button because the rendering surface was too easy to miss.
- Rule: For Streamlit progress/status UX tied to sidebar actions, render the live progress state in at least one main-page-visible surface in addition to any sidebar placeholder; do not assume sidebar-only placement is sufficiently visible.

## 2026-04-19
- Pattern: Logic/semantic migration was implemented, but the user-facing explainer copy still described the pre-migration behavior.
- Rule: When a scoring or classification semantic changes, update every matching explainer/help/disclaimer string in the same change; do not treat copy as a follow-up.
- Rule: Add at least one regression test that exercises the updated explainer/body copy with the runtime parameters it interpolates.

## 2026-04-20
- Pattern: KR 섹터 코드를 해석할 때 로컬 `sector_map.yml`의 오래된 이름/코드 조합을 그대로 신뢰해, 사용자가 준 KRX 공식 화면 기준 분류와 어긋난 응답이 나왔다.
- Rule: KR 지수/섹터 코드 의미를 다룰 때는 먼저 KRX 공식 목록(`주가지수 > KRX`, `finder_equidx` 또는 동등 공식 소스)으로 코드↔이름을 검증하고, 로컬 설정값은 보조 메타데이터(예: export flag, ETF 매핑)로만 취급한다.
- Rule: 공식 소스와 로컬 설정이 충돌하면 비벤치마크 KR 지수명은 공식 소스를 우선하고, 로컬 설정으로 공식 이름을 덮어쓰지 않는다.

## 2026-04-23
- Pattern: `DESIGN.md` 기준으로 프론트엔드를 이식했지만, 실제 한글 UI에서는 라틴 기준 디스플레이 스케일과 letter-spacing이 그대로 남아 폰트가 커 보이고 밀도가 어색했다.
- Rule: 디자인 레퍼런스가 영문 중심이어도 한글 UI는 별도로 검증한다. `display/body/caption/button` 스케일과 global letter-spacing을 함께 점검하고, 한글이 많은 카드/패널/캡션은 한 단계 더 보수적으로 줄인다.
- Rule: 토큰 마이그레이션이 끝난 뒤에도 `버튼 radius`, `탭 radius`, `메트릭/카드 제목 크기`처럼 실제 체감 계층을 만드는 CSS 값을 다시 확인한다. 토큰만 맞추고 체감 비례를 방치하지 않는다.
- Pattern: 설명성 문구를 `analysis-toolbar__title` 같은 제목 계층에 직접 넣어, 한글 긴 문장이 패널 타이틀급으로 읽혔다.
- Rule: 긴 안내 문장은 제목과 분리한다. 제목은 짧게 두고, 정책/범위 설명은 별도 description/caption 계층으로 내려서 body-small 이하로 렌더한다.

## 2026-05-05
- Pattern: 데이터 수집 이력 샘플 요구가 바뀌었는데 계획/테스트 문구가 기존 "최신 5 + 오래된 5" 계약을 계속 들고 있었다.
- Rule: 사용자가 샘플 기준을 수정하면 warehouse query, UI copy, 테스트명/기대값, 계획 산출물까지 같은 샘플 계약으로 동시에 갱신한다.
- Pattern: 데이터 수집 이력 화면에 연구 뷰용 downstream 필터가 공통 렌더링되어, 화면 목적과 무관한 컨트롤이 노출됐다.
- Rule: 페이지 공통 컨트롤은 모든 라우트에 무조건 렌더하지 않는다. `quality`/운영 관리 화면처럼 진단 목적이 다른 페이지는 라우팅 테스트로 필터·분석 컨트롤 비노출을 고정한다.
- Pattern: Market refresh was treated as fixed after improving the visible fallback notice, but KRX raw-cache coverage still trusted poisoned parquet snapshots with empty `close` values.
- Rule: For refresh/cache bugs, prove the exact failing provider path with a manual reproduction and test the cache-validity predicate, not only the UI notice text.
- Rule: Raw market cache coverage must be based on numeric, non-null close values for each required business date; earliest/latest dates alone are insufficient because internal NaN rows can be forward-filled later.
- Rule: When repairing an older requested window, ensure the latest raw-cache snapshot label is updated or merged so a newer poisoned snapshot cannot keep winning `_latest_raw_cache_file()`.

## 2026-05-17
- Pattern: Theme taxonomy feedback was partly structural, but review caught that whole-universe readiness also required real estate/REITs and semantic aliases/rules for value-up and network infrastructure.
- Rule: When implementing taxonomy feedback, do not stop at axis/tag IDs. Convert every feedback-listed semantic phrase into either a required alias, inclusion-rule assertion, mapping rule, or explicit scoped-out non-goal.
- Rule: For taxonomy rename requests, add negative tests for the removed ID and positive tests for all required replacement aliases, not just the first alias.
- Pattern: 테마 ETF 기능이 cache-only loader와 refresh button 테스트는 통과했지만, 실제 환경에서는 warehouse에 0건이라 화면이 `UNAVAILABLE`였고 Streamlit/Python 프로세스의 DuckDB write lock 때문에 button refresh 결과도 저장되지 않았다.
- Rule: 새 warehouse-backed UI 기능은 구현 검증 시 `read_*` 실제 row count와 live refresh의 persistence 결과를 둘 다 수동 재현한다. 단위 테스트의 fake upsert 성공만으로 완료 처리하지 않는다.
- Rule: Streamlit 내부 refresh가 DuckDB에 write해야 하는 기능은 write lock 실패 시에도 live fetch 결과를 화면에 표시할 session snapshot 또는 equivalent fallback을 제공한다.
- Rule: 여러 representative ETF를 둔 proxy lens는 첫 번째 코드만 신뢰하지 않는다. 캐시/live 데이터가 있는 대표 ETF를 선택하는 fallback을 테스트로 고정한다.
- Pattern: theme_taxonomy 수집/품질 레이어는 구현했지만 첫 화면 검토 후보 카드는 여전히 runtime `sector_name`을 직접 렌더해 KRX 섹터명으로 보였다.
- Rule: 분류 체계를 바꾸거나 overlay를 추가할 때는 데이터 수집/상태 패널뿐 아니라 첫 화면 카드, 표, 히트맵처럼 사용자가 결론으로 읽는 모든 주요 렌더링 경로에 표시 모델이 연결됐는지 확인한다.
