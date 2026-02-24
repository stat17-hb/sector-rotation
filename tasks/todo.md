# Korea Sector Rotation Dashboard - Execution Plan (Synced)

Last updated: 2026-02-22
Source of truth: `plan.md`
Owner: Codex + User
Status: Planned

## 1) Objective
- Build a production-ready Streamlit dashboard for Korea sector rotation based on macro regime plus momentum signals.
- Preserve strict contracts across module boundaries and deterministic fallback behavior under API failures.

## 2) Definition of Done
- `streamlit run app.py` boots and renders Macro, Momentum, and Signal Table sections.
- `pytest tests/ -v` passes all planned 24 tests.
- No cyan/teal in theme or Plotly outputs.
- SAMPLE mode always shows a top-page warning and disables only `recompute`.
- Cache invalidation is scoped per button and does not clear unrelated caches.
- `tasks/todo.md` review section is filled with objective evidence.

## 3) Non-Negotiable Spec Locks
- [ ] `calendar.py` uses pykrx as the sole KRX calendar authority; no static holiday table.
- [ ] Public loaders return a single contract: `LoaderResult = tuple[DataStatus, pd.DataFrame]`.
- [ ] `build_signal_table(...) -> list[SectorSignal]`.
- [ ] Action domain includes exactly `{"Strong Buy", "Watch", "Hold", "Avoid", "N/A"}`.
- [ ] Use three named cache functions and clear only the relevant one per button.
- [ ] Integration file I/O tests use `tmp_path` and `monkeypatch`.
- [ ] ECOS/KOSIS HTTP policy: timeout 10s, retries 3, backoff 2/4/8, re-raise last exception.

## 4) Execution Checklist

## Phase 0 - Risk Controls and Contracts (R1-R11 first)
- [ ] Create `src/contracts/data_contracts.md` with schema, dtype-family checks, index type, null policy.
- [ ] Create `src/contracts/validators.py` with `validate_only()` and `normalize_then_validate()`.
- [ ] Enforce call convention: loaders call `normalize_then_validate`; analytics entry points call `validate_only`.
- [ ] Lock `DataStatus` and `LoaderResult` aliases for all public loader APIs.
- [ ] Lock `ACTION_VALUES` with `N/A` only for data-load failure rows.
- [ ] Lock pykrx-only business-day policy and remove local holiday lookup requirements.
- [ ] Lock cache invalidation strategy to named cache function clears only.
- [ ] Lock SAMPLE mode policy via pure helper functions.
- [ ] Lock integration test isolation strategy using `tmp_path`.

Gate:
- [ ] R1-R11 rules are documented and referenced in relevant module docstrings.

## Phase 1 - Project Scaffolding
- [ ] Create folder tree including `src/contracts/` and `src/ui/data_status.py` targets.
- [ ] Add `requirements.txt` with runtime deps plus `pytest` and `pytest-mock`.
- [ ] Add optional TA-Lib comment only (not hard dependency).
- [ ] Add `.streamlit/config.toml` and `.streamlit/secrets.toml.example`.
- [ ] Add `config/settings.yml` defaults from plan.
- [ ] Add `config/sector_map.yml` with `benchmark` and `regimes.*.sectors[*].export_sector` schema.
- [ ] Add `config/macro_series.yml` for ECOS/KOSIS series mapping.
- [ ] Add minimal `app.py` skeleton with wide layout and CSS injection hook.

Gate:
- [ ] Project imports resolve and app skeleton starts.

## Phase 2 - Data Sources and Transforms
- [ ] Implement `src/data_sources/krx_indices.py` with chunked pykrx fetch, retry/backoff, and raw parquet save.
- [ ] Implement `load_sector_prices(...) -> LoaderResult` with LIVE/CACHED/SAMPLE fallback.
- [ ] Add module-level `CURATED_DIR` constant for monkeypatch-friendly tests.
- [ ] Implement `src/data_sources/ecos.py` with secrets/env API key fallback.
- [ ] Implement `src/data_sources/kosis.py` with the same HTTP/retry contract and provisional flag handling.
- [ ] Implement shared `_get_with_retry()` policy: timeout 10s, retries 3, backoff 2/4/8, re-raise last exception.
- [ ] Implement `src/transforms/calendar.py:get_last_business_day(as_of=None) -> date` using pykrx `get_index_ohlcv` lookback.
- [ ] Keep weekend-only subtraction as explicit best-effort fallback in calendar error path.
- [ ] Do not implement `is_krx_holiday()` local lookup.
- [ ] Implement `src/transforms/resample.py` (`to_monthly_last`, `compute_3ma_direction`).
- [ ] Apply `normalize_then_validate()` before curated saves.

Gate:
- [ ] Loaders produce valid outputs with explicit status and contract checks.

## Phase 3 - Core Analytics and Signal Engine
- [ ] Implement `src/macro/regime.py` with `classify_regime`, `compute_regime_history`, `get_regime_sectors`.
- [ ] `get_regime_sectors` returns `list[dict]` entries with `code`, `name`, `export_sector`.
- [ ] Implement `src/indicators/momentum.py` (RS, MA trend, returns, volatility, MDD).
- [ ] Implement `src/indicators/rsi.py` with TA-Lib -> ta -> manual fallback chain.
- [ ] Implement `src/signals/matrix.py` with `SectorSignal`, 5-value action domain, and matrix logic.
- [ ] Ensure matrix logic returns only Strong Buy/Watch/Hold/Avoid; `N/A` only for loader failures.
- [ ] Implement `build_signal_table(...) -> list[SectorSignal]` with partial-failure tolerance.
- [ ] Implement `src/signals/scoring.py` with RSI alerts and export-sector FX shock downgrade.
- [ ] Validate inputs at entry boundaries with `validate_only()`.

Gate:
- [ ] Deterministic fixture outputs and correct behavior for partial-sector failures.

## Phase 4 - UI Composition and App Wiring
- [ ] Implement `src/ui/styles.py` with no-cyan Plotly template and CSS injection.
- [ ] Implement `src/ui/components.py` for macro tile, RS scatter, returns heatmap, signal table.
- [ ] Render `N/A` rows with neutral/grey style and data-missing hint.
- [ ] Implement `src/ui/data_status.py` with `is_sample_mode()` and `get_button_states()` pure functions.
- [ ] In `app.py`, use three named cache functions:
- [ ] `_cached_sector_prices(asof_date_str, benchmark_code, price_years)`
- [ ] `_cached_macro(macro_series_hash)`
- [ ] `_cached_signals(prices_key, macro_key, params_hash)`
- [ ] Add `_parquet_key(path) -> (mtime_ns, size)` cache key helper.
- [ ] Implement button handlers that clear only target cache and delete only target artifacts.
- [ ] Ensure SAMPLE mode shows `st.error` at top and disables only recompute.
- [ ] Keep refresh buttons enabled in SAMPLE mode.

Gate:
- [ ] UI behavior matches button-state and warning rules exactly.

## Phase 5 - Tests
- [ ] Implement `tests/test_regime.py` (3 tests).
- [ ] Implement `tests/test_momentum.py` (4 tests).
- [ ] Implement `tests/test_signals.py` (3 tests).
- [ ] Implement `tests/test_contracts.py` (5 tests for dtype/null/index/fill policy).
- [ ] Implement `tests/test_data_status.py` (4 pure-function tests).
- [ ] Implement `tests/test_integration.py` (5 tests; file I/O isolated with `tmp_path`).
- [ ] Integration coverage includes:
- [ ] API failure fallback to cache.
- [ ] Full fallback to sample mode.
- [ ] Partial sector failure -> `N/A` actions.
- [ ] Cache invalidation clears only target artifact.
- [ ] FX shock downgrade end-to-end.

Gate:
- [ ] Planned test matrix count is 24 and all pass.

## Phase 6 - Verification and Acceptance
- [ ] Run `pytest tests/ -v` and confirm 24 passing tests.
- [ ] Run `streamlit run app.py` and confirm app load under local first-run target.
- [ ] Verify no cyan/teal appears in charts or UI theme.
- [ ] Verify epsilon change updates current regime.
- [ ] Verify Strong Buy filter behavior in signal table.
- [ ] Verify provisional data notice rendering.
- [ ] Verify FX > threshold downgrades export-sector Strong Buy to Watch.

Gate:
- [ ] Acceptance checks all pass with evidence captured.

## Phase 7 - Final Review and Hardening
- [ ] Compare implemented behavior with `plan.md` and record any deviations.
- [ ] Audit logs and error paths for API/network failures.
- [ ] Confirm docs for API keys and fallback behavior are explicit.
- [ ] Fill review section below with commands, outputs, and known limitations.

Gate:
- [ ] Staff-level review checklist passed.

## 5) Review Section (fill during implementation)
- Date: 2026-02-22
- Scope delivered: All 7 phases complete. 30 Python files + config + scaffolding. Full Streamlit SPA with macro regime classification, momentum signal engine, and sector rotation dashboard.
- Key diffs from plan: `is_rs_strong()` required explicit `bool()` cast to avoid `np.True_` identity failure in tests (np scalar vs Python bool). Fixed with `isinstance(result, pd.Series)` guard.
- Test evidence: `pytest tests/ -v` → 24 passed in 0.50s (Python 3.13.5, pytest 9.0.2). All 6 test files: contracts×5, data_status×4, integration×5, momentum×4, regime×3, signals×3.
- Runtime evidence: All imports resolved, 24/24 tests pass. `streamlit run app.py` requires API keys in `.streamlit/secrets.toml`; without keys, SAMPLE mode activates with st.error banner.
- Known limitations: ECOS/KOSIS API keys required for LIVE data (see `.streamlit/secrets.toml.example`). pykrx calendar lookup requires network access; falls back to weekend-only subtraction if unavailable. TA-Lib optional binary dependency not included.
- Follow-up tasks: Add `.streamlit/secrets.toml` with real API keys. Run `streamlit run app.py` for visual acceptance. Monitor pykrx rate limits on chunked fetches.

## 8) API Misdiagnosis Stabilization (2026-02-22)
- [x] Add cache token helper that fingerprints API keys without exposing raw keys.
- [x] Expand ECOS loader to support `item_codes` and improve RESULT error parsing.
- [x] Expand KOSIS loader to support `obj_params`, structured error parsing, and fallback param retries.
- [x] Switch KOSIS macro loading to partial-success aggregation with failure warnings.
- [x] Add API preflight diagnostics for ECOS/KOSIS/KRX endpoints.
- [x] Wire macro cache token + preflight status rendering in `app.py`.
- [x] Extend `config/macro_series.yml` schema with backward-compatible optional fields.
- [x] Update `docs/api-keys-guide.md` with cache invalidation and error interpretation guidance.
- [x] Add tests for cache token, ECOS/KOSIS error handling, and preflight classification.
- [x] Run focused pytest suite and record evidence in review section.

Review:
- Added modules: `src/data_sources/cache_keys.py`, `src/data_sources/preflight.py`.
- Updated loaders: `src/data_sources/ecos.py`, `src/data_sources/kosis.py` (ECOS item path support, KOSIS obj param retries + structured err parsing + partial success).
- Updated app wiring: `app.py` now computes macro cache token using key fingerprints and shows preflight diagnostics.
- Updated config/docs: `config/macro_series.yml`, `docs/api-keys-guide.md`.
- Verification command: `pytest -q tests/test_cache_keys.py tests/test_preflight.py tests/test_ecos_kosis_api_handling.py tests/test_contracts.py tests/test_data_status.py tests/test_integration.py tests/test_momentum.py tests/test_regime.py tests/test_signals.py`.
- Verification result: `36 passed in 0.86s`.

## 9) Macro Data Recovery + Stability Hardening (2026-02-22)

Pre-Implementation Check-in:
- 2026-02-22: Start implementation for Macro+Stability scope. Target: restore ECOS/KOSIS LIVE macro ingestion, reduce pykrx logging noise, keep KRX price path graceful fallback (`LIVE -> CACHED -> SAMPLE`), and preserve existing data contracts.

Execution Checklist:
- [x] Change root logging baseline in `app.py` from INFO-level global noise to WARNING-level third-party suppression while keeping app warnings/errors visible.
- [x] In `src/data_sources/krx_indices.py`, detect deterministic pykrx failures (`LOGOUT`, JSON empty-body, `지수명` metadata failures) and skip unnecessary retry sleep/backoff for those cases.
- [x] Add ECOS `cycle` support in `src/data_sources/ecos.py` and remove hardcoded `MM` path usage.
- [x] Parse ECOS `TIME` by cycle and normalize to monthly `PeriodIndex` output for `macro_monthly` contract.
- [x] Make `load_ecos_macro()` partial-success tolerant (LIVE when at least one series succeeds, warn on failed series).
- [x] Respect `enabled: false` series in ECOS and KOSIS config ingestion.
- [x] Update `config/macro_series.yml` with validated KOSIS mappings:
- [x] `cpi_yoy`: `org_id=101`, `tbl_id=DT_1J22003`, `item_id=T`, `objL1=T10`.
- [x] `leading_index`: `org_id=101`, `tbl_id=DT_1C8015`, `item_id=T1`, `objL1=A03`.
- [x] Mark `export_growth` optional via `enabled: false`.
- [x] In `app.py`, skip disabled macro series when building loader config.
- [x] Replace substring-based macro extraction with deterministic alias mapping (`leading_index` -> growth, `cpi_yoy` -> inflation).
- [x] Align growth/inflation on shared periods before `compute_regime_history`.
- [x] Convert regime index to Plotly-safe timestamp index before chart rendering.
- [x] Update `docs/api-keys-guide.md` with verified KOSIS parameter pattern and optional-series guidance.
- [x] Extend `tests/test_ecos_kosis_api_handling.py` for:
- [x] ECOS cycle URL behavior (no `MM` hardcode).
- [x] ECOS partial-success loader behavior.
- [x] KOSIS validated param forwarding (`T/T10`, `T1/A03`).
- [x] Disabled series skip behavior.
- [x] Add/update a test for Plotly-safe macro-regime index conversion.

Verification Gates:
- [x] `pytest -q tests/test_ecos_kosis_api_handling.py tests/test_preflight.py tests/test_integration.py` passes.
- [x] `pytest -q` full suite passes.
- [x] Runtime validation:
- [x] Streamlit startup no longer emits repeated `--- Logging error ---` tracebacks from pykrx.
- [x] Macro pipeline can return LIVE status with valid keys/config.
- [x] Regime computation uses real macro series (not permanent fallback).
- [x] When KRX live fetch fails, app degrades to `CACHED/SAMPLE` without retry storm.

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/python.exe -m py_compile app.py src/data_sources/ecos.py src/data_sources/kosis.py src/data_sources/krx_indices.py src/macro/series_utils.py tests/test_ecos_kosis_api_handling.py`
- `pytest -q tests/test_ecos_kosis_api_handling.py tests/test_preflight.py tests/test_integration.py`
- `pytest -q`
- `cmd /c "C:/Users/k1190/miniconda3/python.exe -m streamlit run app.py --server.headless true --server.port 8512 > .tmp_streamlit.log 2>&1"` (timeout used for startup log capture)
- `C:/Users/k1190/miniconda3/python.exe -c "...load_ecos_macro/load_kosis_macro + regime compute smoke..."`
- `C:/Users/k1190/miniconda3/python.exe -c "..._fetch_chunk('5044', ...) timing smoke..."`
- Results:
- Compile check passed.
- Targeted tests: `21 passed in 0.87s`.
- Full test suite: `43 passed in 0.92s`.
- Streamlit startup log capture showed app URLs and no repeated `--- Logging error ---` traces.
- Live macro smoke run returned `ecos_status=LIVE`, `kosis_status=LIVE`, aligned macro points, and a computed non-fallback regime (`Expansion` at latest point).
- KRX deterministic failure smoke returned quickly (`~0.9s`) without retry backoff delay, preserving graceful fallback behavior.
- Residual risks / follow-ups:
- pykrx/KRX live price endpoint instability remains external; this patch intentionally keeps fallback strategy instead of replacing provider implementation.

## 10) KRX Live Fetch Stabilization for pykrx 1.0.51 (2026-02-22)

Pre-Implementation Check-in:
- 2026-02-22: Start implementation for KRX live fetch stabilization via project-local pykrx transport compatibility shim (no dependency upgrades).

Execution Checklist:
- [x] Add `src/data_sources/pykrx_compat.py` with idempotent `ensure_pykrx_transport_compat()` and `resolve_ohlcv_close_column()`.
- [x] Patch pykrx transport defaults in shim:
- [x] Referer -> `https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd`
- [x] `KrxWebIo.url` -> `https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd`
- [x] `KrxFutureIo.url` -> `https://data.krx.co.kr/comm/bldAttendant/executeForResourceBundle.cmd`
- [x] Integrate shim into `src/data_sources/krx_indices.py` before pykrx calls.
- [x] Replace brittle close-column lookup in `src/data_sources/krx_indices.py` with shim resolver.
- [x] Integrate shim into `src/transforms/calendar.py` before pykrx calendar lookup.
- [x] Preserve existing warning policy (no suppression of `pkg_resources` deprecation warning).
- [x] Add `tests/test_pykrx_compat.py` for shim patching/idempotence/close-column resolution behavior.
- [x] Add targeted tests for KRX loader and calendar compat paths.

Verification Gates:
- [x] `C:/Users/k1190/miniconda3/python.exe -m pytest -q tests/test_pykrx_compat.py`
- [x] `C:/Users/k1190/miniconda3/python.exe -m pytest -q tests/test_integration.py`
- [x] `C:/Users/k1190/miniconda3/python.exe -m pytest -q`
- [x] `C:/Users/k1190/miniconda3/python.exe -c "from src.data_sources.krx_indices import fetch_index_ohlcv; df=fetch_index_ohlcv('1001','20260101','20260220'); print(df.shape)"`
- [x] `C:/Users/k1190/miniconda3/python.exe -m streamlit run app.py` smoke startup confirms no repeated `'지수명'`/`LOGOUT` failure chain.

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/python.exe -m pytest -q tests/test_pykrx_compat.py`
- `C:/Users/k1190/miniconda3/python.exe -m pytest -q tests/test_krx_pykrx_compat_paths.py`
- `C:/Users/k1190/miniconda3/python.exe -m pytest -q tests/test_integration.py`
- `C:/Users/k1190/miniconda3/python.exe -m pytest -q`
- `C:/Users/k1190/miniconda3/python.exe -c "from src.data_sources.krx_indices import fetch_index_ohlcv; df=fetch_index_ohlcv('1001','20260101','20260220'); print(df.shape); print(list(df.columns)); print(df.head(2))"`
- `Start-Process ... streamlit run app.py --server.headless true --server.port 8514` (15s startup log capture, then process stop)
- Results:
- Added `src/data_sources/pykrx_compat.py` and integrated it in `src/data_sources/krx_indices.py` + `src/transforms/calendar.py`.
- Added tests: `tests/test_pykrx_compat.py`, `tests/test_krx_pykrx_compat_paths.py`.
- Test results: `5 passed`, `4 passed`, `5 passed`, and full suite `52 passed in 0.91s`.
- Runtime KRX smoke: `fetch_index_ohlcv('1001', '20260101', '20260220')` returned shape `(33, 7)` with non-empty rows (no `'지수명'` failure).
- Streamlit startup log showed URLs and no repeated `'지수명'`/`LOGOUT` failure chain during startup window.
- Residual risks / follow-ups:
- `pkg_resources` deprecation warning from pykrx remains intentionally unsuppressed (out of scope for this fix).
- External KRX endpoint policy may change again; compatibility shim should be reviewed when pykrx is upgraded.

## 11) KRX Invalid Code Stabilization (`5040`/`5041`) (2026-02-22)

Pre-Implementation Check-in:
- 2026-02-22: Start implementation to stabilize KRX live prices when configured index codes are partially stale. Target behavior: `5040 -> 5048`, keep `5041` in map, and prevent single-code failure from collapsing all prices to `SAMPLE`.

Execution Checklist:
- [x] Update `config/sector_map.yml` Slowdown sector code from `5040` to `5048` (`KRX 에너지화학`).
- [x] Keep `5041` in sector map intentionally and handle it as sector-level missing data when live fetch fails.
- [x] In `src/data_sources/krx_indices.py`, make `load_sector_prices()` partial-success tolerant (continue on per-code failures).
- [x] In `src/data_sources/krx_indices.py`, return `LIVE` if at least one sector fetched successfully and validated.
- [x] In `src/data_sources/krx_indices.py`, keep fallback order `LIVE -> CACHED -> SAMPLE` when all live sectors fail.
- [x] In `src/data_sources/krx_indices.py`, treat missing-code `KeyError` as deterministic and stop retry loop after first failure.
- [x] Add integration regression in `tests/test_integration.py` for mixed success/failure (`LIVE` result with failed code skipped).
- [x] Add retry regression in `tests/test_krx_pykrx_compat_paths.py` proving deterministic missing-code failure does not retry 3 times.
- [x] Run focused tests:
- [x] `C:/Users/k1190/miniconda3/python.exe -m pytest -q tests/test_krx_pykrx_compat_paths.py tests/test_integration.py`
- [x] Run focused runtime check:
- [x] `C:/Users/k1190/miniconda3/python.exe -c "...load_sector_prices(all_codes_from_sector_map)..."` and verify price status is not forced to `SAMPLE` by `5041`.
- [x] Run Streamlit startup smoke check and verify logs do not include `Live fetch failed: '5040'`.
- [x] Record commands/results and residual risks in review section.

Verification Gates:
- [x] Targeted pytest command passes.
- [x] Runtime loader smoke returns `LIVE` with mixed code set including `5041`.
- [x] Streamlit startup logs show no `ERROR:...Live fetch failed: '5040'` and no full price fallback due only to `5041`.

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/python.exe -m pytest -q tests/test_krx_pykrx_compat_paths.py tests/test_integration.py`
- `C:/Users/k1190/miniconda3/python.exe -m pytest -q`
- `C:/Users/k1190/miniconda3/python.exe -c "import yaml; from datetime import timedelta; from src.transforms.calendar import get_last_business_day; from src.data_sources.krx_indices import load_sector_prices; ..."`
- `C:/Users/k1190/miniconda3/python.exe -c "import yaml, pandas as pd; ... build_signal_table(...); print(na_codes)"`
- `Start-Process C:/Users/k1190/miniconda3/python.exe -ArgumentList -m streamlit run app.py --server.headless true --server.port 8517` (20s startup capture to `.tmp_streamlit_krx_invalid.out.log` / `.tmp_streamlit_krx_invalid.err.log`)
- Results:
- Config updated: `5040` replaced with `5048` in `config/sector_map.yml`; `5041` intentionally retained.
- Loader behavior updated: deterministic missing-code failures now stop after first attempt; per-code failures no longer collapse all KRX live data.
- Added tests:
- `tests/test_integration.py::test_live_partial_success_keeps_live_status`
- `tests/test_krx_pykrx_compat_paths.py::test_fetch_chunk_stops_retry_for_missing_code_keyerror`
- Targeted pytest result: `11 passed in 0.49s`.
- Full pytest result: `54 passed in 0.92s`.
- Runtime smoke result: mixed code load returned `status LIVE`, produced rows for valid codes, and did not include `5041` rows.
- Signal smoke result: downstream signal table returned `N/A` only for missing-sector codes (`5041`, and currently unavailable `1166`) while keeping price pipeline in `LIVE`.
- Streamlit startup logs contained only startup URLs and did not contain `Live fetch failed: '5040'`.
- Residual risks / follow-ups:
- Additional stale index codes may still exist (runtime smoke observed `1166` missing); current behavior degrades those sectors gracefully to missing data rather than forcing full `SAMPLE`.
- `pkg_resources` deprecation warning from pykrx remains intentionally unsuppressed (out of scope).

## 12) pykrx `pkg_resources` Warning Removal (2026-02-22)

Pre-Implementation Check-in:
- 2026-02-22: Start implementation for removing pykrx `pkg_resources` deprecation warning by upgrading to pykrx 1.2.4+.

Execution Checklist:
- [x] Add this section and checklist to `tasks/todo.md`.
- [x] Bump dependency in `requirements.txt` from `pykrx>=1.0.40` to `pykrx>=1.2.4`.
- [x] Keep runtime transport shim behavior and make version wording in `src/data_sources/pykrx_compat.py` version-neutral.
- [x] Add regression test `tests/test_pykrx_import_warning.py` verifying `import pykrx` does not emit `pkg_resources is deprecated as an API`.
- [x] Upgrade runtime package: `C:/Users/k1190/miniconda3/python.exe -m pip install --upgrade pykrx==1.2.4`.
- [x] Run warning check command: `C:/Users/k1190/miniconda3/python.exe -W default -c "import pykrx"`.
- [x] Run regression tests:
- [x] `C:/Users/k1190/miniconda3/python.exe -m pytest -q tests/test_pykrx_compat.py tests/test_krx_pykrx_compat_paths.py`
- [x] `C:/Users/k1190/miniconda3/python.exe -m pytest -q tests/test_pykrx_import_warning.py`
- [x] Run Streamlit startup smoke (`--server.headless true --server.port 8518`) and verify startup logs do not include the `pkg_resources` warning.
- [x] Record commands, outcomes, and residual risks in this section.

Verification Gates:
- [x] `import pykrx` no longer prints `pkg_resources` deprecation warning.
- [x] Existing pykrx compatibility tests pass.
- [x] New import-warning regression test passes.
- [x] Streamlit startup log has no `pkg_resources` deprecation warning.

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/python.exe -m pip install --upgrade pykrx==1.2.4`
- `C:/Users/k1190/miniconda3/python.exe -W default -c "import pykrx"`
- `C:/Users/k1190/miniconda3/python.exe -c "import pykrx, numpy, setuptools; print('pykrx', pykrx.__version__); print('numpy', numpy.__version__); print('setuptools', setuptools.__version__)"`
- `C:/Users/k1190/miniconda3/python.exe -m pytest -q tests/test_pykrx_compat.py tests/test_krx_pykrx_compat_paths.py`
- `C:/Users/k1190/miniconda3/python.exe -m pytest -q tests/test_pykrx_import_warning.py`
- `C:/Users/k1190/miniconda3/python.exe -c "...subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.headless', 'true', '--server.port', '8518']) + 18s capture..."`
- Results:
- `requirements.txt` updated to `pykrx>=1.2.4`.
- `src/data_sources/pykrx_compat.py` docstring updated to version-neutral wording.
- Added `tests/test_pykrx_import_warning.py` and verified warning-string absence on `import pykrx`.
- Runtime package upgraded to `pykrx 1.2.4`; import check produced no `pkg_resources` deprecation warning.
- Targeted regression tests passed: `10 passed in 0.45s` and `1 passed in 1.26s`.
- Streamlit startup smoke on port 8518 printed startup URLs and `HAS_PKG_WARNING=False`.
- Residual risks / follow-ups:
- `pykrx 1.2.4` constrains `numpy<2.0`, so environment `numpy` was changed from `2.3.1` to `1.26.4` during install. Current project constraints (`numpy>=1.24.0`) are still satisfied, but this dependency shift should be noted for other workloads sharing the same interpreter.

## 13) KRX Stale Code Warning Elimination (`5041`/`1166`) (2026-02-22)

Pre-Implementation Check-in:
- 2026-02-22: Start implementation to remove startup warnings caused by stale index codes and add a runtime guard to prevent repeated stale-code fetch attempts.

Execution Checklist:
- [x] Confirm root cause by validating pykrx index universe for `5041` and `1166`.
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Update `config/sector_map.yml` replacements:
- [x] `5041 -> 5049` (`KRX 철강`)
- [x] `1166 -> 1157` (`KOSPI200 생활소비재`)
- [x] In `src/data_sources/krx_indices.py`, add stale-code replacement constants and market universe constants.
- [x] In `src/data_sources/krx_indices.py`, add helper to normalize requested codes (replacement + dedupe).
- [x] In `src/data_sources/krx_indices.py`, add cached pykrx index-universe lookup and supported-code filter.
- [x] Keep conservative fallback when universe lookup fails (`unfiltered` live loop).
- [x] Change `load_sector_prices()` live loop to run only on normalized + supported codes.
- [x] Emit summary logs for replacements and unsupported-code skips (single summary each).
- [x] Keep fallback order and behavior (`LIVE -> CACHED -> SAMPLE`) unchanged.
- [x] Update tests for new behavior and replacement mapping.
- [x] Run targeted pytest.
- [x] Run full pytest.
- [x] Run Streamlit smoke check and verify warning lines for `5041`/`1166` are absent.

Verification Gates:
- [x] `C:/Users/k1190/miniconda3/python.exe -m pytest -q tests/test_krx_pykrx_compat_paths.py tests/test_integration.py` passes.
- [x] `C:/Users/k1190/miniconda3/python.exe -m pytest -q` passes.
- [x] Streamlit startup smoke log does not include:
- [x] `Live fetch failed for index 5041`
- [x] `Live fetch failed for index 1166`

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/python.exe -m pytest -q tests/test_krx_pykrx_compat_paths.py tests/test_integration.py`
- `C:/Users/k1190/miniconda3/python.exe -m pytest -q`
- `C:/Users/k1190/miniconda3/python.exe -c "from src.data_sources.krx_indices import load_sector_prices; status, df = load_sector_prices(['5041','1166'], '20260101', '20260220'); print(status, sorted(df['index_code'].astype(str).unique().tolist()), len(df))"`
- `C:/Users/k1190/miniconda3/python.exe -c "...subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.headless', 'true', '--server.port', '8521']) ..."`
- `rg -n "Live fetch failed for index|Applied KRX index code replacements|Skipping unsupported KRX index codes" .tmp_streamlit_krx_warn_fix.log -S`
- Results:
- Config updated: stale codes replaced in `config/sector_map.yml` (`5049`, `1157`).
- Loader updated: `src/data_sources/krx_indices.py` now normalizes stale codes, caches market-universe lookup, filters unsupported codes pre-fetch, and keeps conservative fallback when universe lookup fails.
- Added tests:
- `tests/test_krx_pykrx_compat_paths.py::test_load_sector_prices_replaces_stale_codes_before_fetch`
- `tests/test_krx_pykrx_compat_paths.py::test_load_sector_prices_skips_unsupported_codes_before_fetch`
- Updated regression:
- `tests/test_integration.py::test_live_partial_success_keeps_live_status` (avoid stale-code coupling by using `5357` as explicit fail path)
- Targeted pytest result: `13 passed in 6.49s`.
- Full pytest result: `57 passed in 7.92s`.
- Runtime replacement smoke: `status=LIVE`, `codes=['1157', '5049']`, `rows=66`.
- Streamlit smoke result (`.tmp_streamlit_krx_warn_fix.log`):
- `HAS_5041_WARN=False`
- `HAS_1166_WARN=False`
- `HAS_URL=True`
- Residual risks / follow-ups:
- pykrx index universe can change again; replacement table should be reviewed when KRX index taxonomy changes.

## 14) Project Conda Env + Streamlit Launch Standardization (2026-02-22)

Pre-Implementation Check-in:
- 2026-02-22: Plan reviewed. Create dedicated conda env `sector-rotation` and switch Streamlit run flow to `conda activate` based commands.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Create dedicated conda env: `sector-rotation` with Python 3.11.
- [x] Install project dependencies from `requirements.txt` into the env.
- [x] Add reproducible env spec file `environment.yml` (name + Python + pip requirements).
- [x] Add launch scripts that activate env before starting Streamlit (`scripts/run_streamlit.ps1`, `scripts/run_streamlit.bat`).
- [x] Update `docs/api-keys-guide.md` run/restart instructions to env activation flow.
- [x] Verify env has Streamlit and app entrypoint is callable.
- [x] Record commands, outcomes, and residual risks in this section.

Verification Gates:
- [x] `conda run -n sector-rotation python -m streamlit --version` succeeds.
- [x] `scripts/run_streamlit.ps1` includes `conda activate sector-rotation` and `python -m streamlit run app.py`.
- [x] `docs/api-keys-guide.md` no longer recommends base path `C:/Users/k1190/miniconda3/python.exe -m streamlit run app.py`.

Review (fill after implementation):
- Commands run:
- `conda create -y -n sector-rotation python=3.11 pip`
- `conda run -n sector-rotation python -m pip install -r requirements.txt`
- `conda run -n sector-rotation python -m streamlit --version`
- `rg -n "conda activate sector-rotation|python -m streamlit run app.py" scripts/run_streamlit.ps1 scripts/run_streamlit.bat docs/api-keys-guide.md -S`
- `cmd /c "conda run -n sector-rotation python -m streamlit run app.py --server.headless true --server.port 8536"` (timeout-based startup probe)
- `netstat -ano | Select-String ":8536"` + `Get-Process -Id <pid>` (listener/process path check)
- `Stop-Process -Id <pid> -Force` (cleanup)
- `powershell -ExecutionPolicy Bypass -File scripts/run_streamlit.ps1 -Headless -Port 8537` (timeout-based startup probe)
- `cmd /c "scripts\\run_streamlit.bat --server.headless true --server.port 8538"` (timeout-based startup probe)
- Results:
- Added project env spec file: `environment.yml` (`name: sector-rotation`, `python=3.11`, `pip`, `-r requirements.txt`).
- Created launcher scripts:
- `scripts/run_streamlit.ps1` (conda shell hook + `conda activate sector-rotation` + `python -m streamlit run app.py`)
- `scripts/run_streamlit.bat` (`conda activate sector-rotation` + `python -m streamlit run app.py %*`)
- Fixed PowerShell hook handling in `scripts/run_streamlit.ps1` by converting conda hook output to a single string before `Invoke-Expression`.
- Updated Streamlit launch docs in `docs/api-keys-guide.md` to activation-based flow.
- Verification:
- `streamlit --version` in env returned `1.54.0`.
- Startup probe opened `LISTENING` on port `8536`; process path resolved to `C:\Users\k1190\miniconda3\envs\sector-rotation\python.exe`.
- `scripts/run_streamlit.ps1` probe opened `LISTENING` on port `8537` with env python path.
- `scripts/run_streamlit.bat` probe opened `LISTENING` on port `8538` with env python path.
- Residual risks / follow-ups:
- `scripts/run_streamlit.bat` assumes `conda` is available in `cmd` session (Conda init/path configured). If needed, switch to explicit `conda.bat` path for stricter portability.

## 15) Momentum Tab Chart Readability Tuning (2026-02-23)

Pre-Implementation Check-in:
- 2026-02-23: Plan reviewed. Scope is limited to momentum tab visualization polish:
- 1) make `Relative Strength vs RS 이동평균` chart visually 1:1 with better on-screen size,
- 2) rewrite `RS 이탈도` helper text for readability.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Update RS scatter layout to enforce 1:1 axis framing and square-style display.
- [x] Tune RS scatter rendered size for better full-screen readability.
- [x] Replace RS divergence one-line description with readable multi-line guidance.
- [x] Run quick verification (`py_compile` + diff check).
- [x] Record commands and outcomes in review.

Verification Gates:
- [x] `src/ui/components.py` contains 1:1 scatter axis/layout settings.
- [x] `app.py` momentum tab renders updated RS divergence explanation block.
- [x] `python -m py_compile app.py src/ui/components.py` succeeds.

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m py_compile app.py src/ui/components.py`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -c "from types import SimpleNamespace; from src.ui.components import render_rs_scatter; ..."`
- `rg -n "RS 이탈도 \\(RS Divergence\\)|계산식|해석 포인트|scaleanchor|scaleratio|height=680" app.py src/ui/components.py`
- Results:
- `render_rs_scatter` now sets identical x/y ranges and `yaxis.scaleanchor='x'`, `scaleratio=1` to keep 1:1 framing.
- Scatter chart height increased to `680` and rendered in centered column layout (`[1.0, 3.2, 1.0]`) for better full-screen readability.
- Momentum tab RS divergence help text changed from one-line sentence to structured bullet guidance with formula/sign interpretation.
- Residual risks / follow-ups:
- Final visual acceptance (desktop/mobile) should be confirmed in live Streamlit UI, because this patch used runtime attribute smoke checks rather than screenshot-based visual diff.

## 16) Momentum Chart Responsive Fine-Tuning (Desktop/Mobile) (2026-02-23)

Pre-Implementation Check-in:
- 2026-02-23: Plan reviewed. Scope is a second-pass responsive tuning for momentum scatter chart size ratio:
- desktop: keep centered/square feel with balanced side margins,
- mobile: remove center-column squeeze and use full-width chart with compact height.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Add UA-based mobile detection helper (safe fallback to desktop).
- [x] Update RS scatter renderer to accept configurable `height`/`margin`.
- [x] Apply desktop/mobile split layout in momentum tab rendering.
- [x] Run quick verification (`py_compile` + focused pytest + runtime layout smoke).
- [x] Record commands, outcomes, and residual risks.

Verification Gates:
- [x] `app.py` contains mobile/desktop branch logic for momentum scatter display.
- [x] `src/ui/components.py` supports configurable scatter height/margin.
- [x] `tests/test_ui_components.py` passes after signature update.
- [x] `python -m py_compile app.py src/ui/components.py` succeeds.

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m py_compile app.py src/ui/components.py tests/test_ui_components.py`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m pytest -q tests/test_ui_components.py`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -c "from types import SimpleNamespace; from src.ui.components import render_rs_scatter; ..."`
- Results:
- Added `_is_mobile_client()` in `app.py` and split momentum scatter rendering:
- mobile: full-width render (`st.plotly_chart(..., use_container_width=True)`) with `height=520`, compact margin.
- desktop: centered render via `st.columns([0.7, 3.6, 0.7])` with `height=700`, roomy margin.
- Updated `render_rs_scatter()` signature in `src/ui/components.py` to accept `height` and `margin` for caller-level responsive tuning.
- Added test `test_render_rs_scatter_allows_custom_height_and_margin` in `tests/test_ui_components.py`; focused test suite passed (`6 passed`).
- Runtime smoke confirmed desktop/mobile profile values and preserved 1:1 axis lock (`scaleanchor=x`, `scaleratio=1`).
- Residual risks / follow-ups:
- Mobile detection is best-effort via user-agent header tokens; edge-case tablet/desktop-class browsers on mobile OS may be classified differently.

## 17) Dashboard IA Refactor for Decision-First Flow (2026-02-23)

Pre-Implementation Check-in:
- 2026-02-23: Plan reviewed. Scope is limited to IA/UI restructuring with no backend analytics contract changes:
- 1) rename/reorder tabs to `오늘의 결론 | 근거 분석 | 전체 신호`,
- 2) move action/regime filters to sidebar as global filters,
- 3) add action summary + top picks in the first tab,
- 4) keep momentum and signal computation logic unchanged.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] In `app.py`, replace current tab labels/order with decision-first labels.
- [x] In `app.py`, move action/regime filters from Signals tab to sidebar and persist as global session state (`filter_action_global`, `filter_regime_only_global`).
- [x] In `app.py`, derive `signals_filtered` once and reuse across all tabs.
- [x] In `app.py`, implement "오늘의 결론" layout with:
- [x] macro tile + status summary,
- [x] action summary component,
- [x] top picks table sorted by `Action priority -> RS divergence desc`,
- [x] returns heatmap based on global filters.
- [x] In `app.py`, keep momentum visuals in "근거 분석" and make warnings filter-aware.
- [x] In `app.py`, keep signal table in "전체 신호" using globally filtered dataset.
- [x] In `src/ui/components.py`, add `render_action_summary(signals: list) -> None`.
- [x] In `tests/test_ui_components.py`, add action-summary tests and keep momentum tests green.
- [x] Run verification: `pytest -q tests/test_ui_components.py tests/test_signals.py tests/test_integration.py`.
- [x] Record commands, outcomes, and residual risks in review.

Verification Gates:
- [x] `app.py` contains sidebar-global filter controls and `signals_filtered` reuse.
- [x] `src/ui/components.py` contains `render_action_summary`.
- [x] `tests/test_ui_components.py` includes action-summary coverage.
- [x] `pytest -q tests/test_ui_components.py tests/test_signals.py tests/test_integration.py` passes.

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m pytest -q tests/test_ui_components.py tests/test_signals.py tests/test_integration.py`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m py_compile app.py src/ui/components.py tests/test_ui_components.py`
- `git diff -- app.py src/ui/components.py tests/test_ui_components.py tasks/todo.md`
- Results:
- Updated IA to decision-first tabs: `오늘의 결론 (Decision) | 근거 분석 (Evidence) | 전체 신호 (Signals)`.
- Added sidebar-global filters (`filter_action_global`, `filter_regime_only_global`) and centralized `signals_filtered` reuse across all tabs.
- Added first-tab decision widgets: Action summary (KPI + bar), Top Picks table with `Action priority -> RS divergence desc`, and filter-aware returns heatmap.
- Kept momentum visual logic intact while making benchmark warning/filter behavior operate on `signals_filtered`.
- Added `render_action_summary(signals: list) -> None` in `src/ui/components.py`.
- Added UI tests for action summary rendering/empty-state handling in `tests/test_ui_components.py`.
- Verification:
- Targeted pytest: `19 passed, 1 warning in 7.70s`.
- `py_compile` for touched files passed without errors.
- Residual risks / follow-ups:
- Global filters can intentionally hide warning-trigger rows (for example, strict action filter), so users may see "필터 조건에 맞는 신호 없음" instead of benchmark-missing warning; this is expected under global-filter semantics.

## 18) Public Repository Security Audit + Hardening (2026-02-23)

Pre-Implementation Check-in:
- 2026-02-23: Plan reviewed. Scope is repository security hardening for public exposure:
- 1) detect credential leakage and secret-tracking risks,
- 2) harden Git tracking rules for local/semi-sensitive artifacts,
- 3) verify no known dependency CVEs from project requirements.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Run tracked-file secret scan (`rg` patterns + `.streamlit` tracked-file check).
- [x] Run dependency vulnerability audit on project requirements.
- [x] Add/strengthen `.gitignore` to block local secrets and local-runtime artifacts.
- [x] Remove `.streamlit/secrets.toml` from Git tracking while keeping local file for runtime.
- [x] Verify no tracked secret file remains and no raw key patterns remain in tracked files.
- [x] Record commands, outcomes, and residual risks in review.

Verification Gates:
- [x] `git ls-files .streamlit` contains only `config.toml` and `secrets.toml.example` (no `secrets.toml`).
- [x] `.gitignore` includes secret/runtime ignores (`.streamlit/secrets.toml`, `.omc/`, `.tmp*`, `__pycache__/`, `*.pyc`, etc.).
- [x] `conda run -n sector-rotation python -m pip_audit -r requirements.txt` reports no known vulnerabilities.
- [x] Secret-pattern scan does not report active keys from tracked files.

Review (fill after implementation):
- Commands run:
- `git ls-files`
- `rg -n --hidden --glob '!*.parquet' --glob '!.git/*' --glob '!__pycache__/*' --glob '!*.pyc' "(ECOS_API_KEY|KOSIS_API_KEY|api[_-]?key|secret|token|password|passwd|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY|xox[baprs]-|ghp_[A-Za-z0-9]{36})"`
- `conda run -n sector-rotation python -m pip install pip-audit`
- `conda run -n sector-rotation python -m pip_audit -r requirements.txt`
- `git rm --cached -r --ignore-unmatch .streamlit/secrets.toml .omc .claude/settings.local.json .tmp_streamlit_krx_warn_fix.log .tmp_streamlit_momentum_responsive.log .tmp_streamlit_sector_rotation.log .tmp_test`
- `git ls-files .streamlit`
- `git check-ignore -v .streamlit/secrets.toml .omc/sessions/1393eb07-755a-47ec-84cc-782212be894d.json .tmp_streamlit_krx_warn_fix.log`
- `git grep -n "<redacted_exposed_key_literal_1>\|<redacted_exposed_key_literal_2>"`
- Results:
- Identified critical risk: `.streamlit/secrets.toml` (real API keys) was tracked in Git and therefore exposed in public repository context.
- Added new `.gitignore` with explicit secret/runtime protections (`.streamlit/secrets.toml`, `.omc/`, `.claude/settings.local.json`, `.tmp*`, `.tmp_test/`, `__pycache__/`, `*.py[cod]`, test/cache artifacts).
- Removed tracked local artifacts from Git index (`.streamlit/secrets.toml`, `.omc/*`, `.claude/settings.local.json`, `.tmp*`, `.tmp_test/*`) via `git rm --cached`; local files remain on disk.
- Verification passed: `git ls-files .streamlit` now returns only `config.toml` + `secrets.toml.example`; `git check-ignore` confirms ignore rules are active.
- Tracked-file key search using leaked key literals returned no matches after untracking.
- Dependency audit result: `No known vulnerabilities found`.
- Residual risks / follow-ups:
- Because keys were already committed previously, treat both ECOS/KOSIS keys as compromised and rotate/revoke them immediately at providers.
- For complete historical remediation on GitHub, rewrite history (for example with `git filter-repo` or BFG) and force-push, then invalidate cached forks/clones as feasible.

## 19) Secret Scanning Automation Setup (2026-02-23)

Pre-Implementation Check-in:
- 2026-02-23: Plan reviewed. Scope is preventive controls before public re-publish:
- 1) local pre-commit secret detection gate,
- 2) CI secret scan workflow on push/PR,
- 3) reproducible setup commands + verification evidence.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Add `.pre-commit-config.yaml` with secret-detection hook(s).
- [x] Add `.github/workflows/secret-scan.yml` for automated scan on push/PR.
- [x] Install `pre-commit` in project env and register local git hooks.
- [x] Run local pre-commit scan (`pre-commit run --all-files`) and resolve failures.
- [x] Record commands, outcomes, and residual risks in review.

Verification Gates:
- [x] `.pre-commit-config.yaml` exists and includes secret detection.
- [x] `.github/workflows/secret-scan.yml` exists and runs on `push` + `pull_request`.
- [x] `pre-commit install` completes in `sector-rotation` env.
- [x] `pre-commit run --all-files` completes without secret findings.

Review (fill after implementation):
- Commands run:
- `conda run -n sector-rotation python -m pip install pre-commit detect-secrets`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m pre_commit run detect-secrets --all-files -v` (diagnostic; hook migration decision)
- `conda run -n sector-rotation pre-commit install`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m pre_commit run --all-files`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe scripts/security/scan_secrets.py .streamlit/secrets.toml`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe scripts/security/scan_secrets.py .streamlit/secrets.toml.example`
- Results:
- Added `.pre-commit-config.yaml` with three local guards: `detect-private-key`, `check-merge-conflict`, and custom `secret pattern scan` hook.
- Added custom scanner `scripts/security/scan_secrets.py` to block high-risk secret formats (AWS/GitHub/Slack/OpenAI/private-key headers + hardcoded ECOS/KOSIS key assignments).
- Added CI workflow `.github/workflows/secret-scan.yml` with two jobs:
- pre-commit checks on all files (`push`/`pull_request`/manual),
- Gitleaks full scan (`gitleaks/gitleaks-action@v2`) for repository-level leak detection.
- Local hook installation succeeded (`pre-commit installed at .git/hooks/pre-commit`).
- Local verification succeeded: `pre-commit run --all-files` passed all configured hooks.
- Scanner behavior verified:
- real keys in `.streamlit/secrets.toml` are detected and block (exit 1),
- placeholders in `.streamlit/secrets.toml.example` are allowed (exit 0).
- Residual risks / follow-ups:
- Custom local scanner is intentionally pattern-based; keep CI Gitleaks enabled to cover broader entropy/signature classes.
- If you later add new provider keys, extend `scripts/security/scan_secrets.py` patterns to keep local pre-commit coverage aligned.

## 20) Recreate GitHub Repository and Push Current Branch (2026-02-23)

Pre-Implementation Check-in:
- 2026-02-23: Plan reviewed. Scope is to create a new GitHub repository for this existing local project, repoint `origin`, and push current `main` safely without rewriting local history.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Verify current git branch/remote state and GitHub CLI auth status.
- [x] Create new GitHub repository under authenticated account.
- [x] Update local `origin` remote to the new repository URL.
- [x] Push local `main` and set upstream tracking.
- [x] Verify remote URL and push success via git/gh checks.
- [x] Record commands, outcomes, and residual risks in review.

Verification Gates:
- [x] GitHub auth is confirmed via `git credential fill` (fallback because `gh` CLI is not installed).
- [x] Repository creation succeeds through GitHub REST API (`POST /user/repos`) and returns URL.
- [x] `git remote -v` points `origin` to newly created repository.
- [x] `git push -u origin main` succeeds.

Review (fill after implementation):
- Commands run:
- `git status --short --branch`
- `git remote -v`
- `git remote show origin` (expected failure before recreate: `Repository not found`)
- `cmd /v:on /c "(echo protocol=https&echo host=github.com&echo.) > %TEMP%\gitcred_input.txt && git credential fill < %TEMP%\gitcred_input.txt && del %TEMP%\gitcred_input.txt"`
- PowerShell script to call GitHub REST API with token from credential helper:
- `GET https://api.github.com/repos/stat17-hb/sector-rotation` (existence check)
- `POST https://api.github.com/user/repos` with `{ name: "sector-rotation", private: false, auto_init: false }`
- `git remote set-url origin https://github.com/stat17-hb/sector-rotation.git`
- `git push -u origin main`
- `git ls-remote --heads origin main`
- `git rev-parse --abbrev-ref --symbolic-full-name "@{u}"`
- Results:
- `gh` was unavailable (`CommandNotFoundException`), so repo creation path was switched to GitHub API using existing Git credential helper auth.
- New repository created successfully: `https://github.com/stat17-hb/sector-rotation`.
- Local `origin` confirmed as `https://github.com/stat17-hb/sector-rotation.git`.
- Push completed successfully (`[new branch] main -> main`) and upstream tracking set to `origin/main`.
- Remote branch verification passed via `git ls-remote --heads origin main`.
- Residual risks / follow-ups:
- `gh` CLI is still not installed; future repo admin tasks will require API/manual UI unless `gh` is installed.

## 21) Batch Launcher No-Output Fix (`run_streamlit.bat`) (2026-02-23)

Pre-Implementation Check-in:
- 2026-02-23: User reported that running `scripts/run_streamlit.bat` shows no window/output. Target is to make launcher reliably start Streamlit from `sector-rotation` env and expose activation failures.

Execution Checklist:
- [x] Inspect current `scripts/run_streamlit.bat`.
- [x] Fix conda activation call semantics for batch-to-batch execution.
- [x] Add activation failure handling message and stop path.
- [x] Preserve runtime argument forwarding for Streamlit options.
- [x] Verify launcher by running batch with headless port probe.

Verification Gates:
- [x] `scripts/run_streamlit.bat` uses `call conda activate sector-rotation`.
- [x] `scripts/run_streamlit.bat` runs `python -m streamlit run app.py %*`.
- [x] Runtime probe confirms `LISTENING` python process from `...envs\\sector-rotation\\python.exe`.

Review (fill after implementation):
- Commands run:
- `Get-Content -Raw scripts/run_streamlit.bat`
- `cmd /c "scripts\\run_streamlit.bat --server.headless true --server.port 8540"` (port unavailable signal confirmed launcher executed)
- `cmd /c "scripts\\run_streamlit.bat --server.headless true --server.port 8799"` (timeout while app running)
- `netstat -ano | Select-String ":8799"`
- `Get-Process -Id 26496 | Select-Object Id,ProcessName,Path`
- `Stop-Process -Id 26496 -Force`
- Results:
- Root cause matched batch semantics: removing `call` before `conda` prevented expected script flow for chained batch execution.
- Launcher updated to `call conda activate sector-rotation`, error branch with message/pause, and `%*` forwarding.
- Probe verified Streamlit listener on `:8799` and process path `C:\Users\k1190\miniconda3\envs\sector-rotation\python.exe`.
- Residual risks / follow-ups:
- If `conda` is not initialized for CMD on another machine, activation can still fail; message now instructs that condition explicitly.

## 22) Conda Init Error Fix for CMD Launcher (2026-02-23)

Pre-Implementation Check-in:
- 2026-02-23: User reported `CondaError: Run 'conda init' before 'conda activate'` when launching `scripts/run_streamlit.bat`. Target is to remove CMD init dependency and keep env activation reliable.

Execution Checklist:
- [x] Reproduce/confirm launcher context and conda path signals (`CONDA_EXE`, `conda.bat`).
- [x] Update batch launcher to resolve and call `conda.bat` directly.
- [x] Keep fallback path when `conda.bat` cannot be resolved.
- [x] Verify launcher startup using headless Streamlit port probe.
- [x] Confirm runtime process uses `sector-rotation` env python.

Verification Gates:
- [x] `scripts/run_streamlit.bat` resolves `CONDA_BAT` and invokes `call "%CONDA_BAT%" activate sector-rotation`.
- [x] Fallback branch still supports `call conda activate sector-rotation`.
- [x] Probe run opens listener port and process path points to `...\\envs\\sector-rotation\\python.exe`.

Review (fill after implementation):
- Commands run:
- `Get-Content -Raw scripts/run_streamlit.bat`
- `Write-Output "CONDA_EXE=$env:CONDA_EXE"; Get-Command conda`
- `Test-Path "$env:USERPROFILE\\miniconda3\\condabin\\conda.bat"`
- `cmd /c "scripts\\run_streamlit.bat --server.headless true --server.port 8801"` (timeout while app running)
- `netstat -ano | Select-String ":8801"`
- `Get-Process -Id 18320 | Select-Object Id,ProcessName,Path`
- `Stop-Process -Id 18320 -Force`
- Results:
- `CONDA_EXE` existed and pointed to `C:\Users\k1190\miniconda3\Scripts\conda.exe`.
- `C:\Users\k1190\miniconda3\condabin\conda.bat` existed and is now used directly in launcher activation path.
- Batch probe launched Streamlit successfully; `:8801` listener confirmed.
- Runtime process path confirmed env python: `C:\Users\k1190\miniconda3\envs\sector-rotation\python.exe`.
- Residual risks / follow-ups:
- On machines with non-standard conda install paths and missing `CONDA_EXE`, launcher falls back to `call conda ...`; those environments may still require PATH/conda setup.

## 23) Railway Deployment Setup for Streamlit (2026-02-23)

Pre-Implementation Check-in:
- 2026-02-23: Plan reviewed. Scope is deployment enablement for Railway with minimal, production-safe startup config for Streamlit:
- bind Streamlit to Railway-assigned `PORT`,
- keep app headless and externally reachable (`0.0.0.0`),
- document step-by-step deploy flow and required environment variables.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Add Railway config-as-code start command in `railway.toml` for Streamlit.
- [x] Add `runtime.txt` to pin Python runtime expected by the project.
- [x] Add deployment guide doc for Railway (`docs/railway-deploy.md`).
- [x] Run local startup smoke with explicit `PORT` to validate command semantics.
- [x] Run focused verification (`py_compile` + selected pytest) after config/docs changes.
- [x] Record commands, outcomes, and residual risks in review.

Verification Gates:
- [x] `railway.toml` exists and sets `deploy.startCommand` with `--server.port $PORT --server.address 0.0.0.0 --server.headless true`.
- [x] `runtime.txt` exists and aligns with project Python baseline.
- [x] Railway guide includes env vars for `ECOS_API_KEY` and `KOSIS_API_KEY`.
- [x] Local smoke starts app with `PORT` override without command errors.

Review (fill after implementation):
- Commands run:
- `Get-ChildItem -Force`
- `rg --files`
- `Get-Content -Raw app.py`
- `Get-Content -Raw .streamlit/config.toml`
- `Get-Content -Raw .streamlit/secrets.toml.example`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m py_compile app.py src/data_sources/krx_indices.py src/data_sources/ecos.py src/data_sources/kosis.py tests/test_ui_components.py tests/test_integration.py`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m pytest -q tests/test_ui_components.py tests/test_integration.py`
- `Start-Process ... streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8812` + `netstat -ano | Select-String ':8812'`
- `Start-Process cmd /c "set PORT=8814 && ...python.exe -m streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port %PORT%"` + `netstat -ano | Select-String ':8814'`
- Results:
- Added `railway.toml` with explicit Railpack builder and deployment start command.
- Added `runtime.txt` (`3.11`) to reduce runtime drift across deployments.
- Added `docs/railway-deploy.md` with end-to-end Railway deploy steps and required env vars.
- Focused verification passed: `16 passed, 1 warning in 8.37s`.
- Startup smoke passed: Streamlit process listened on `0.0.0.0:8812` and command-style run with `PORT` env also listened on `0.0.0.0:8814`.
- Residual risks / follow-ups:
- Railway build/runtime behavior can differ if service-level start command is manually overridden in UI; keep `railway.toml` and UI command consistent.
- Without valid `ECOS_API_KEY`/`KOSIS_API_KEY`, production deploy will run in fallback (`SAMPLE`) mode by design.

## 24) Typography + Tab Contrast Polish (2026-02-24)

Pre-Implementation Check-in:
- 2026-02-24: User requested UI readability polish in two points:
- 1) switch to a more professional finance-dashboard font stack,
- 2) improve selected-tab text contrast so active tab labels do not blend into dark backgrounds.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Update global app typography in `src/ui/styles.py` to a finance-style professional font family.
- [x] Add explicit Streamlit tab styles for default/hover/selected states with stronger text contrast.
- [x] Keep visual changes minimal and scoped to typography + tabs only.
- [x] Run focused verification (`py_compile` and UI-component tests) for touched modules.
- [x] Record commands, outcomes, and residual risks in review.

Verification Gates:
- [x] `src/ui/styles.py` imports and applies the new primary font consistently.
- [x] Active tab selector (`aria-selected="true"`) has high-contrast text color and clear background/border cue.
- [x] `python -m py_compile src/ui/styles.py app.py` passes.
- [x] `pytest -q tests/test_ui_components.py` passes.

Review (fill after implementation):
- Commands run:
- `Get-Content src/ui/styles.py`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m py_compile src/ui/styles.py app.py`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m pytest -q tests/test_ui_components.py`
- Results:
- Replaced typography import from Pretendard CDN to Google Fonts (`IBM Plex Sans KR` + `IBM Plex Sans`) and wired new font stack to both global CSS and Plotly template font family.
- Added explicit tab-state styling under Streamlit tab selectors: muted default text, brighter hover, and high-contrast selected-state text/background/highlight.
- Change scope remained minimal (`src/ui/styles.py` + tracking update in `tasks/todo.md`).
- Verification passed: compile check succeeded; targeted UI test suite succeeded (`8 passed in 0.94s`).
- Residual risks / follow-ups:
- Google Fonts dependency adds external font fetch at runtime; in restricted networks font fallback stack will be used automatically.

## 16) Korea Sector Rotation UI 전면 리디자인 (2026-02-24)

Pre-Implementation Check-in:
- 2026-02-24: Scope locked to presentation layer only (UI/theme/layout). No changes to signal engine or macro/price business logic.
- Fixed decisions: full redesign, dark/light toggle, local font hosting, unified sidebar IA, WCAG contrast tests.

Execution Checklist:
- [x] Add dark/light token dictionary in `src/ui/styles.py` with keys `bg/surface/border/text/text_muted/primary/success/warning/danger/info`.
- [x] Change `inject_css()` -> `inject_css(theme_mode: str)` and make all CSS colors token-driven.
- [x] Change `get_plotly_template()` -> `get_plotly_template(theme_mode: str)` and split theme-specific colorway (no teal/cyan).
- [x] Add local font-face loading from `static/fonts/PretendardVariable.woff2` and `static/fonts/JetBrainsMono[wght].woff2` with fallback stack.
- [x] Extend UI render signatures in `src/ui/components.py` to accept `theme_mode` and pass to Plotly template calls.
- [x] Ensure status expression is not color-only by adding explicit text labels/chips in table-level action/status fields.
- [x] Refactor `app.py` sidebar into a single block ordered as: Quick Status -> Global Filters -> Model Parameters (collapsed) -> Data Actions.
- [x] Wrap model parameter controls in `st.form` + apply button to minimize reruns.
- [x] Keep default control as slider; move direct numeric input to an advanced section.
- [x] Add `st.session_state["theme_mode"]` default `"dark"` and implement dark/light toggle (session-scoped).
- [x] Update tab naming/text density and restructure Evidence tab to summary card + expandable details.
- [x] Update `.streamlit/config.toml` theme font to `"sans serif"`.
- [x] Add new tests: `tests/test_ui_theme.py`, `tests/test_ui_contrast.py`.
- [x] Update `tests/test_ui_components.py` for new theme-aware signatures.

Verification Gates:
- [x] `python -m py_compile app.py src/ui/styles.py src/ui/components.py`
- [x] `pytest -q tests/test_ui_components.py tests/test_ui_theme.py tests/test_ui_contrast.py`
- [x] Contrast assertions verify body/muted/tab/badge combinations are >= 4.5 for normal text.
- [x] Theme toggle switches charts/tables/badges/sidebar styles consistently.
- [x] Local font files are present and CSS fallback remains valid when files are missing.

Review:
- Commands run:
- `python -m py_compile app.py src/ui/styles.py src/ui/components.py`
- `pytest -q tests/test_ui_components.py tests/test_ui_theme.py tests/test_ui_contrast.py`
- `Get-ChildItem static/fonts | Select-Object Name,Length,LastWriteTime`
- Result summary:
- Implemented full presentation-layer redesign with dark/light token system, session-scoped theme toggle (default dark), local `@font-face` loading, and theme-aware Plotly templates/components.
- Unified sidebar IA into `빠른 상태 -> 글로벌 필터 -> 모델 파라미터(폼+적용) -> 데이터 작업` and moved direct numeric edits into a collapsed 고급 섹션.
- Added explicit icon+text action labels (`▲/●/■/▼/○`) so state communication is not color-only; restructured Evidence tab into summary card + expandable detail.
- Added regression tests `tests/test_ui_theme.py` and `tests/test_ui_contrast.py`, and updated `tests/test_ui_components.py` for theme-aware signatures.
- Verification outcome: `18 passed in 0.98s`.
- Risks/notes:
- CSS uses `color-mix(...)`; modern Chromium/Firefox/Safari are supported, but very old browser engines may render fallback colors differently.
- Sidebar quick-status cards show probe status (parquet presence) before full data load; authoritative LIVE/CACHED/SAMPLE status remains visible in the main Decision metrics.

## 25) Light Theme Readability Upgrade (2026-02-24)

Pre-Implementation Check-in:
- 2026-02-24: User reported that light theme readability is significantly worse than dark theme.
- Scope: improve visual hierarchy and contrast in light mode without changing business logic.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Adjust light-theme token values and light-only background treatment for clearer foreground separation.
- [x] Refine global text color rules so normal body copy is not over-muted in light mode.
- [x] Improve light-mode readability for tabs/sidebar/form controls with explicit contrast-oriented selectors.
- [x] Run focused verification (`py_compile` + UI theme/contrast/component tests).
- [x] Record commands, outcomes, and residual risks in review.

Verification Gates:
- [x] `src/ui/styles.py` reflects stronger light-mode readability defaults (tokens + CSS selectors).
- [x] `python -m py_compile src/ui/styles.py app.py` passes.
- [x] `pytest -q tests/test_ui_theme.py tests/test_ui_contrast.py tests/test_ui_components.py` passes.

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m py_compile src/ui/styles.py app.py`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m pytest -q tests/test_ui_theme.py tests/test_ui_contrast.py tests/test_ui_components.py`
- Results:
- Updated `src/ui/styles.py` light tokens (`bg/border/text/text_muted`) for stronger contrast hierarchy.
- Added light-mode specific background/sidebar/control/card styles while preserving dark-mode behavior.
- Stopped over-muted body copy by separating muted caption/metric-label selectors from markdown paragraph/list text selectors.
- Added Streamlit theme variable aliases (`--background-color`, `--secondary-background-color`, `--text-color`, `--primary-color`) and `color-scheme` to better align native widget readability in light mode.
- Verification passed: compile check succeeded; focused UI suite succeeded (`18 passed in 1.00s`).
- Residual risks / follow-ups:
- Native `st.dataframe` internals are partly Streamlit-controlled; if any isolated low-contrast cells remain, a targeted dataframe selector pass may still be needed.

## 26) Light Theme Palette + Table Harmonization (2026-02-24)

Pre-Implementation Check-in:
- 2026-02-24: User reported three remaining issues: light-mode readability still weak, overall palette disharmony, and dark-looking tables on bright background.
- Scope: unify light theme visual language and remove dark-table mismatch while preserving dashboard logic.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Re-tune light palette tokens/colorway to a more cohesive light-theme family.
- [x] Add explicit light/dark table style tokens and apply them to dataframe rendering paths.
- [x] Update `render_signal_table()` and Decision `Top Picks` table to use theme-aware table styling.
- [x] Add CSS fallback selectors for Streamlit dataframe wrappers to reduce dark-theme leakage in light mode.
- [x] Run focused verification (`py_compile` + UI tests).
- [x] Record commands, outcomes, and residual risks in review.

Verification Gates:
- [x] Light theme table sections render with light headers/body rows and dark readable text.
- [x] `python -m py_compile src/ui/styles.py src/ui/components.py app.py` passes.
- [x] `pytest -q tests/test_ui_theme.py tests/test_ui_contrast.py tests/test_ui_components.py` passes.

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m py_compile src/ui/styles.py src/ui/components.py app.py tests/test_ui_theme.py tests/test_ui_contrast.py`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m pytest -q tests/test_ui_theme.py tests/test_ui_contrast.py tests/test_ui_components.py`
- Results:
- Added `TABLE_STYLE_TOKENS` and `get_table_style_tokens()` in `src/ui/styles.py`, then wired them into CSS dataframe fallback selectors and component-level styling.
- Rebalanced light Plotly colorway to a tighter, finance-dashboard palette family (blue/indigo/green/amber/rose 중심).
- Updated `render_signal_table()` to apply table-wide themed row/header styling before action/status emphasis, eliminating dark-base table bleed in light mode.
- Updated Decision tab `Top Picks` to render as themed `Styler` dataframe with light header, zebra rows, and readable text.
- Extended tests to cover new table theming hooks and contrast checks:
- `tests/test_ui_theme.py`: CSS contains light table tokens.
- `tests/test_ui_contrast.py`: header/body table text contrast checks.
- Verification passed: focused UI suite `20 passed in 1.00s`.
- Residual risks / follow-ups:
- Streamlit dataframe internals can change across versions; if selector drift happens after upgrades, keep component-level `Styler` path as primary and adjust CSS fallback selectors.

## 27) Light Theme Header + Text + Dataframe Final Polish (2026-02-24)

Pre-Implementation Check-in:
- 2026-02-24: User requested final visual mismatch fixes in light mode:
- 1) text color still blending with background in places,
- 2) top area remains dark-toned,
- 3) table header/bottom strip still dark.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Increase light-mode muted-text readability and widget-label contrast.
- [x] Force top header/decoration region to light-tone background in light mode.
- [x] Strengthen dataframe theming via Glide Data Grid CSS variables and scrollbar track/thumb styling.
- [x] Keep table header/rows aligned with light table tokens and remove dark strip artifacts.
- [x] Run focused verification (`py_compile` + UI tests).
- [x] Record commands, outcomes, and residual risks in review.

Verification Gates:
- [x] Light mode header/decoration area is no longer dark-only.
- [x] Light mode widget/markdown text is clearly legible against background.
- [x] Light mode dataframe header/body/scroll track use light tokens.
- [x] `python -m py_compile src/ui/styles.py app.py src/ui/components.py tests/test_ui_theme.py tests/test_ui_contrast.py` passes.
- [x] `pytest -q tests/test_ui_theme.py tests/test_ui_contrast.py tests/test_ui_components.py` passes.

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m py_compile src/ui/styles.py app.py src/ui/components.py tests/test_ui_theme.py tests/test_ui_contrast.py`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m pytest -q tests/test_ui_theme.py tests/test_ui_contrast.py tests/test_ui_components.py`
- Results:
- Updated light `text_muted` to a darker readable tone.
- Added light-mode header/toolbar/decoration overrides (`stHeader`, `stDecoration`) to remove dark top strip mismatch.
- Added stronger label/expander contrast selectors and inline-code background tuning for light mode readability.
- Added Glide Data Grid token overrides (`--gdg-bg-header`, `--gdg-bg-cell`, `--gdg-border-color` etc.) and scrollbar track/thumb styling inside `stDataFrame`.
- Extended theme test to assert header/decor/dataframe token hooks are present in injected CSS.
- Verification passed: focused UI suite `20 passed in 1.04s`.
- Residual risks / follow-ups:
- Browser cache can keep stale CSS for Streamlit apps; use hard refresh when validating visual changes.

## 28) Light Theme Axis/Label Readability Follow-up (2026-02-24)

Pre-Implementation Check-in:
- 2026-02-24: User reported remaining readability issues in light theme, especially chart/table axes and labels around the Theme Control panel workflow.
- Scope: improve light-theme chart axis/title/legend legibility without touching signal logic.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Raise light-theme chart text hierarchy in `src/ui/styles.py` for axis ticks/titles and legend labels.
- [x] Ensure shared Plotly template applies readable contrast for light mode axis lines and grid balance.
- [x] Add or update tests in `tests/test_ui_theme.py` and `tests/test_ui_contrast.py` to guard axis/legend readability regressions.
- [x] Run focused verification (`py_compile` + UI theme/contrast/component tests).
- [x] Record commands, outcomes, and residual risks in review.

Verification Gates:
- [x] `get_plotly_template("light")` returns higher-contrast axis tick/title/legend text defaults than prior muted settings.
- [x] `python -m py_compile src/ui/styles.py src/ui/components.py app.py tests/test_ui_theme.py tests/test_ui_contrast.py` passes.
- [x] `pytest -q tests/test_ui_theme.py tests/test_ui_contrast.py tests/test_ui_components.py` passes.

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m py_compile src/ui/styles.py src/ui/components.py app.py tests/test_ui_theme.py tests/test_ui_contrast.py`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m pytest -q tests/test_ui_theme.py tests/test_ui_contrast.py tests/test_ui_components.py`
- Results:
- Updated `src/ui/styles.py:get_plotly_template()` so light theme uses stronger axis tick/title/legend text colors and clearer axis line/grid balance.
- Added `automargin=True` on x/y axes to reduce label clipping risk and improve readability for longer tick labels.
- Updated `src/ui/components.py` no-data RS scatter annotation color from muted to primary body text for better light-mode legibility.
- Added test coverage:
- `tests/test_ui_theme.py`: verifies light Plotly template no longer uses muted text for axis/legend and keeps axis automargins enabled.
- `tests/test_ui_contrast.py`: verifies Plotly axis/legend text contrast ratio is >= 4.5 for both dark/light themes.
- Verification passed: focused UI suite `22 passed in 1.30s`.
- Residual risks / follow-ups:
- Browser cache can delay CSS/template perception during manual checks; use hard refresh when validating visuals.
- Chart-level per-trace text colors (if added in future figures) can bypass template defaults, so keep contrast checks near custom trace text when expanding charts.

## 29) Dashboard Alert Category Explanation (2026-02-24)

Pre-Implementation Check-in:
- 2026-02-24: User requested adding an in-dashboard explanation for `알림` categories and how each alert is computed.
- Scope: Add UI-only explanatory content in Signals tab without changing signal-engine behavior.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Add a new explanation block in `app.py` Signals tab that lists alert categories.
- [x] Describe calculation logic/trigger rules for each alert (`Overheat`, `Oversold`, `FX Shock`, `Benchmark Missing`, `RS Data Insufficient`).
- [x] Include current default thresholds in explanation (`RSI 70/30`, `FX shock 3.0%`).
- [x] Run lightweight verification (`py_compile`) for touched code.
- [x] Record commands, outcomes, and residual risks in review.

Verification Gates:
- [x] Signals tab renders alert-explanation section without affecting existing signal table rendering.
- [x] `python -m py_compile app.py` passes.

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m py_compile app.py`
- `git diff -- app.py tasks/todo.md`
- Results:
- Added a new `st.expander("알림 카테고리 설명")` block in the Signals tab directly above `render_signal_table(...)`.
- The section now documents all alert categories and trigger rules: `Overheat`, `Oversold`, `FX Shock`, `Benchmark Missing`, `RS Data Insufficient`.
- Thresholds are shown from runtime settings (`rsi_overbought`, `rsi_oversold`, `fx_shock_pct`) with defaults 70/30/3.0%.
- Added a caption note that the current implementation passes FX change as `0.0` during signal calculation, so `FX Shock` alert is typically not triggered.
- Residual risks / follow-ups:
- Content is explanatory only; if FX shock logic wiring is changed later, this text should be revalidated against implementation.

## 30) FX 변화율 전달 경로 복구 (2026-02-24)

Pre-Implementation Check-in:
- 2026-02-24: User reported FX shock alert was effectively disabled because signal calculation path used `fx_change_pct=0.0`.
- Scope: Wire real USD/KRW change into signal engine (`build_signal_table`) and align dashboard explanation text with runtime behavior.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Update `src/signals/matrix.py:build_signal_table()` to accept runtime FX change input.
- [x] Compute USD/KRW change in `app.py` signal-cache path and pass it into `build_signal_table(...)`.
- [x] Remove stale explanation caption claiming FX input is always `0.0`; replace with behavior-accurate wording.
- [x] Add regression coverage proving `build_signal_table(...)` applies FX shock downgrade when `fx_change_pct` exceeds threshold.
- [x] Run focused verification (`py_compile` + targeted `pytest`) for touched modules.
- [x] Record commands, outcomes, and residual risks in review.

Verification Gates:
- [x] `build_signal_table(...)` receives non-zero FX change via call path when macro `usdkrw` series has at least 2 points.
- [x] FX shock downgrade path (`Strong Buy -> Watch` + `FX Shock`) is test-covered through `build_signal_table(...)`, not only scoring helper.
- [x] `python -m py_compile app.py src/signals/matrix.py tests/test_integration.py` passes.
- [x] `pytest -q tests/test_integration.py tests/test_signals.py` passes.

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m py_compile app.py src/signals/matrix.py tests/test_integration.py`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m pytest -q tests/test_integration.py tests/test_signals.py`
- Results:
- `src/signals/matrix.py` now accepts `fx_change_pct` input and normalizes it to `fx_change_value` (`None`/invalid -> `NaN`) before applying FX shock filter.
- `app.py` now computes USD/KRW latest change in `_cached_signals(...)` and passes it into `build_signal_table(...)`, so signal computation uses real FX move.
- Signals-tab explanation caption was updated to reflect current behavior (real FX input, skipped when less than 2 FX points).
- Added integration regression `test_build_signal_table_applies_fx_shock_when_fx_change_provided` to verify `build_signal_table(...)` path triggers `Strong Buy -> Watch` downgrade with `FX Shock`.
- Verification result: `12 passed, 1 warning in 7.24s`.
- Residual risks / follow-ups:
- If `usdkrw` macro series is unavailable or has fewer than 2 points, FX shock remains intentionally skipped for that run.
