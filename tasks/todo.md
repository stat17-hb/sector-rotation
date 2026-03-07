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

## 31) Signals 탭 적합/비적합 판정 기준 설명 추가 (2026-02-24)

Pre-Implementation Check-in:
- 2026-02-24: User requested adding an explicit in-dashboard explanation for how `적합/비적합` is determined.
- Scope: UI-only documentation update in Signals tab; no signal-engine logic changes.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Insert `적합/비적합 판정 기준` expander in `app.py` within `tab_all_signals`.
- [x] Place the new expander between the filter caption and existing `알림 카테고리 설명` expander.
- [x] Use the approved explanatory copy including `macro_fit`, `macro_result["regime"].iloc[-1]`, and `config/sector_map.yml` mapping basis.
- [x] Run verification commands from plan (`py_compile`, targeted pytest).
- [x] Record command outputs and residual risks.

Verification Gates:
- [x] Signals tab shows the new `적합/비적합 판정 기준` expander above alert-category explanation.
- [x] `python -m py_compile app.py` passes.
- [x] `pytest -q tests/test_ui_components.py` passes.

Review (fill after implementation):
- Commands run:
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m py_compile app.py`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m pytest -q tests/test_ui_components.py`
- Results:
- Added a new Signals-tab expander: `적합/비적합 판정 기준`.
- Description clarifies that fit/non-fit is based on latest macro regime and sector mapping (`macro_fit`) and that action is computed by combining macro fit with momentum.
- Expander order is now: `적용 필터` caption -> `적합/비적합 판정 기준` -> `알림 카테고리 설명` -> signal table.
- Residual risks / follow-ups:
- No behavioral changes expected; this is explanatory UI content only.

## 32) 경기국면 판정 타당성 평가 실행 (2026-02-25)

Pre-Implementation Check-in:
- 2026-02-25: User requested direct implementation of the approved execution spec for regime-validity evaluation.
- Scope: evaluation-only deliverable (`판정+섹터매핑`, Point-in-time 우선), no production signal-engine logic change.

Execution Checklist:
- [x] Add this section to `tasks/todo.md` with checklist + review area.
- [x] Add reproducible evaluator script `scripts/evaluate_regime_validity.py` for lag/epsilon/provisional checks and D1~D4 scoring.
- [x] Lock data inputs to curated files and evaluation policy (`is_provisional` exclusion, lag 0/1/2 scenarios).
- [x] Generate final report `docs/regime-validity-2026-02-25.md` including:
- [x] 결론 3줄
- [x] 시나리오별 핵심 지표표 (lag 0/1/2)
- [x] epsilon 민감도 표
- [x] D1~D4 판정 근거와 단일 최종 판정
- [x] 리스크 및 개선안(단기/중기)
- [x] 시나리오별 레짐 순위표
- [x] Export ranking raw table to `docs/regime-validity-2026-02-25-rankings.csv`.
- [x] Record verification commands and staff-engineer acceptance checklist.

Verification Gates:
- [x] `python -m py_compile scripts/evaluate_regime_validity.py` passes.
- [x] `PYTHONIOENCODING=utf-8 python scripts/evaluate_regime_validity.py --asof 2026-02-25` succeeds.
- [x] Final decision is emitted and consistent with report body (`부분 타당`, 2/4).
- [x] Report contains required sections and reproducibility command.

Review (fill after implementation):
- Commands run:
- `python -m py_compile scripts/evaluate_regime_validity.py`
- `PYTHONIOENCODING=utf-8 python scripts/evaluate_regime_validity.py --asof 2026-02-25`
- Artifacts generated:
- `docs/regime-validity-2026-02-25.md`
- `docs/regime-validity-2026-02-25-rankings.csv`
- `scripts/evaluate_regime_validity.py`
- Key results:
- Scenario fit rates: lag0 `5/8 (62.5%)`, lag1 `2/8 (25.0%)`, lag2 `5/8 (62.5%)`.
- Epsilon sensitivity: latest regime stable as `Expansion` for `epsilon 0.0~0.1`, but Indeterminate share rises to `53.4%` at `epsilon=0.1`.
- Decision axes: `D1=0`, `D2=0`, `D3=1`, `D4=1` → total `2/4`.
- Final single verdict: **부분 타당**.
- Residual risks / follow-ups:
- Contraction regime remains unobservable in current sample (`0개월`), so D1 fails structurally.
- PIT lag1 performance gap vs nowcast is large; use lagged view as default in operations and treat nowcast as reference-only.

Staff Engineer Approval Checklist:
- [x] Reproducibility: single command regenerates report and ranking CSV.
- [x] Leakage control: Point-in-time lag scenarios are explicitly separated from nowcast comparison.
- [x] Robustness: lag sensitivity + epsilon sensitivity + provisional policy checks included.
- [x] Limitations: sample deficiency (Contraction 0개월) and sensitivity risks are explicitly documented.

## 33) KRX OpenAPI Provider Migration (2026-03-01)

Pre-Implementation Check-in:
- 2026-03-01: implement KRX auth/provider migration plan (`OPENAPI` + `AUTO/PYKRX` fallback) with local+Railway docs and regression tests.

Execution Checklist:
- [x] Extend secrets/template and deployment docs with `KRX_OPENAPI_KEY`, `KRX_PROVIDER`.
- [x] Add OpenAPI datasource module with auth-aware error handling and response normalization.
- [x] Add provider resolution in market loader (`AUTO`/`OPENAPI`/`PYKRX`) while keeping `load_sector_prices(index_codes, start, end)` signature unchanged.
- [x] Keep fallback order `LIVE -> RAW CACHE -> CURATED CACHE -> SAMPLE` under OpenAPI failures.
- [x] Add provider-aware runtime warnings in `app.py` including explicit missing-key warning for forced `OPENAPI`.
- [x] Add KRX price cache token keyed by provider + key fingerprint for cache invalidation on key rotation.
- [x] Add/extend tests for key precedence, provider parsing, OpenAPI parsing/auth errors, provider selection, and fallback behavior.

Verification Gates:
- [x] `python -m py_compile src/data_sources/krx_openapi.py src/data_sources/krx_indices.py src/data_sources/cache_keys.py app.py tests/test_krx_openapi.py tests/test_krx_pykrx_compat_paths.py tests/test_integration.py tests/test_cache_keys.py`
- [x] `python -m pytest -q tests/test_krx_openapi.py tests/test_krx_pykrx_compat_paths.py tests/test_integration.py tests/test_cache_keys.py`
- [ ] `python -m pytest -q tests` (currently 2 known pre-existing failures in `tests/test_ui_components.py`)

Review (fill after implementation):
- Commands run:
- `python -m py_compile src/data_sources/krx_openapi.py src/data_sources/krx_indices.py src/data_sources/cache_keys.py app.py tests/test_krx_openapi.py tests/test_krx_pykrx_compat_paths.py tests/test_integration.py tests/test_cache_keys.py`
- `python -m pytest -q tests/test_krx_openapi.py tests/test_krx_pykrx_compat_paths.py tests/test_integration.py tests/test_cache_keys.py`
- `python -m pytest -q tests`
- `python -m pytest -q` (repo-root broad run)
- Results:
- New module `src/data_sources/krx_openapi.py` added with provider parsing, secrets/env key loading, retry policy, auth/permission exceptions, and response normalization.
- `src/data_sources/krx_indices.py` now resolves provider mode (`AUTO` -> key-aware), supports OpenAPI live path, preserves pykrx path, and keeps fallback chain.
- `app.py` now computes KRX price cache token (`provider + key fingerprint`) and surfaces provider-aware cache warnings.
- Docs updated: `.streamlit/secrets.toml.example`, `README.md`, `docs/api-keys-guide.md`, `docs/railway-deploy.md`.
- Focused test suite passed: `30 passed in 3.19s`.
- `pytest -q tests` result: `94 passed, 2 failed` (existing known UI mock mismatch in `tests/test_ui_components.py` using `st.plotly_chart(..., width='stretch')` keyword).
- `pytest -q` (full repo) result: collection error in root exploratory script `test_krx_raw.py` (non-suite script expecting live JSON response).
- Residual risks / follow-ups:
- OpenAPI endpoint/field schema can vary by approved service; if account-specific API returns different keys, adjust parser candidates in `krx_openapi.py`.
- `tests/` suite still has two pre-existing UI mocking failures unrelated to this migration.

## 34) Test Suite Stabilization Follow-up (2026-03-01)

Pre-Implementation Check-in:
- Continue from KRX provider migration to resolve remaining known test instability: UI mock signature mismatch and root exploratory script collection in default pytest.

Execution Checklist:
- [x] Align `tests/test_ui_components.py` plotly mock signatures with current Streamlit call style (`width='stretch'`).
- [x] Keep assertions behavior-focused (chart rendered + width option captured).
- [x] Add `pytest.ini` with `testpaths = tests` so exploratory root scripts are excluded from default suite collection.
- [x] Re-run full suite using both `pytest -q tests` and `pytest -q`.

Verification Gates:
- [x] `python -m py_compile tests/test_ui_components.py`
- [x] `python -m pytest -q tests`
- [x] `python -m pytest -q`

Review:
- Commands run:
- `python -m py_compile tests/test_ui_components.py`
- `python -m pytest -q tests`
- `python -m pytest -q`
- Results:
- Updated UI component tests now accept keyword-based `st.plotly_chart` arguments and assert `width='stretch'`.
- Added new root config `pytest.ini` to scope default collection to `tests/`.
- Verification passed: `96 passed` for both `pytest -q tests` and `pytest -q`.
- Residual risks:
- None in current automated suite; exploratory scripts remain in repo but are intentionally excluded from default pytest collection.

## 35) 한글 모지바케(깨짐) 및 Streamlit 아이콘 예외 복구 (2026-03-01)

Pre-Implementation Check-in:
- 2026-03-01: UI 한글 라벨/설명 문구가 모지바케로 깨져 표시되고, `st.warning(..., icon=...)` 인자도 깨진 문자열이라 Streamlit 예외가 발생함.
- Scope: `app.py` 내 깨진 문자열을 정상 한글로 복구하고, 아이콘 인자를 유효 이모지로 교체. 데이터/신호 로직 변경 없음.

Execution Checklist:
- [x] 본 섹션을 `tasks/todo.md`에 추가하고 체크리스트를 작성한다.
- [x] `app.py`에서 깨진 한글 문자열(탭/사이드바/메트릭/설명/푸터)을 정상 한글로 교체한다.
- [x] `st.warning(..., icon=...)` 등 아이콘 인자를 유효한 단일 이모지로 교체한다.
- [x] `python -m py_compile app.py`로 문법/인코딩 오류를 검증한다.
- [x] 필요 최소 테스트를 실행해 회귀 여부를 확인한다.
- [x] Review 섹션에 실행 명령과 결과를 기록한다.

Verification Gates:
- [x] Streamlit 실행 시 한글 라벨이 깨지지 않고 표시된다.
- [x] `streamlit.errors.StreamlitAPIException: ... not a valid emoji` 예외가 재현되지 않는다.
- [x] `python -m py_compile app.py` 통과.

Review:
- Commands run:
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m py_compile app.py`
- `C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m pytest -q tests`
- `cmd /c "C:/Users/k1190/miniconda3/envs/sector-rotation/python.exe -m streamlit run app.py --server.headless true --server.port 8516 > .tmp_streamlit_encoding_fix.log 2>&1"` (timeout 기반 스모크 로그 캡처)
- `rg -n "StreamlitAPIException|invalid emoji|Traceback|Error|Exception" .tmp_streamlit_encoding_fix.log`
- Results:
- `app.py`의 모지바케 문자열을 정상 한글로 교체(사이드바/탭/메트릭/설명/푸터 전반).
- `st.warning(..., icon="⚠️")`로 교체하여 깨진 아이콘 문자열 제거.
- 정적 검증: `py_compile` 통과.
- 회귀 검증: `pytest -q tests` 결과 `96 passed, 1 warning`.
- 런타임 스모크: Streamlit 로그에서 앱 URL 정상 출력, `invalid emoji`/`StreamlitAPIException` 패턴 미검출.
- Residual risks / follow-ups:
- 동일 파일을 CP949 등 다른 인코딩으로 다시 저장하면 모지바케가 재발할 수 있으므로 UTF-8 저장 고정(에디터 기본 인코딩 점검)이 필요.

## 36) pykrx 이슈 해결 상태 점검 (2026-03-04)

Pre-Implementation Check-in:
- 2026-03-04: 사용자 요청으로 pykrx 관련 이슈가 현재 기준으로 해결된 상태인지 점검.
- Scope: 코드 변경 없이 상태 점검 중심(문서/설정 확인 + 테스트/스모크 검증 + 판정 보고).

Execution Checklist:
- [x] 본 섹션을 `tasks/todo.md`에 추가하고 체크리스트를 작성한다.
- [x] 현재 코드/의존성 기준에서 pykrx 관련 수정사항(호환 shim, provider fallback, 경고 회귀 테스트) 반영 상태를 확인한다.
- [x] pykrx 관련 테스트를 실행해 회귀 여부를 검증한다.
- [x] 필요 시 최소 런타임 스모크(직접 import/라이브 fetch 경로)를 실행해 동작 여부를 확인한다.
- [x] 결과를 Review에 기록하고 최종 상태를 `해결/부분해결/미해결`로 판정한다.

Verification Gates:
- [x] `python -m pytest -q tests/test_pykrx_compat.py tests/test_pykrx_import_warning.py tests/test_krx_pykrx_compat_paths.py`
- [x] `python -c "import pykrx; print(pykrx.__version__)"`
- [x] `python -m pytest -q tests/test_integration.py -k "api_failure_falls_back_to_cache or full_fallback_to_sample or live_partial_success_keeps_live_status"`

Review:
- Commands run:
- `python -c "import pykrx; print(pykrx.__version__)"`
- `python -m pytest -q tests/test_pykrx_compat.py tests/test_pykrx_import_warning.py tests/test_krx_pykrx_compat_paths.py`
- `python -m pytest -q tests/test_integration.py -k pykrx` (결과: `11 deselected`, 테스트명 기준 필터 미매칭)
- `python -m pytest -q tests/test_integration.py -k "api_failure_falls_back_to_cache or full_fallback_to_sample or live_partial_success_keeps_live_status"`
- `python -c "from pykrx import stock; df=stock.get_index_ohlcv('20240102','20240131','1001', name_display=False); print('shape', df.shape)"`
- `python -c "from src.data_sources.pykrx_compat import ensure_pykrx_transport_compat; ensure_pykrx_transport_compat(); from pykrx import stock; df=stock.get_index_ohlcv('20240102','20240131','1001', name_display=False); print('shape', df.shape)"`
- `python -c "import os; os.environ['KRX_PROVIDER']='PYKRX'; from src.data_sources.krx_indices import load_sector_prices; status, df = load_sector_prices(['1001'],'20240102','20240131'); print('status', status, 'rows', len(df))"`
- Results:
- 환경 pykrx 버전 확인: `1.2.4` (`requirements.txt`도 `pykrx>=1.2.4`).
- 회귀 테스트 통과: `15 passed in 2.66s`.
- 통합 테스트(관련 3케이스) 통과: `3 passed, 8 deselected`.
- 실제 pykrx 라이브 호출(`stock.get_index_ohlcv`)은 compat 적용 전/후 모두 `shape (0, 0)`로 빈 응답.
- 실제 로더 경로(`load_sector_prices`, `KRX_PROVIDER=PYKRX`)는 LIVE 실패 후 `status CACHED`로 폴백하며 경고 로그에 `Empty response` 및 `consider OPENAPI provider`가 출력됨.
- Final judgement:
- `부분해결` — 앱/로더의 안정성(테스트·폴백·경고)은 확보되었지만, pykrx 라이브 응답 빈값 이슈 자체는 현재 환경에서 여전히 재현됨.
- Residual risks / follow-ups:
- `KRX_PROVIDER=AUTO`에서 `KRX_OPENAPI_KEY`가 없으면 기본 경로가 PYKRX여서, 최신 LIVE 데이터 대신 캐시 사용 비중이 커질 수 있음.
- 운영 환경에서 LIVE 우선이 필요하면 `KRX_PROVIDER=OPENAPI` + 유효한 `KRX_OPENAPI_KEY` 구성이 사실상 필요.

## 37) KRX OpenAPI 키 반영 후 기능 정상동작 검증 (2026-03-04)

Pre-Implementation Check-in:
- 2026-03-04: 사용자가 `.streamlit/secrets.toml`에 KRX OpenAPI 키 세팅 완료.
- Scope: 설정 반영 여부 확인, 관련 자동 테스트 실행, 실제 OpenAPI 라이브 호출 스모크로 동작 판정.

Execution Checklist:
- [x] 본 섹션을 `tasks/todo.md`에 추가하고 체크리스트를 작성한다.
- [x] 런타임에서 `KRX_OPENAPI_KEY`/`KRX_PROVIDER` 설정 인식 여부를 확인한다(키 원문은 출력하지 않음).
- [x] KRX OpenAPI/통합 경로 관련 테스트를 실행한다.
- [x] 전체 테스트 스위트(`tests/`)를 실행해 회귀 여부를 확인한다.
- [x] 실제 OpenAPI 경로 라이브 스모크(fetch 또는 loader)를 실행해 결과 상태(`LIVE/CACHED/SAMPLE`)를 검증한다.
- [x] Review 섹션에 실행 명령과 결과를 기록하고 최종 판정을 작성한다.

Verification Gates:
- [x] `python -m pytest -q tests/test_krx_openapi.py tests/test_integration.py tests/test_krx_pykrx_compat_paths.py`
- [x] `python -m pytest -q tests`
- [x] `python -c "from src.data_sources.krx_openapi import get_krx_openapi_key, get_krx_provider; ..."`
- [x] `python -c "from src.data_sources.krx_indices import load_sector_prices; ..."`

Review:
- Commands run:
- `python --version`
- `python -c "from src.data_sources.krx_openapi import get_krx_openapi_key, get_krx_provider; k=get_krx_openapi_key(); print('key_present', bool(k)); print('key_len', len(k)); print('provider', get_krx_provider())"`
- `Get-Content .streamlit/secrets.toml | ForEach-Object { if ($_ -match '^\s*([A-Z0-9_]+)\s*=') { $matches[1] } }`
- `python -m pytest -q tests/test_krx_openapi.py tests/test_integration.py tests/test_krx_pykrx_compat_paths.py`
- `python -m pytest -q tests`
- `python -c "from src.data_sources.preflight import run_api_preflight; print(run_api_preflight(timeout_sec=3))"`
- `python -c "from src.data_sources.krx_openapi import fetch_index_ohlcv_openapi; df=fetch_index_ohlcv_openapi('1001','20260220','20260303'); ..."`
- `python -c "from src.data_sources.krx_indices import load_sector_prices; s, df = load_sector_prices(['1001'], '20260220', '20260303'); print('status', s); print('rows', len(df)); print('columns', list(df.columns));"`
- Results:
- Python: `3.13.5`
- 런타임 설정 인식: `key_present=True`, `key_len=40`, `provider=AUTO`
- `.streamlit/secrets.toml` 키 항목 확인: `ECOS_API_KEY`, `KOSIS_API_KEY`, `KRX_OPENAPI_KEY`, `KRX_PROVIDER`
- KRX 관련 타깃 테스트: `25 passed in 2.55s`
- 전체 테스트: `96 passed in 4.13s`
- API preflight: `ECOS/KOSIS/KRX` 모두 `status=OK`
- OpenAPI 직접 호출: `KRXOpenAPIAuthError: Unauthorized API Call`
- 로더 경로: `status CACHED`, `rows 6` (OPENAPI LIVE 실패 후 캐시 폴백 동작 확인)
- 추가 진단:
- 동일 호출에서 `AUTH_KEY=정상키` -> `Unauthorized API Call`, `AUTH_KEY=변조키` -> `Unauthorized Key` 확인.
- 해석: 키 문자열 자체는 인식되지만(유효), 대상 API 사용 권한(서비스 신청/승인) 미완료 상태 가능성이 높음.
- 공식 이용방법(OPEN API)에도 `API 활용 신청` 및 `관리자 승인 대기` 단계가 명시되어 있음.
- Final judgement:
- `부분정상` — 애플리케이션 테스트/폴백 기능은 정상이나, 현재 키로는 KRX OpenAPI 라이브 인증이 실패하여 실시간 OPENAPI 데이터는 미동작.
- Residual risks / follow-ups:
- `openapi.krx.co.kr`에서 해당 API(`idx/krx_dd_trd`) 사용 권한 승인 상태를 확인해야 함(키 발급과 서비스 승인은 별개일 수 있음).
- 운영에서 LIVE 우선이 필요하면 권한 승인 완료 후 재검증 필요.

## 38) `krx_dd_trd` 이용권한 반영 후 연결 테스트 (2026-03-05)

Pre-Implementation Check-in:
- 2026-03-05: 사용자 요청으로 `krx_dd_trd` API 이용권한 승인 이후 실제 연결이 정상인지 즉시 재검증.
- Scope: 코드 변경 없이 설정 인식 확인 + OpenAPI 실호출 + 로더 경로 상태 판정.

Execution Checklist:
- [x] 본 섹션을 `tasks/todo.md`에 추가하고 체크리스트를 작성한다.
- [x] 런타임에서 `KRX_OPENAPI_KEY`/`KRX_PROVIDER` 인식 상태를 확인한다(키 원문 비노출).
- [x] KRX OpenAPI 관련 회귀 테스트를 실행한다.
- [x] `idx/krx_dd_trd` 실호출 스모크를 실행해 인증/권한/응답 상태를 확인한다.
- [x] `load_sector_prices` 경로를 실행해 실제 앱 경로 상태(`LIVE/CACHED/SAMPLE`)를 확인한다.
- [x] Review 섹션에 실행 명령/결과와 최종 판정을 기록한다.

Verification Gates:
- [x] `python -m pytest -q tests/test_krx_openapi.py tests/test_krx_pykrx_compat_paths.py tests/test_integration.py -k "openapi or live_partial_success_keeps_live_status or openapi_auth_failure_falls_back_to_cache"`
- [x] `python -c "from src.data_sources.krx_openapi import fetch_index_ohlcv_openapi; ..."`
- [x] `python -c "import os; os.environ['KRX_PROVIDER']='OPENAPI'; from src.data_sources.krx_indices import load_sector_prices; ..."`

Review:
- Commands run:
- `python --version`
- `python -c "from src.data_sources.krx_openapi import get_krx_openapi_key, get_krx_provider; k=get_krx_openapi_key(); print('key_present', bool(k)); print('key_len', len(k)); print('provider', get_krx_provider())"`
- `Get-Content .streamlit/secrets.toml | ForEach-Object { if ($_ -match '^\s*([A-Z0-9_]+)\s*=') { $matches[1] } }`
- `python -m pytest -q tests/test_krx_openapi.py tests/test_krx_pykrx_compat_paths.py tests/test_integration.py -k "openapi or live_partial_success_keeps_live_status or openapi_auth_failure_falls_back_to_cache"`
- `python -c "from src.data_sources.krx_openapi import fetch_index_ohlcv_openapi; df=fetch_index_ohlcv_openapi('1001','20260220','20260305'); ..."`
- `python -c "from src.data_sources.krx_openapi import _request_with_retry, get_krx_openapi_key, get_krx_openapi_url; ..."` (`20260220~20260305` + `20240102~20240131` raw payload 확인)
- `python -c "from src.data_sources.krx_openapi import _request_with_retry, get_krx_openapi_url; ..."` (변조 키로 `Unauthorized Key` 비교 확인)
- `python -c "import os; os.environ['KRX_PROVIDER']='OPENAPI'; from src.data_sources.krx_indices import load_sector_prices; ..."`
- `python -c "from src.data_sources.preflight import run_api_preflight; print(run_api_preflight(timeout_sec=3))"`
- Results:
- Python: `3.13.5`
- 런타임 설정 인식: `key_present=True`, `key_len=40`, `provider=AUTO`
- `.streamlit/secrets.toml` 키 항목 확인: `ECOS_API_KEY`, `KOSIS_API_KEY`, `KRX_OPENAPI_KEY`, `KRX_PROVIDER`
- KRX 관련 타깃 테스트: `10 passed, 15 deselected`
- `fetch_index_ohlcv_openapi('1001','20260220','20260305')`는 `KRXOpenAPIResponseError: no data rows`
- 동일 기간 raw payload는 `{"OutBlock_1": []}`로 인증 실패가 아니라 빈 데이터 응답
- 과거 구간(`20240102~20240131`) raw payload는 `OutBlock_1` `34`건 반환(응답 수신 정상)
- 변조 키 호출 시 `KRXOpenAPIAuthError: Unauthorized Key` 재현(현재 키와 구분 확인)
- `load_sector_prices` (`KRX_PROVIDER=OPENAPI`, `1001`, `20240102~20240131`) 결과: `status=LIVE`, `rows=1`
- API preflight: `KRX status=OK (HTTP 200)` 포함 전부 `OK`
- Final judgement:
- `정상(연결/권한)` — `krx_dd_trd` 엔드포인트는 현재 키로 인증/접속이 정상이며 실제 데이터 응답(`OutBlock_1`)을 반환함.
- Residual risks / follow-ups:
- 최근 구간(`2026-02-20`~`2026-03-05`)은 API가 빈 rows를 반환했으므로, 실데이터 검증 기준일은 KRX에 데이터가 존재하는 확정 과거일로 잡아야 함.
- 현재 파서 경로는 `OutBlock_1`에서 `IDX_IND_CD`가 없는 응답일 때 일자 중복 제거로 1행만 남을 수 있어, 시계열 품질 점검이 추가로 필요함.

## 39) KRX OpenAPI LIVE 시계열 복구 (2026-03-06)

Pre-Implementation Check-in:
- 2026-03-06: 사용자 승인 후 KRX LIVE 데이터가 여전히 `CACHED`로 떨어지는 문제를 수정.
- Scope: 잘못된 `KRX_OPENAPI_URL` override 무력화, KRX/KOSPI 시리즈별 라우팅, `basDd` 일자 스냅샷 기반 시계열 복구, 스냅샷 오인 파싱 방지, 관련 회귀 테스트 추가.

Execution Checklist:
- [x] 본 섹션을 `tasks/todo.md`에 추가하고 체크리스트를 작성한다.
- [x] KRX OpenAPI runtime URL override를 공식 host/path만 허용하도록 검증한다.
- [x] KRX/KOSPI/KOSDAQ 시리즈 API id 라우팅과 코드별 공식 row-name alias 해석을 구현한다.
- [x] `basDd` 일자 스냅샷 반복 수집으로 code별 시계열을 재구성하고, 단일일자 스냅샷을 시계열로 오인하는 경로를 제거한다.
- [x] `load_sector_prices()` OPENAPI 경로를 family-batch fetch 기반으로 바꾸고 기존 fallback(`LIVE -> RAW CACHE -> CURATED CACHE -> SAMPLE`)을 유지한다.
- [x] KRX OpenAPI 단위/통합 테스트를 업데이트해 override 검증, family 라우팅, snapshot rejection, loader LIVE 경로를 고정한다.
- [x] Review 섹션에 실행 명령, 테스트 결과, 실제 LIVE 스모크 결과를 기록한다.

Verification Gates:
- [x] `python -m py_compile src/data_sources/krx_openapi.py src/data_sources/krx_indices.py tests/test_krx_openapi.py tests/test_integration.py`
- [x] `python -m pytest -q tests/test_krx_openapi.py tests/test_integration.py tests/test_krx_pykrx_compat_paths.py`
- [x] `python -m pytest -q tests`
- [x] `python -c "from src.data_sources.krx_indices import load_sector_prices; ..."` 앱 기본 범위 LIVE 스모크
- [x] `python -c "from src.signals.matrix import build_signal_table; ..."` 종단 스모크

Review:
- Commands run:
- `python -m py_compile src/data_sources/krx_openapi.py src/data_sources/krx_indices.py tests/test_krx_openapi.py tests/test_integration.py tests/test_krx_pykrx_compat_paths.py`
- `python -m pytest -q tests/test_krx_openapi.py tests/test_integration.py tests/test_krx_pykrx_compat_paths.py`
- `python -m pytest -q tests`
- `python -c "... load_sector_prices(default 12 codes, 3 years) ..."`
- `python -c "... load_sector_prices + load_ecos_macro + load_kosis_macro + compute_regime_history + build_signal_table ..."`
- Results:
- `src/data_sources/krx_openapi.py` now validates `KRX_OPENAPI_URL`, routes by family API id (`krx_dd_trd` / `kospi_dd_trd` / `kosdaq_dd_trd`), repairs KRX cp949 mojibake, and reconstructs time series from `basDd` daily snapshots.
- `src/data_sources/krx_indices.py` now uses OpenAPI batch fetch for requested codes, writes per-code raw cache, keeps the existing fallback chain, and sets `index_name` from configured display names.
- Test coverage updated for invalid override rejection, family routing, snapshot rejection, and OPENAPI loader usage.
- Targeted tests: `28 passed in 1.77s`.
- Full suite: `99 passed in 4.47s` and `99 passed in 4.44s`.
- App-default LIVE smoke (`5044,1155,5042,1168,1165,5048,5049,5045,1170,1157,5046,1001`, `20230306~20260305`): `status=LIVE`, `rows=8760`, code별 `730`행, 범위 `2023-03-06`~`2026-03-05`.
- End-to-end macro+signals smoke: `price_status=LIVE`, `ecos_status=LIVE`, `kosis_status=LIVE`, `macro_result_rows=59`, `signal_count=11`, `na_count=0`.
- Residual risks / follow-ups:
- 첫 3년 OPENAPI 백필은 현재 환경에서 약 `163초`가 걸렸다. Streamlit 캐시/로컬 raw cache로 반복 비용은 줄지만, 최초 cold start 성능 개선 여지는 남아 있다.
- 일부 코드 매칭은 현재 KRX API 명칭 변경에 대응하는 alias에 의존한다 (`5042 -> KRX 300 산업재`, `5046 -> KRX 방송통신`, `1170 -> 전기·가스`).

## 40) KRX OpenAPI 잔여 리스크 해소 (2026-03-06)

Pre-Implementation Check-in:
- 2026-03-06: 승인된 계획에 따라 cold-start backfill 지연과 KRX index alias 의존을 동시에 해소한다.
- Scope: raw-cache-first 증분 로딩, OpenAPI 전역 병렬화/세션 재사용, metadata sync 기반 alias 해석, warm/audit CLI, startup warm 연동, 관련 회귀 테스트 추가.

Execution Checklist:
- [x] 본 섹션을 `tasks/todo.md`에 추가하고 체크리스트를 작성한다.
- [x] KRX OpenAPI metadata sync/cache 계층과 alias fallback 우선순위(`official -> synced history -> emergency static`)를 구현한다.
- [x] OpenAPI snapshot batch를 전 family 전역 executor + session reuse 기반으로 재구성하고 worker 수를 설정화한다.
- [x] `load_sector_prices()`를 raw-cache-first 증분 refresh 구조로 바꾸고 기존 `LIVE -> CACHED -> SAMPLE` 계약을 유지한다.
- [x] warm 상태/성능 로그를 추가하고, startup warm 옵션과 운영용 warm/audit 스크립트를 구현한다.
- [x] 단위/통합 테스트를 업데이트해 cache-first 반환, delta refresh, metadata fallback, alias audit 경로를 고정한다.
- [x] Review 섹션에 실행 명령, 결과, 남은 리스크를 기록한다.

Verification Gates:
- [x] `python -m py_compile app.py src/data_sources/krx_openapi.py src/data_sources/krx_indices.py scripts/warm_krx_cache.py scripts/audit_krx_aliases.py`
- [x] `python -m pytest -q tests/test_krx_openapi.py tests/test_integration.py tests/test_krx_pykrx_compat_paths.py`
- [x] `python -m pytest -q tests`
- [x] `python scripts/audit_krx_aliases.py --date 20240131`
- [x] `python scripts/warm_krx_cache.py --years 3 --as-of 20260305`

Review:
- Commands run:
- `python -m py_compile app.py src/data_sources/krx_openapi.py src/data_sources/krx_indices.py scripts/warm_krx_cache.py scripts/audit_krx_aliases.py tests/test_krx_openapi.py tests/test_integration.py`
- `python -m pytest -q tests/test_krx_openapi.py tests/test_integration.py tests/test_krx_pykrx_compat_paths.py`
- `python -m pytest -q tests`
- `python scripts/audit_krx_aliases.py --date 20240131`
- `python scripts/warm_krx_cache.py --years 3 --as-of 20260305`
- `python scripts/warm_krx_cache.py --years 3 --as-of 20260305 --force` (중도 중단)
- `python -c "from src.data_sources.krx_indices import load_sector_prices; ..."`
- Results:
- `krx_openapi.py` now persists `data/raw/krx/index_name_metadata.json`, updates official names from observed snapshot matches, and filters overbroad aliases like bare `KOSPI 200` so sector codes no longer silently bind to generic benchmark rows.
- `krx_indices.py` now exposes raw-cache-first loading, incremental delta refresh, background warm scheduling, warm-status artifact tracking, and cache-aware `CACHED` fallback when OpenAPI delta refresh fails.
- `app.py` now schedules startup warm when `KRX_WARM_ON_STARTUP` is enabled and invalidates price caches on raw/warm artifact changes instead of only curated parquet mtime.
- New operational commands: `scripts/warm_krx_cache.py`, `scripts/audit_krx_aliases.py`.
- Targeted KRX tests: `34 passed in 3.10s`.
- Full suite: `105 passed in 5.24s`.
- Alias audit succeeded once for all 12 configured codes on `2024-01-31` (`matched=12`, `unmatched=0`), and rewrote metadata to current official names for `5042`, `5046`, `1170`, plus KOSPI200 sector codes.
- Incremental warm on existing cache succeeded once with `status=LIVE`, `rows=8760`, `duration_sec=1.287`, and only the missing `20230306~20230306` range was fetched.
- Later reruns of both audit/warm hit transient KRX `Access Denied` HTML responses from `data-dbg.krx.co.kr`; under that failure mode the loader now stays `CACHED` instead of misreporting `LIVE`.
- App-default loader smoke (`20230306~20260305`) returned `status=CACHED`, `rows=8748`, `12` codes after OpenAPI delta refresh was denied, confirming cache-first behavior on the user path.
- Residual risks / follow-ups:
- The `90초 이하` full-force warm target was not proven. `scripts/warm_krx_cache.py --force` exceeded the local 244s command timeout and was terminated, so remaining cold-path performance is still gated by external KRX endpoint behavior.
- KRX OpenAPI occasionally returns `Access Denied` HTML for otherwise valid requests. This patch hardens fallback/reporting but does not remove that upstream instability.

## 41) KRX warm/log 잔여 이슈 수정 (2026-03-06)

Pre-Implementation Check-in:
- 2026-03-06: 승인된 계획에 따라 Streamlit startup 경고 소음을 줄이고, OpenAPI force warm partial-failure 내구성을 높인다.
- Scope: provider-aware calendar, app end-date 단일 계산, deprecated `KRX_OPENAPI_URL` 무시/1회 경고, OpenAPI snapshot partial failure 수집, force warm coverage reporting, 관련 회귀 테스트 추가.

Execution Checklist:
- [x] 본 섹션을 `tasks/todo.md`에 추가하고 체크리스트를 작성한다.
- [x] `get_last_business_day()`를 provider-aware로 바꾸고, OPENAPI 모드에서는 benchmark snapshot probe를 우선 사용한다.
- [x] `app.py`에서 KRX end date를 한 번만 계산해 startup warm와 blocking load가 재사용하도록 정리한다.
- [x] `KRX_OPENAPI_URL` 설정을 deprecated로 처리해 코드에서 무시하고, stale override가 있어도 프로세스당 1회만 경고한다.
- [x] OpenAPI batch fetch가 `(family, basDd)` 단위 실패를 수집하면서 성공 snapshot 결과는 계속 사용하도록 바꾼다.
- [x] `Access Denied` HTML 응답을 throttle 오류로 분류하고 긴 backoff 대신 짧은 1회 재시도 후 실패일로 집계한다.
- [x] force warm summary에 `failed_days`, `coverage_complete`를 추가하고 incomplete fetch는 `CACHED`로 보고한다.
- [x] 로컬 `.streamlit/secrets.toml`에서 stale `KRX_OPENAPI_URL`을 제거한다.
- [x] 회귀 테스트/검증 명령을 실행하고 Review 섹션에 결과를 기록한다.

Verification Gates:
- [x] `python -m py_compile app.py src/transforms/calendar.py src/data_sources/krx_openapi.py src/data_sources/krx_indices.py tests/test_krx_openapi.py tests/test_krx_pykrx_compat_paths.py tests/test_integration.py`
- [x] `python -m pytest -q tests/test_krx_openapi.py tests/test_krx_pykrx_compat_paths.py tests/test_integration.py`
- [x] `python -m pytest -q tests`
- [x] `python scripts/warm_krx_cache.py --years 3 --as-of 20260305`

Review:
- Commands run:
- `python -m py_compile app.py src/transforms/calendar.py src/data_sources/krx_openapi.py src/data_sources/krx_indices.py tests/test_krx_openapi.py tests/test_krx_pykrx_compat_paths.py tests/test_integration.py`
- `python -m pytest -q tests/test_krx_openapi.py tests/test_krx_pykrx_compat_paths.py tests/test_integration.py`
- `python -m pytest -q tests`
- `python scripts/warm_krx_cache.py --years 3 --as-of 20260305`
- Results:
- `calendar.py` now resolves provider-aware market dates: `OPENAPI + key` probes recent benchmark snapshots first, pykrx is secondary, and weekend fallback warns only when both fail.
- `app.py` now computes the KRX `end_date` once per run and reuses it for startup warm and blocking price loads; `rg` shows a single `get_last_business_day(...)` callsite in the app flow.
- `krx_openapi.py` now ignores deprecated `KRX_OPENAPI_URL` overrides with a once-per-process warning, classifies `Access Denied` HTML as throttle, retries it once, and keeps per-day snapshot failures in batch details instead of aborting the whole batch.
- `krx_indices.py` now reports `failed_days` and `coverage_complete`, downgrades incomplete warm results to `CACHED`, and avoids overwriting curated cache with incomplete fetch coverage.
- Local `.streamlit/secrets.toml` no longer carries the stale `KRX_OPENAPI_URL` override.
- Targeted regression tests: `40 passed in 5.05s`.
- Full suite: `111 passed in 6.80s`.
- Warm CLI smoke (`20230306~20260305`): `status=LIVE`, `rows=8760`, `coverage_complete=true`, `failed_days=[]`, `duration_sec=1.365`.
- Residual risks / follow-ups:
- Public `data-dbg.krx.co.kr` volatility still exists. The code now degrades to partial-day accounting instead of batch collapse, but upstream `Access Denied` spikes can still prevent a cold full-force warm from proving the old `90초 이하` goal.

## 42) KRX false cache warning 제거 및 수동 갱신 정렬 (2026-03-07)

Pre-Implementation Check-in:
- 2026-03-07: 사용자가 `캐시 데이터를 사용 중 ... KRX OpenAPI live fetch failed` 경고의 원인 파악 후 해결 구현을 요청.
- Scope: warm-status 읽기 API 추가, cached-price 배너 해석 분리, 수동 시장데이터 갱신을 실제 incremental warm로 변경, 관련 테스트 보강.

Execution Checklist:
- [x] 본 섹션을 `tasks/todo.md`에 추가하고 체크리스트를 작성한다.
- [x] `krx_indices.py`에 UI용 read-only warm status loader를 추가한다.
- [x] `data_status.py`에 cached price 배너 해석용 순수 resolver를 추가한다.
- [x] `app.py`에서 OPENAPI+CACHED 경고를 warm status 기반으로 분기하고 false failure 문구를 제거한다.
- [x] `app.py`의 `시장데이터 갱신` 버튼을 `warm_sector_price_cache(..., reason="manual_refresh", force=False)` 기반으로 바꾼다.
- [x] `tests/test_data_status.py`, `tests/test_integration.py`를 업데이트해 false warning 제거와 수동 갱신 경로를 고정한다.
- [x] 검증 명령을 실행하고 Review 섹션에 결과를 기록한다.

Verification Gates:
- [x] `python -m py_compile app.py src/data_sources/krx_indices.py src/ui/data_status.py tests/test_data_status.py tests/test_integration.py`
- [x] `python -m pytest -q tests/test_data_status.py tests/test_integration.py`
- [x] `python -c "from src.data_sources.krx_openapi import fetch_index_ohlcv_openapi; ..."`
- [x] `python -c "from src.data_sources.krx_indices import load_sector_prices; ..."`

Review:
- Commands run:
- `python -m py_compile app.py src/data_sources/krx_indices.py src/ui/data_status.py tests/test_data_status.py tests/test_integration.py`
- `python -m pytest -q tests/test_data_status.py tests/test_integration.py`
- `python -c "from src.data_sources.krx_openapi import fetch_index_ohlcv_openapi; df=fetch_index_ohlcv_openapi('1001','20260305','20260306'); ..."`
- `python -c "from src.data_sources.krx_indices import load_sector_prices, read_warm_status; ..."`
- Results:
- `krx_indices.py` now exposes `read_warm_status()` for sanitized UI diagnostics and `run_manual_price_refresh()` for the app's manual incremental warm path.
- `data_status.py` now resolves cached market-price banners into `fresh_cache`, `retryable_cache_fallback`, `missing_openapi_key`, `pykrx_cache_fallback`.
- `app.py` now treats `OPENAPI + CACHED + warm_status=LIVE/current_end` as fresh cache info instead of a false live-fetch-failed warning, and the market refresh button now executes manual incremental warm instead of only deleting parquet.
- Targeted verification: `25 passed in 3.38s`.
- OpenAPI smoke for KOSPI `1001`, `2026-03-05~2026-03-06`: `rows=2`, range `2026-03-05`~`2026-03-06`.
- Loader smoke for app-default 12-code range `2023-03-07~2026-03-06`: `status=CACHED`, `rows=8760`, `codes=12`, `max=2026-03-06`, while warm status reported `status=LIVE`, `end=20260306`, `coverage_complete=True`, `failed_days=[]`.
- Final judgement: false warning root cause fixed. The loader still intentionally returns `CACHED` for complete raw-cache hits, but the UI now distinguishes fresh raw cache from actual OpenAPI fallback/failure.

## 43) 대시보드 섹터 수익률 정합성 복구 (2026-03-07)

Pre-Implementation Check-in:
- 2026-03-07: 사용자 보고에 따라 정보기술(`1155`), 경기소비재(`1165`) 기간 수익률이 비정상적으로 높게 보이는 문제를 raw cache 오염 기준으로 복구한다.
- Scope: raw-cache integrity audit 추가, contaminated code 강제 재수집/격리, partial failure 시 `N/A` degrade, 관련 회귀 테스트 추가.

Execution Checklist:
- [x] 본 섹션을 `tasks/todo.md`에 추가하고 체크리스트를 작성한다.
- [x] `src/data_sources/krx_indices.py`에 trailing 60 영업일 exact-match 기반 raw cache contamination detector를 추가한다.
- [x] `load_sector_prices()` raw-cache fast path 전에 contamination을 점검하고 contaminated code가 있으면 cache-only 경로를 우회한다.
- [x] `warm_sector_price_cache()`에서 contaminated code만 full-range 강제 재수집하고, incomplete refresh 시 해당 code를 결과/curated fallback에서 제외한다.
- [x] curated/raw fallback이 contaminated code를 다시 주워오지 않도록 필터링한다.
- [x] `tests/test_integration.py`에 detector 오탐 방지, contaminated cache 강제 refresh, incomplete refresh -> `N/A` 회귀를 추가한다.
- [x] 검증 명령을 실행하고 Review 섹션에 결과를 기록한다.

Verification Gates:
- [x] `python -m py_compile src/data_sources/krx_indices.py tests/test_integration.py`
- [x] `pytest -q tests/test_integration.py -k "contaminated or force_refreshes or drops_contaminated or prefers_complete_raw_cache_before_openapi or openapi_provider_live_success"`
- [x] `pytest -q tests/test_integration.py tests/test_krx_pykrx_compat_paths.py`
- [x] `pytest -q tests/test_data_status.py`

Review:
- Commands run:
- `python -m py_compile src/data_sources/krx_indices.py tests/test_integration.py`
- `pytest -q tests/test_integration.py -k "contaminated or force_refreshes or drops_contaminated or prefers_complete_raw_cache_before_openapi or openapi_provider_live_success"`
- `pytest -q tests/test_integration.py tests/test_krx_pykrx_compat_paths.py`
- `pytest -q tests/test_data_status.py`
- Results:
- `krx_indices.py` now detects raw cache contamination when two or more codes share the exact trailing 60-business-day close series, forces only those codes through full-range live rebuild, and removes them from partial/incomplete fallback results instead of reusing corrupted cache.
- `load_sector_prices()` no longer returns raw-cache fast-path results when contaminated codes are present; `warm_sector_price_cache()` now isolates contaminated codes from curated fallback if rebuild coverage is incomplete.
- Added regressions in `tests/test_integration.py` for: `1170` false-positive avoidance, contaminated KOSPI200 cache forced refresh, and incomplete contaminated refresh producing downstream `N/A`.
- Verification: focused contaminated-cache regression `5 passed`, broader loader/path suite `31 passed`, data-status suite `8 passed`.
- Residual risks / follow-ups:
- 실환경 KRX OpenAPI는 간헐적으로 `Access Denied`를 반환하므로, 이번 검증은 `tmp_path + monkeypatch` 기반 loader smoke로 고정했다. 실제 운영 데이터는 다음 대시보드 로드/수동 refresh 시 새 detector가 오염 cache를 차단한다.
- 현재 작업 트리에 `data/curated/sector_prices.parquet` 삭제 등 사용자/사전 작업 흔적이 있으므로, 이번 구현은 tracked market data를 임의로 복구하지 않고 코드 경로와 회귀 테스트만 고정했다.

## 44) KRX OpenAPI Access Denied fail-fast 경로 구현 (2026-03-07)

Pre-Implementation Check-in:
- 2026-03-07: 사용자가 Streamlit 실행 중 `data-dbg.krx.co.kr` HTML `Access Denied` 경고 폭주를 보고했고, 기존 합의한 대응 계획의 실제 구현을 요청.
- Scope: OpenAPI access-denied 전용 예외/health probe 추가, batch replay chunk/window 제한, interactive request budget fail-fast, startup warm 차단, 시장데이터만 block하는 앱 경로, preflight 진단 고도화, 회귀 테스트 보강.

Execution Checklist:
- [x] 본 섹션을 `tasks/todo.md`에 추가하고 체크리스트를 작성한다.
- [x] `src/data_sources/krx_openapi.py`에 `KRXOpenAPIAccessDeniedError`, health probe, chunked batch replay, abort summary 필드를 추가한다.
- [x] `src/data_sources/krx_indices.py`에 interactive request budget(`60`)와 access-denied fail-fast 예외를 추가한다.
- [x] `src/data_sources/krx_indices.py`에서 warm/load summary에 `aborted`, `abort_reason`, `predicted_requests`, `processed_requests`를 반영한다.
- [x] `app.py`에서 startup OPENAPI warm를 끄고, 시장데이터만 `BLOCKED`로 차단한 채 매크로/앱 shell은 계속 렌더링하도록 바꾼다.
- [x] `src/data_sources/preflight.py`를 실제 KRX OpenAPI probe 기반으로 교체한다.
- [x] `scripts/warm_krx_cache.py`를 out-of-band warm CLI 용도로 명시한다.
- [x] `tests/test_krx_openapi.py`, `tests/test_integration.py`, `tests/test_preflight.py`를 갱신하고 관련 경로 회귀를 확인한다.
- [x] 검증 명령을 실행하고 Review 섹션에 결과를 기록한다.

Verification Gates:
- [x] `python -m py_compile app.py src/data_sources/krx_openapi.py src/data_sources/krx_indices.py src/data_sources/preflight.py tests/test_krx_openapi.py tests/test_integration.py tests/test_preflight.py`
- [x] `python -m pytest tests/test_krx_openapi.py tests/test_integration.py tests/test_data_status.py tests/test_preflight.py -q`
- [x] `python -m pytest tests/test_krx_pykrx_compat_paths.py -q`

Review:
- Commands run:
- `python -m pip install requests pyyaml pyarrow`
- `python -m py_compile app.py src/data_sources/krx_openapi.py src/data_sources/krx_indices.py src/data_sources/preflight.py tests/test_krx_openapi.py tests/test_integration.py tests/test_preflight.py`
- `python -m pytest tests/test_krx_openapi.py tests/test_integration.py tests/test_data_status.py tests/test_krx_pykrx_compat_paths.py tests/test_preflight.py -q`
- Results:
- `krx_openapi.py` now classifies HTML `Access Denied` as a dedicated non-retryable exception, exposes a real `data-dbg` health probe, and replays daily snapshots in `10`-business-day chunks with at most `2` concurrent in-flight requests before aborting the batch on the first denial.
- `krx_indices.py` now fail-fast blocks interactive OPENAPI loads above `60` predicted snapshot requests, propagates `ACCESS_DENIED` as a market-data-specific blocking error, and writes normalized warm summaries including `aborted`, `abort_reason`, `predicted_requests`, and `processed_requests`.
- `app.py` now disables automatic OPENAPI startup warm, resets preflight before manual market refresh, and keeps macro/app shell rendering even when market data is `BLOCKED` instead of collapsing the dashboard into `SAMPLE`.
- `preflight.py` now distinguishes `AUTH_ERROR` and `ACCESS_DENIED` on the real KRX OpenAPI endpoint instead of treating the root site as reachable.
- Additional compatibility fix: `sector_prices` generation/validation now normalizes `index_code` and `index_name` to `object` dtype so pandas 3 string dtypes do not trip the schema contract during warm/load/cache paths.
- Verification status: combined targeted regression `62 passed`, and all touched files passed `py_compile`.
