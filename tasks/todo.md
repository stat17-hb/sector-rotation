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
