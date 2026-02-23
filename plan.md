# Plan: Korea Sector Rotation Dashboard

## Context

Greenfield implementation of a Streamlit-based Korean stock market sector rotation dashboard as specified in the PRD (`sector-rotation/AGENTS.md`). The project is currently empty (only has AGENTS.md and CLAUD.md). The implementation follows patterns established in sibling projects `stock-dashboard` (Streamlit SPA structure, config/, components/) and `Macro_Liquidity_Monitor` (data loader abstraction, regime classification, session state caching).

---

## Directory Structure to Create

```
sector-rotation/
├── app.py                          # Streamlit SPA entry point
├── requirements.txt
├── .streamlit/
│   ├── config.toml                 # Theme: D2Coding font, no-cyan palette
│   └── secrets.toml.example        # Template for API keys
├── config/
│   ├── sector_map.yml              # regime → index_code list mappings
│   ├── macro_series.yml            # ECOS/KOSIS series IDs + labels
│   └── settings.yml                # Default params (epsilon, windows, thresholds)
├── static/
│   └── fonts/                      # D2Coding.woff2 (to be placed by user)
├── data/
│   ├── raw/                        # data/raw/{source}/{dataset}/{yyyymmdd}.parquet
│   ├── curated/                    # data/curated/{dataset}.parquet
│   └── features/                   # data/features/{yyyymmdd}.parquet
├── src/
│   ├── contracts/
│   │   ├── validators.py          # validate_only + normalize_then_validate (R5)
│   │   └── data_contracts.md      # DataFrame schema documentation (R1)
│   ├── data_sources/
│   │   ├── krx_indices.py          # pykrx wrapper: retry, chunked fetch, fallback
│   │   ├── ecos.py                 # ECOS Open API (금리, 환율)
│   │   └── kosis.py               # KOSIS Open API (CPI, 수출)
│   ├── transforms/
│   │   ├── calendar.py            # last business day, KRX holiday handling
│   │   └── resample.py            # daily→monthly resample, 3MA direction
│   ├── indicators/
│   │   ├── momentum.py            # RS, SMA20/60 trend, period returns, vol, MDD
│   │   └── rsi.py                 # RSI (daily + weekly), TA-Lib → ta fallback
│   ├── macro/
│   │   └── regime.py              # 4-phase classification: Recovery/Expansion/Slowdown/Contraction
│   ├── signals/
│   │   ├── matrix.py              # macro_fit × momentum_state → action
│   │   └── scoring.py             # RSI alerts + FX shock filter
│   └── ui/
│       ├── styles.py              # D2Coding CSS injection, Plotly no-cyan template
│       ├── components.py          # Reusable: macro_tile, scatter_4q, heatmap, signal_table
│       └── data_status.py         # is_sample_mode + get_button_states pure functions
└── tests/
    ├── test_regime.py
    ├── test_momentum.py
    ├── test_signals.py
    ├── test_contracts.py
    ├── test_data_status.py
    └── test_integration.py
```

---

## Critical Risk Mitigations

### R1 — Data Contracts (Critical)

All module boundaries must enforce explicit dtype/index/null contracts. Add to `src/contracts/data_contracts.md`:

- Each DataFrame schema: exact column names, dtypes (e.g. `close: float64`, `is_provisional: bool`), index type (DatetimeIndex vs PeriodIndex), and null handling rule (e.g. "forward-fill max 3 periods, else drop row")
- Each function signature must document which contract it consumes/produces

### R2 — KRX Holiday Handling (Critical)

Weekend-only logic will misfire on KRX market holidays.
**Decision**: pykrx is the sole KRX calendar authority (see R6 below — no static list).

### R3 — Button-specific Cache Scope (High)

Each button's cache invalidation must be scoped exactly:

| Button | Deletes | Re-fetches |
|--------|---------|------------|
| 시장데이터 갱신 | `data/curated/sector_prices.parquet` | pykrx (live API) |
| 매크로데이터 갱신 | `data/curated/macro_monthly.parquet` | ECOS + KOSIS APIs |
| 전체 재계산 | `data/features/*.parquet` only | No API call, recomputes from curated |

See R8 for concrete implementation.

### R4 — Forced UI Warning on Synthetic Data (High)

When fallback level = "SAMPLE" (synthetic data), the dashboard MUST:

1. Show a full-width `st.error` banner at the TOP of the page
2. Disable the "전체 재계산" button (can't recompute from synthetic)
3. `load_dashboard_data()` returns `data_status` dict; caller checks via `is_sample_mode()`

### R5 — Contract Validation: Two-Function Split (Critical upgrade to R1)

Two separate functions in `src/contracts/validators.py`:

```python
# Type-family checks (not string literals) — handles nullable dtypes
from pandas.api.types import is_float_dtype, is_bool_dtype, is_object_dtype, is_datetime64_any_dtype

SCHEMAS = {
    "sector_prices": {
        "columns": {
            "index_code": is_object_dtype,
            "index_name": is_object_dtype,
            "close": is_float_dtype,       # matches float64, Float64, float32
        },
        "index_type": "DatetimeIndex",
        "non_null": ["index_code", "close"],
        "fill_policy": {"close": ("ffill", 3)},  # ffill max 3 → drop beyond
    },
    "macro_monthly": {
        "columns": {
            "series_id": is_object_dtype,
            "value": is_float_dtype,
            "source": is_object_dtype,
            "fetched_at": is_datetime64_any_dtype,
            "is_provisional": is_bool_dtype,   # matches bool, boolean (nullable)
        },
        "index_type": "PeriodIndex",
        "non_null": ["series_id", "source", "is_provisional"],
        "fill_policy": {},
    },
    "signals": {
        "columns": {
            "index_code": is_object_dtype,
            "macro_regime": is_object_dtype,
            "macro_fit": is_bool_dtype,
            "action": is_object_dtype,
            "is_provisional": is_bool_dtype,
        },
        "index_type": None,
        "non_null": ["index_code", "action", "macro_fit"],
        "fill_policy": {},
    },
}

def validate_only(df: pd.DataFrame, schema_name: str) -> None:
    """Read-only structural check. Raises ContractError if schema violated.
    Does NOT modify df. Use at module entry points to fail fast."""

def normalize_then_validate(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """Apply fill_policy (ffill + drop), then call validate_only.
    Returns new DataFrame. Use inside data loaders before saving to parquet."""
```

Call convention:

- Data loaders (`load_sector_prices`, `load_macro`): call `normalize_then_validate()` before saving
- Analytics functions (`build_signal_table`, `classify_regime`): call `validate_only()` at entry — receive already-normalized data

**`tests/test_contracts.py`** parametrized tests:

- Valid df: `validate_only` passes, `normalize_then_validate` returns df unchanged
- Wrong dtype (e.g. int column where float expected): `validate_only` raises `ContractError`
- Null in `non_null` column: raises `ContractError`
- Wrong index type: raises `ContractError`
- `ffill_max3`: gaps 1–3 filled; row dropped when gap > 3; `validate_only` on result passes

### R6 — KRX Holiday: pykrx Single Source, No Static List (Critical upgrade to R2)

**Unified policy**: `calendar.py` uses pykrx as the **sole** calendar authority. No static list anywhere in the codebase.

```python
def get_last_business_day(reference_date=None) -> date:
    """Return last KRX trading day. pykrx get_index_ohlcv handles all KRX holidays."""
    from pykrx import stock
    ref = reference_date or date.today()
    start = (ref - timedelta(days=14)).strftime("%Y%m%d")
    end = ref.strftime("%Y%m%d")
    try:
        df = stock.get_index_ohlcv(start, end, "1001")
        if df.empty:
            raise ValueError("Empty OHLCV response")
        return df.index[-1].date()
    except Exception:
        # Explicit weekend-only fallback — labeled, no false KRX holiday awareness
        d = ref
        for _ in range(5):
            d -= timedelta(days=1)
            if d.weekday() < 5:
                return d
        return ref - timedelta(days=1)
```

No static holiday table, no annual maintenance.

### R7 — Action Enum: N/A Included for Partial Failures (High)

**Decision**: Add `"N/A"` to the Action domain for sectors where data fetch failed.

```python
# src/signals/matrix.py
ACTION_VALUES = {"Strong Buy", "Watch", "Hold", "Avoid", "N/A"}
# "N/A" is only assigned when a sector's price data could not be loaded.
# It is never produced by the normal signal matrix logic.
```

Signal table UI renders N/A rows with grey styling and a "데이터 없음" tooltip.

### R8 — Concrete Cache Invalidation (High upgrade to R3)

Three named cache functions, each cleared individually:

```python
def _parquet_key(path: str) -> tuple:
    """Return (mtime_ns, size) for a parquet file, or (0, 0) if missing."""
    p = Path(path)
    if not p.exists():
        return (0, 0)
    s = p.stat()
    return (s.st_mtime_ns, s.st_size)   # ns precision + size avoids mtime collision

@st.cache_data(ttl=CACHE_TTL)
def _cached_sector_prices(asof_date_str, benchmark_code, price_years): ...

@st.cache_data(ttl=CACHE_TTL)
def _cached_macro(macro_series_hash): ...

@st.cache_data(ttl=CACHE_TTL)
def _cached_signals(prices_key, macro_key, params_hash):
    # prices_key = _parquet_key("data/curated/sector_prices.parquet")
    # macro_key  = _parquet_key("data/curated/macro_monthly.parquet")
    # params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    ...

# Button handlers — each clears only its own cache function:
if refresh_market:
    Path("data/curated/sector_prices.parquet").unlink(missing_ok=True)
    _cached_sector_prices.clear()
    st.rerun()

if refresh_macro:
    Path("data/curated/macro_monthly.parquet").unlink(missing_ok=True)
    _cached_macro.clear()
    st.rerun()

if recompute:
    shutil.rmtree("data/features", ignore_errors=True)
    Path("data/features").mkdir(exist_ok=True)
    _cached_signals.clear()
    st.rerun()
```

### R9 — SAMPLE Mode: Pure Function + Button Rules (High upgrade to R4)

```python
# src/ui/data_status.py
def is_sample_mode(data_status: dict) -> bool:
    return any(v == "SAMPLE" for v in data_status.values())

def get_button_states(data_status: dict) -> dict:
    sample = is_sample_mode(data_status)
    return {
        "refresh_market":  True,    # always enabled — attempts to escape SAMPLE
        "refresh_macro":   True,    # always enabled — attempts to escape SAMPLE
        "recompute":       not sample,  # disabled in SAMPLE (meaningless on synthetic data)
    }
```

SAMPLE mode button rules:

| Button | LIVE | CACHED | SAMPLE |
|--------|------|--------|--------|
| 시장데이터 갱신 | ✅ | ✅ | ✅ (escape attempt) |
| 매크로데이터 갱신 | ✅ | ✅ | ✅ (escape attempt) |
| 전체 재계산 | ✅ | ✅ | ❌ disabled |

`tests/test_data_status.py` — pure function tests (no Streamlit):

- `is_sample_mode({"price": "SAMPLE", "ecos": "LIVE"})` → True
- `is_sample_mode({"price": "LIVE", "ecos": "CACHED"})` → False
- `get_button_states({"price": "SAMPLE"})["recompute"]` → False
- `get_button_states({"price": "LIVE"})["recompute"]` → True

### R10 — Integration Tests with tmp_path Isolation (Medium)

`tests/test_integration.py` — all file I/O tests use pytest `tmp_path` + `monkeypatch`:

```python
def test_api_failure_falls_back_to_cache(tmp_path, monkeypatch):
    curated = tmp_path / "curated"
    curated.mkdir()
    monkeypatch.setattr("src.data_sources.krx_indices.CURATED_DIR", curated)
    # pre-seed parquet in tmp_path, mock pykrx to raise, assert CACHED returned
```

Five tests:

1. **`test_api_failure_falls_back_to_cache(tmp_path, monkeypatch)`**: mock pykrx → `ConnectionError`; pre-seed parquet; assert `("CACHED", df)`
2. **`test_full_fallback_to_sample(tmp_path, monkeypatch)`**: mock all APIs + empty curated dir; assert `("SAMPLE", df)`
3. **`test_partial_sector_failure_skips_gracefully(tmp_path, monkeypatch)`**: mock 2/10 fetches → raise; `build_signal_table` returns `list[SectorSignal]`; assert `len(signals) == 10`, `sum(1 for s in signals if s.action == "N/A") == 2`
4. **`test_cache_invalidation_clears_only_target(tmp_path)`**: seed both parquets; run market refresh; assert `sector_prices.parquet` gone, `macro_monthly.parquet` intact
5. **`test_fx_shock_e2e`**: pure logic (no file I/O); inject USD/KRW +4%; assert all `export_sector=True` "Strong Buy" → "Watch"

### R11 — Loader Return Contract + ECOS/KOSIS HTTP Policy (Medium)

**Loader return type — single contract across all data source modules:**

```python
DataStatus = Literal["LIVE", "CACHED", "SAMPLE"]
LoaderResult = tuple[DataStatus, pd.DataFrame]

# Every public loader returns (status, df):
def load_sector_prices(...) -> LoaderResult: ...
def load_ecos_macro(...) -> LoaderResult: ...
def load_kosis_macro(...) -> LoaderResult: ...
# Never return a dict or bare DataFrame — always (status, df)
```

**HTTP retry policy** — preserves last exception on final failure:

```python
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
BACKOFF_BASE = 2  # waits 2^(attempt+1) seconds: 2, 4, 8

def _get_with_retry(url: str) -> dict:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))
        except requests.HTTPError as e:
            if e.response.status_code in {429, 503}:
                last_exc = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_BASE ** (attempt + 1))
            else:
                raise  # 4xx client errors: don't retry
    assert last_exc is not None  # loop only exits without return if an exception was caught
    raise last_exc
```

---

## Implementation Phases

### Phase 1 — Project Scaffolding

**Files:** `requirements.txt`, `.streamlit/config.toml`, `config/*.yml`, `static/fonts/` placeholder

**requirements.txt** (key packages):

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
pykrx>=1.0.40
requests>=2.31.0
pyyaml>=6.0
ta>=0.11.0         # RSI/indicators fallback
pyarrow>=14.0.0    # parquet I/O
pytest>=7.4.0
pytest-mock>=3.12.0
# Optional (binary wheel):
# ta-lib>=0.4.28
```

**`.streamlit/config.toml`**: D2Coding font via `[[theme.fontFaces]]` + `server.enableStaticServing=true`. Primary color Navy (`#1B2A4A`), background Charcoal (`#1E1E2E`), no cyan anywhere.

**`config/sector_map.yml`** — exact schema (`export_sector` inline per entry, no separate config):

```yaml
benchmark:
  code: "1001"      # string, not int
  name: "KOSPI"

regimes:
  Recovery:
    description: "성장 Up, 물가/금리 Down"
    sectors:
      - code: "5044"
        name: "KRX 반도체"
        export_sector: true   # true → subject to FX shock downgrade
      - code: "1155"
        name: "KOSPI200 정보기술"
        export_sector: true
  Expansion:
    description: "성장 Up, 물가/금리 Up"
    sectors:
      - code: "5042"
        name: "KRX 산업재"
        export_sector: false
      - code: "1168"
        name: "KOSPI200 금융"
        export_sector: false
      - code: "1165"
        name: "KOSPI200 경기소비재"
        export_sector: false
  Slowdown:
    description: "성장 Down, 물가/금리 Up"
    sectors:
      - code: "5040"
        name: "KRX 에너지화학"
        export_sector: false
      - code: "5041"
        name: "KRX 소재"
        export_sector: true
      - code: "1170"
        name: "KOSPI200 유틸리티"
        export_sector: false
  Contraction:
    description: "성장 Down, 물가/금리 Down"
    sectors:
      - code: "5045"
        name: "KRX 헬스케어"
        export_sector: false
      - code: "1166"
        name: "KOSPI200 필수소비재"
        export_sector: false
      - code: "5046"
        name: "KRX 미디어통신"
        export_sector: false
```

`get_regime_sectors(regime, sector_map)` returns `list[dict]` with keys `code`, `name`, `export_sector`. No secondary lookup needed.

**`config/macro_series.yml`**: ECOS series IDs for 기준금리, 국고채3년, USD/KRW; KOSIS codes for CPI YoY, 수출증감률, 경기선행지수순환변동치.

**`config/settings.yml`**: Default params: `epsilon=0`, `rs_ma_period=20`, `ma_fast=20`, `ma_slow=60`, `rsi_period=14`, `rsi_overbought=70`, `rsi_oversold=30`, `fx_shock_pct=3.0`, `benchmark_code="1001"`.

---

### Phase 2 — Data Sources

#### `src/data_sources/krx_indices.py`

- `fetch_index_ohlcv(index_code, start, end) → pd.DataFrame`
  - Chunked fetching (≤2-year intervals per chunk, per pykrx issue #167)
  - Retry (3x, exponential backoff) + structured logging on failure
  - Saves to `data/raw/krx/{index_code}/{yyyymmdd}.parquet`
- `load_sector_prices(...) -> LoaderResult` — public loader, returns `(status, df)`
- `CURATED_DIR` module-level constant (monkeypatch target for tests)

#### `src/data_sources/ecos.py`

- `fetch_series(stat_code, item_code, start_ym, end_ym) → pd.DataFrame`
  - Uses ECOS Open API (key from `st.secrets["ECOS_API_KEY"]` or `os.environ`)
  - All HTTP via `_get_with_retry()` (timeout=10s, retry=3, backoff 2/4/8s)
- `load_ecos_macro(...) -> LoaderResult`
- Pre-defined helpers: `fetch_base_rate()`, `fetch_bond_3y()`, `fetch_usdkrw()`

#### `src/data_sources/kosis.py`

- Same pattern as ecos.py; same `_get_with_retry()` policy
- `load_kosis_macro(...) -> LoaderResult`
- CLI data marked with `is_provisional=True` for recent 3 months

#### `src/transforms/calendar.py`

- `get_last_business_day(as_of=None) → date` — uses pykrx `get_index_ohlcv` as sole KRX calendar source (R6); falls back to weekend-only subtraction labeled as best-effort
- No `is_krx_holiday()` — holiday awareness comes entirely from pykrx returned dates

#### `src/transforms/resample.py`

- `to_monthly_last(df_daily) → pd.DataFrame`
- `compute_3ma_direction(series, epsilon=0) → pd.Series` returns "Up"/"Down"/"Flat"

---

### Phase 3 — Core Analytics

#### `src/macro/regime.py`

```python
def classify_regime(growth_dir: str, inflation_dir: str) -> str:
    # Returns: "Recovery" | "Expansion" | "Slowdown" | "Contraction" | "Indeterminate"

def get_regime_sectors(regime: str, sector_map: dict) -> list[dict]:
    # Returns list of {code, name, export_sector} dicts for given regime

def compute_regime_history(growth_series, inflation_series, epsilon=0) -> pd.DataFrame:
    # Returns date-indexed DataFrame with columns: growth_dir, inflation_dir, regime
```

#### `src/indicators/momentum.py`

```python
def compute_rs(sector_close, benchmark_close) -> pd.Series
def compute_rs_ma(rs_series, period=20) -> pd.Series
def is_rs_strong(rs, rs_ma) -> bool   # rs > rs_ma
def compute_sma(series, window) -> pd.Series
def is_trend_positive(close, fast=20, slow=60) -> bool   # SMA20 > SMA60
def compute_period_returns(close) -> dict  # {1W, 1M, 3M, 6M, 12M}
def compute_volatility(close, window) -> float  # annualized
def compute_mdd(close, window) -> float
```

#### `src/indicators/rsi.py`

```python
def compute_rsi(close: pd.Series, period=14) -> pd.Series:
    # Try ta-lib first, fallback to ta library, fallback to manual Wilder smoothing
def compute_weekly_rsi(close: pd.Series, period=14) -> pd.Series:
    # Resample daily → weekly (last), then compute_rsi
```

#### `src/signals/matrix.py`

```python
ACTION_VALUES = {"Strong Buy", "Watch", "Hold", "Avoid", "N/A"}

@dataclass
class SectorSignal:
    index_code: str
    sector_name: str
    macro_regime: str
    macro_fit: bool
    rs: float
    rs_ma: float
    rs_strong: bool
    trend_ok: bool        # SMA20 > SMA60
    momentum_strong: bool # rs_strong AND trend_ok
    rsi_d: float
    rsi_w: float
    action: str           # "Strong Buy" | "Watch" | "Hold" | "Avoid" | "N/A" (data load failure only)
    alerts: list[str]     # ["Overheat", "Oversold", "FX Shock"]
    returns: dict         # {1M, 3M, 6M, 12M}
    volatility_20d: float
    mdd_3m: float
    asof_date: str
    is_provisional: bool

def compute_action(macro_fit: bool, momentum_strong: bool) -> str:
    # (True, True)→Strong Buy, (True, False)→Watch,
    # (False, True)→Hold, (False, False)→Avoid

def build_signal_table(
    sector_prices, benchmark_prices, macro_result,
    sector_map, settings
) -> list[SectorSignal]
```

#### `src/signals/scoring.py`

```python
def apply_rsi_alerts(signal: SectorSignal, overbought=70, oversold=30) -> SectorSignal
def apply_fx_shock_filter(
    signal: SectorSignal, fx_change_pct: float,
    export_sectors: list, threshold_pct=3.0
) -> SectorSignal
    # FX change > threshold AND export_sector=True → downgrade Strong Buy → Watch
```

---

### Phase 4 — Streamlit Dashboard

#### `src/ui/styles.py`

- `inject_css()`: loads D2Coding, injects via `st.markdown(unsafe_allow_html=True)`
- `get_plotly_template()`: custom colorway — NO cyan (`#00FFFF`, `#17BECF`, `#00BCD4` excluded)

#### `src/ui/components.py`

- `render_macro_tile(regime, growth_val, inflation_val, fx_change)`
- `render_rs_scatter(signals)` — Plotly 4-quadrant scatter
- `render_returns_heatmap(signals)` — sector × period heatmap
- `render_signal_table(signals, filter_action, filter_regime_only)` — N/A rows grey with tooltip

#### `app.py` — SPA Layout (uses R8 three-cache structure)

```
st.set_page_config(layout="wide")
inject_css()

# Sidebar: asof_date, benchmark, momentum sliders, 3 buttons
# btn_states from get_button_states(data_status) after load

# Three named @st.cache_data functions (R8):
#   _cached_sector_prices(asof_date_str, benchmark_code, price_years)
#   _cached_macro(macro_series_hash)
#   _cached_signals(prices_key, macro_key, params_hash)

# Button handlers clear ONLY their own cache function (R8)
# SAMPLE mode: st.error banner + recompute disabled (R9)

# TOP: Macro Summary (st.metric + Plotly gauge + provisional warning)
# MIDDLE: RS vs Trend scatter + cumulative return line + returns heatmap
# BOTTOM: Signal table with action/regime filters
```

---

### Phase 5 — Tests

#### `tests/test_regime.py` (3)

- `test_classify_regime_all_four_phases()`
- `test_classify_regime_flat_returns_indeterminate()`
- `test_compute_3ma_direction()`

#### `tests/test_momentum.py` (4)

- `test_rs_above_ma_is_strong()`
- `test_trend_positive_sma20_gt_sma60()`
- `test_rsi_bounds_0_100()`
- `test_weekly_rsi_resampling()`

#### `tests/test_signals.py` (3)

- `test_action_matrix_all_four_combinations()`
- `test_overheat_alert_added_when_rsi_above_70()`
- `test_fx_shock_downgrades_strong_buy()`

#### `tests/test_contracts.py` (5) — see R5

#### `tests/test_data_status.py` (4) — see R9

#### `tests/test_integration.py` (5) — see R10, all file I/O via `tmp_path`

---

## Critical Files to Create

| File | Purpose |
|------|---------|
| `app.py` | Streamlit SPA entry |
| `requirements.txt` | Dependencies |
| `.streamlit/config.toml` | D2Coding + theme (no cyan) |
| `config/sector_map.yml` | Regime→index_code mappings with export_sector |
| `config/macro_series.yml` | ECOS/KOSIS series IDs |
| `config/settings.yml` | Default algorithm parameters |
| `src/data_sources/krx_indices.py` | pykrx wrapper + LoaderResult |
| `src/data_sources/ecos.py` | ECOS API wrapper + LoaderResult |
| `src/data_sources/kosis.py` | KOSIS API wrapper + LoaderResult |
| `src/transforms/calendar.py` | pykrx-based business day (R6) |
| `src/transforms/resample.py` | 3MA + monthly resample |
| `src/macro/regime.py` | 4-phase classifier |
| `src/indicators/momentum.py` | RS, trend, returns |
| `src/indicators/rsi.py` | RSI with fallback chain |
| `src/signals/matrix.py` | SectorSignal dataclass + action matrix (5-value enum) |
| `src/signals/scoring.py` | RSI alerts + FX shock filter |
| `src/ui/styles.py` | CSS + Plotly no-cyan template |
| `src/ui/components.py` | Chart components |
| `src/ui/data_status.py` | `is_sample_mode` + `get_button_states` pure functions |
| `src/contracts/validators.py` | `validate_only` + `normalize_then_validate` |
| `src/contracts/data_contracts.md` | DataFrame schema documentation |
| `tests/test_regime.py` | Regime classifier tests (3) |
| `tests/test_momentum.py` | RS/trend/RSI indicator tests (4) |
| `tests/test_signals.py` | Action matrix + alert tests (3) |
| `tests/test_contracts.py` | Schema validation parametrized tests (5) |
| `tests/test_data_status.py` | Pure function tests (4) |
| `tests/test_integration.py` | API fallback + FX shock E2E tests (5) |

---

## Reusable Patterns from Sibling Projects

| Pattern | Source | Use |
|---------|--------|-----|
| `@st.cache_data(ttl=3600*6)` | stock-dashboard, Macro_Liquidity_Monitor | Named cache functions |
| `st.secrets["KEY"]` with env fallback | stock-dashboard | ECOS/KOSIS API keys |
| Dataclass-based config | Macro_Liquidity_Monitor `config.py` | `SectorSignal` dataclass |
| CSS injection via `st.markdown` | stock-dashboard `components/styles.py` | D2Coding font |
| Chunked API fetch with retry | Macro_Liquidity_Monitor loaders | pykrx long-range fetches |

---

## Verification

1. **Unit tests**: `pytest tests/ -v` — all 24 tests green (regime×3, momentum×4, signals×3, contracts×5, data_status×4, integration×5)
2. **Streamlit run**: `streamlit run app.py` — app loads in <10s on first run (local)
3. **Visual check**: No cyan/teal anywhere; D2Coding font visible in sidebar and table
4. **Regime check**: Change `epsilon` in sidebar → regime label updates
5. **Signal table**: Filter "Strong Buy" → only those sectors appear
6. **Provisional warning**: macro section shows "잠정치" notice
7. **FX filter**: Manually set FX change >3% in config → Strong Buy → Watch for export sectors

---

## API Keys Required (user action)

Before running, create `.streamlit/secrets.toml`:

```toml
ECOS_API_KEY = "your_key_here"   # from ecos.bok.or.kr/api
KOSIS_API_KEY = "your_key_here"  # from kosis.kr/openapi
```

Without keys, the dashboard will run with **sample/cached data** (graceful fallback with explicit UI warning).
