# Data Contracts

## Overview

All public data loader functions return `LoaderResult = tuple[DataStatus, pd.DataFrame]`.
Analytics entry points receive already-normalized DataFrames.

## DataStatus

```python
DataStatus = Literal["LIVE", "CACHED", "SAMPLE"]
LoaderResult = tuple[DataStatus, pd.DataFrame]
```

---

## Schema: sector_prices

**Produced by**: `load_sector_prices()`
**Consumed by**: `build_signal_table()`

| Column | Dtype | Nullable |
|--------|-------|----------|
| index_code | object (str) | No |
| index_name | object (str) | Yes |
| close | float64 | No (after fill) |

- **Index**: DatetimeIndex (daily, KRX business days)
- **Null policy**: `close` — ffill max 3 periods, then drop remaining nulls
- **Fill convention**: `normalize_then_validate()` applies fill before saving to parquet

---

## Schema: macro_monthly

**Produced by**: `load_ecos_macro()`, `load_kosis_macro()`
**Consumed by**: `classify_regime()`, `compute_regime_history()`

| Column | Dtype | Nullable |
|--------|-------|----------|
| series_id | object (str) | No |
| value | float64 | Yes |
| source | object (str) | No |
| fetched_at | datetime64[ns] | No |
| is_provisional | bool | No |

- **Index**: PeriodIndex (monthly, freq='M')
- **Null policy**: no ffill — missing macro values kept as NaN, caller handles
- **Provisional flag**: last 3 months from KOSIS marked `is_provisional=True`

---

## Schema: signals

**Produced by**: `build_signal_table()`
**Consumed by**: UI components

| Column | Dtype | Nullable |
|--------|-------|----------|
| index_code | object (str) | No |
| macro_regime | object (str) | No |
| macro_fit | bool | No |
| action | object (str) | No |
| is_provisional | bool | No |

- **Index**: RangeIndex (default)
- **Action domain**: `{"Strong Buy", "Watch", "Hold", "Avoid", "N/A"}`
- `"N/A"` only when sector price data could not be loaded
- **Null policy**: no nulls allowed in `index_code`, `action`, `macro_fit`

---

## Call Convention

- **Loaders** (`load_sector_prices`, `load_ecos_macro`, `load_kosis_macro`):
  Call `normalize_then_validate(df, schema_name)` before saving to parquet.

- **Analytics** (`build_signal_table`, `classify_regime`):
  Call `validate_only(df, schema_name)` at entry — data is already normalized.
