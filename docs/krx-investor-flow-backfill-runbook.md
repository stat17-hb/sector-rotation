# KRX Investor-Flow Historical Backfill Runbook

## 목적

이 문서는 현재 tracked-sector historical raw-backfill collector contract를 실제 KRX auth/network 환경에서 검증하고, full raw historical backfill을 운영하는 절차를 고정한다.

이 문서는 generic KRX raw endpoint viability를 다루지 않는다.
검증 대상은 현재 구현된 collector 경로다.

`tracked sectors -> constituent resolution -> ticker set -> raw investor-flow fetch -> raw fact persist`

## 전제 조건

- KRX auth/network가 실제로 동작하는 환경 1개
- `KRX_ID`, `KRX_PW`가 설정된 세션
- current repo checkout
- current product regression suite를 실행할 수 있는 Python 환경

## Phase 1 — Sample Validation

### 고정 규칙

- retry ceiling: `3`
- fixed backoff: `30초`
- no retry:
  - `AUTH_REQUIRED`
  - `ACCESS_DENIED`
  - schema-contract mismatch

### abort criteria

- 2개 연속 sample chunk fail
- sample chunk 하나라도 failed trading day ratio > `10%`
- sample chunk 하나라도 all-sector auth/contract hard failure

### 실행

```powershell
python scripts/backfill_investor_flow_history.py `
  --mode validate-samples `
  --market KR `
  --end-date <YYYYMMDD> `
  --chunk-business-days 20 `
  --retry-attempts 3 `
  --retry-sleep-sec 30 `
  --assume-non-regression-passed
```

옵션:

- 이미 live로 검증한 earliest date가 있으면 `--oldest-date <YYYYMMDD>`를 같이 줄 수 있다.

### pass 조건

- earliest 3 chunks
- middle 1 chunk
- recent 1 chunk

각 sample chunk가 모두 다음을 만족해야 한다.

- normalized raw rows persisted
- failed trading day ratio <= `5%` after retries
- unresolved auth/contract hard failure 없음

### `oldest_collectable_date` validation

다음을 모두 확인해야 한다.

1. current collector path로 earliest date discovery
2. discovery success는 completeness contract가 아니라 minimal existence contract다.
3. 즉 requested day와 일치하는 `trade_date` normalized raw row가 최소 1개 있어야 한다.
4. constituent `CACHED_FALLBACK(...)`에만 의존한 경우는 discovery 성공으로 보지 않는다.
5. corresponding backfill 후 warehouse `min(trade_date)` == validated earliest date

### non-regression

sample validation 후 아래 회귀 검증을 실행한다.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q `
  tests\test_warehouse_investor_flow.py `
  tests\test_krx_investor_flow_data_source.py `
  tests\test_dashboard_runtime.py `
  tests\test_dashboard_tabs.py `
  tests\test_data_status.py
```

추가 확인:

- app refresh semantics unchanged
- product read cap unchanged
- `historical_backfill` runs do not affect freshness/status messaging

## Phase 2 — Full Raw Historical Backfill

### 실행

```powershell
python scripts/backfill_investor_flow_history.py `
  --mode full `
  --market KR `
  --oldest-date <validated_earliest_yyyymmdd> `
  --end-date <YYYYMMDD> `
  --chunk-business-days 20 `
  --retry-attempts 3 `
  --retry-sleep-sec 30 `
  --assume-non-regression-passed
```

## Phase 2 closure standard

Risk 1은 아래를 모두 만족해야 closed다.

- backfill progress reaches target end date
- cumulative failed trading day ratio <= `2%` after retries
- unresolved hard failures 없음
- warehouse `min(trade_date)` == validated `oldest_collectable_date`
- current product non-regression checks pass

## Spot Checks

full run 후 아래 3개 범위를 점검한다.

- oldest range
- middle range
- recent range

각 range에서:

- row-count sanity
- expected ticker presence where applicable

## Failure Classification

- zero-row day with no explicit auth/contract error -> `data-gap failure`
- transport/timeout -> `transient transport failure`
- auth/access-denied/schema mismatch -> `hard failure`

## Risk 2 Gate

- historical sector design은 Phase 1 pass 후 가능
- historical sector implementation/pilot은 Phase 2 closure 후에만 가능

historical sector는 반드시 다음 계약을 먼저 고정해야 한다.

- separate materialized table
- date-aware constituent history source
- snapshot/version provenance
- correction/rerun/overwrite policy
- no automatic union with operational sector fact table
