# Momentum Method Comparison (2026-05-15)

## Inputs
- market_id: `KR`
- benchmark_code: `1001`
- methods: `legacy_rs_ma_v0`, `hybrid_return_rank_v1`
- window_months: `36`
- momentum_skip_recent_days: `0`
- momentum_lookback_6m_days: `126`
- momentum_lookback_12m_days: `252`
- momentum_rank_threshold_pct: `0.6`

## Metrics
- legacy whipsaws: `96`
- hybrid whipsaws: `15`
- whipsaw reduction: `84.4%`
- hybrid median adjacent-month Spearman rho: `0.91`
- hybrid 10th-percentile rho: `0.63`
- pass_whipsaw: `True`
- pass_rank_stability: `True`

## Monthly Eligible Sector Counts
- 2023-06-30 | legacy_rs_ma_v0 | eligible sectors: 11
- 2023-07-31 | legacy_rs_ma_v0 | eligible sectors: 11
- 2023-08-31 | legacy_rs_ma_v0 | eligible sectors: 11
- 2023-09-27 | legacy_rs_ma_v0 | eligible sectors: 11
- 2023-10-31 | legacy_rs_ma_v0 | eligible sectors: 11
- 2023-11-30 | legacy_rs_ma_v0 | eligible sectors: 11
- 2023-12-28 | legacy_rs_ma_v0 | eligible sectors: 11
- 2024-01-31 | legacy_rs_ma_v0 | eligible sectors: 11
- 2024-02-29 | hybrid_return_rank_v1 | eligible sectors: 11
- 2024-02-29 | legacy_rs_ma_v0 | eligible sectors: 11
- 2024-03-29 | hybrid_return_rank_v1 | eligible sectors: 11
- 2024-03-29 | legacy_rs_ma_v0 | eligible sectors: 11
- 2024-04-30 | hybrid_return_rank_v1 | eligible sectors: 11
- 2024-04-30 | legacy_rs_ma_v0 | eligible sectors: 11
- 2024-05-31 | hybrid_return_rank_v1 | eligible sectors: 11
- 2024-05-31 | legacy_rs_ma_v0 | eligible sectors: 11
- 2024-06-28 | hybrid_return_rank_v1 | eligible sectors: 11
- 2024-06-28 | legacy_rs_ma_v0 | eligible sectors: 11
- 2024-07-31 | hybrid_return_rank_v1 | eligible sectors: 11
- 2024-07-31 | legacy_rs_ma_v0 | eligible sectors: 11
- 2024-08-30 | hybrid_return_rank_v1 | eligible sectors: 11
- 2024-08-30 | legacy_rs_ma_v0 | eligible sectors: 11
- 2024-09-30 | hybrid_return_rank_v1 | eligible sectors: 11
- 2024-09-30 | legacy_rs_ma_v0 | eligible sectors: 11
- 2024-10-31 | hybrid_return_rank_v1 | eligible sectors: 11
- 2024-10-31 | legacy_rs_ma_v0 | eligible sectors: 11
- 2024-11-29 | hybrid_return_rank_v1 | eligible sectors: 11
- 2024-11-29 | legacy_rs_ma_v0 | eligible sectors: 11
- 2024-12-30 | hybrid_return_rank_v1 | eligible sectors: 11
- 2024-12-30 | legacy_rs_ma_v0 | eligible sectors: 11
- 2025-01-31 | hybrid_return_rank_v1 | eligible sectors: 11
- 2025-01-31 | legacy_rs_ma_v0 | eligible sectors: 11
- 2025-02-28 | hybrid_return_rank_v1 | eligible sectors: 11
- 2025-02-28 | legacy_rs_ma_v0 | eligible sectors: 11
- 2025-03-31 | hybrid_return_rank_v1 | eligible sectors: 11
- 2025-03-31 | legacy_rs_ma_v0 | eligible sectors: 11
- 2025-04-30 | hybrid_return_rank_v1 | eligible sectors: 11
- 2025-04-30 | legacy_rs_ma_v0 | eligible sectors: 11
- 2025-05-30 | hybrid_return_rank_v1 | eligible sectors: 11
- 2025-05-30 | legacy_rs_ma_v0 | eligible sectors: 11
- 2025-06-30 | hybrid_return_rank_v1 | eligible sectors: 11
- 2025-06-30 | legacy_rs_ma_v0 | eligible sectors: 11
- 2025-07-31 | hybrid_return_rank_v1 | eligible sectors: 11
- 2025-07-31 | legacy_rs_ma_v0 | eligible sectors: 11
- 2025-08-29 | hybrid_return_rank_v1 | eligible sectors: 11
- 2025-08-29 | legacy_rs_ma_v0 | eligible sectors: 11
- 2025-09-30 | hybrid_return_rank_v1 | eligible sectors: 11
- 2025-09-30 | legacy_rs_ma_v0 | eligible sectors: 11
- 2025-10-31 | hybrid_return_rank_v1 | eligible sectors: 11
- 2025-10-31 | legacy_rs_ma_v0 | eligible sectors: 11
- 2025-11-28 | hybrid_return_rank_v1 | eligible sectors: 11
- 2025-11-28 | legacy_rs_ma_v0 | eligible sectors: 11
- 2025-12-30 | hybrid_return_rank_v1 | eligible sectors: 11
- 2025-12-30 | legacy_rs_ma_v0 | eligible sectors: 11
- 2026-01-30 | hybrid_return_rank_v1 | eligible sectors: 11
- 2026-01-30 | legacy_rs_ma_v0 | eligible sectors: 11
- 2026-02-27 | hybrid_return_rank_v1 | eligible sectors: 11
- 2026-02-27 | legacy_rs_ma_v0 | eligible sectors: 11
- 2026-03-31 | hybrid_return_rank_v1 | eligible sectors: 11
- 2026-03-31 | legacy_rs_ma_v0 | eligible sectors: 11
- 2026-04-30 | hybrid_return_rank_v1 | eligible sectors: 11
- 2026-04-30 | legacy_rs_ma_v0 | eligible sectors: 11
- 2026-05-15 | hybrid_return_rank_v1 | eligible sectors: 11
- 2026-05-15 | legacy_rs_ma_v0 | eligible sectors: 11

## Decision
- Hybrid cutover is approved.

## Reproduction
```bash
python scripts/evaluate_momentum_method.py --market KR --benchmark-code 1001 --asof 2026-05-15 --window-months 36 --update-current
```
