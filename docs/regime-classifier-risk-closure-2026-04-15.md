# Regime Classifier Risk Closure (2026-04-15)

- replay window: `2016-04` -> `2026-01`
- divergence: `33/118 = 28.0%`
- baseline lag0 fit: `18.2%`
- baseline lag1/PIT fit: `27.3%`
- baseline transition count: `27`
- latest confirmed regime: `Expansion`

## Current Accepted Experiment
- `carry_single_flat_regime`: `ACCEPTED`
- lag1/PIT: `27.3% -> 27.3%`
- changed confirmed months: `4/118 = 3.4%`

## Pre-Registered Candidate
- name: `flat_to_prior_nonflat_bridge`
- lookback: `3 months`
- affected months: `9`
- changed confirmed months vs current: `0/118 = 0.0%`
- candidate lag1/PIT fit: `27.3%`
- unexplained churn formula: `changed_action_months / comparable_replay_months`
- unexplained churn: `0/1298 = 0.0%`
- latest confirmed regime unchanged: `True`
- rationale: Candidate changes no confirmed-regime months versus the current accepted path.

## Decision
- Freeze classifier semantics now. The named pre-registration case is not strong enough to justify another classifier change on this fixed replay window.
- `lag0` nowcast and `lag1/PIT` remain explicitly separated.

## Reproduction
```bash
python scripts/evaluate_regime_validity.py --path dashboard-parity --asof 2026-04-15
```
