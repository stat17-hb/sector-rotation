# Sector-Fit Risk Closure (2026-04-15)

- replay window: `2016-04` -> `2026-01` (118 points)
- label column: `confirmed_regime`
- include provisional: `False`
- static map top-half hits: `9/11 = 81.8%`
- overlap_rate: `16/20 = 0.80`

## Sample Counts
- Recovery: `24`
- Expansion: `34`
- Slowdown: `27`
- Contraction: `29`

## Shared Leaders
- `1155`
- `1170`
- `5042`
- `5044`
- `5045`
- `5048`
- `5049`

## Lag1 Candidate Map
- Recovery: `1155, 1168`
- Expansion: `5042, 5044, 5045`
- Slowdown: `1157, 1165, 1170`
- Contraction: `5046, 5048, 5049`
- candidate lag1 top-half hits: `10/11 = 90.9%`
- candidate overlap_rate: `0.85`
- current mapping matches candidate: `True`

## Decision
- Runtime empirical fit remains `lag0 nowcast empirical reference` only.
- Static mapping is evaluated on the current canonical confirmed-regime path.
- Map gate: `current mapping accepted`
- Reason: Current mapping matches the reproducible lag1 candidate and that candidate achieves 10/11 top-half hits with overlap_rate 0.85.
- Runtime empirical fit remains informational-only even when the static map improves materially.

## Reproduction
```bash
python scripts/validate_sector_mapping.py --path dashboard-parity --label-column confirmed_regime --asof 2026-04-15
```
