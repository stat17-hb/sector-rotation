# 경기국면 판정 타당성 평가 보고서 (dashboard-parity) (2026-04-15)

> Authority source: pinned curated snapshot + this parity artifact
> `tasks/todo.md` is used only as a reconciliation checkpoint.

## Historical Note
- `docs/regime-validity-2026-02-25.md` and `docs/sector-mapping-validation.md` are historical snapshots from the older raw-regime path.
- This report uses the current dashboard-derived regime-construction rules on the current curated artifacts.

## Parity Snapshot
- path: `dashboard-parity`
- label column: `confirmed_regime`
- include provisional: `False`
- replay window: `2016-04` -> `2026-01` (118 points)
- primary counts: `Recovery 24 / Expansion 34 / Slowdown 27 / Contraction 29 / Indeterminate 4`

## Comparison
| Slice | Window | Counts |
|---|---|---|
| Primary `dashboard-parity` | 2016-04 -> 2026-01 | Recovery 24 / Expansion 34 / Slowdown 27 / Contraction 29 / Indeterminate 4 |
| Legacy raw path | 2016-04 -> 2026-01 | Recovery 25 / Expansion 30 / Slowdown 23 / Contraction 28 / Indeterminate 12 |
| Dashboard runtime 60M reference | 2021-03 -> 2026-02 | Recovery 19 / Expansion 8 / Slowdown 16 / Contraction 13 / Indeterminate 4 |

## Gate Summary
- raw vs confirmed divergence: `33/118 = 28.0%`
- lag0 fit: `81.8%`
- lag1 fit: `90.9%`
- Stage 1 gate: `OPEN`
  - confirmed-vs-raw divergence 28.0% > 15%
- Stage 1 experiment: `carry_single_flat_regime` enabled (ACCEPTED)
  - changed months: `4/118 = 3.4%`
  - lag1 fit: `27.3% -> 27.3%`
  - transitions: `25 -> 27`
  - latest confirmed regime stable: `True`

## Decision
- Stage 1R default posture: `freeze classifier semantics and improve reporting/wording only`.
- The only named exception is `flat_to_prior_nonflat_bridge`, and it should open only if pre-registration on this fixed replay window is strong enough.
- `lag0` nowcast and `lag1/PIT` evidence must stay separated.

## Lag Scenarios
| Scenario | Observed months | Counts (R/E/S/C/I) | Top-half fit |
|---|---:|---|---:|
| lag=0 | 114 | Recovery 24 / Expansion 34 / Slowdown 27 / Contraction 29 / Indeterminate 0 | 9/11 (81.8%) |
| lag=1 | 113 | Recovery 24 / Expansion 33 / Slowdown 27 / Contraction 29 / Indeterminate 0 | 10/11 (90.9%) |
| lag=2 | 112 | Recovery 24 / Expansion 32 / Slowdown 27 / Contraction 29 / Indeterminate 0 | 8/11 (72.7%) |

## Epsilon Sensitivity
| epsilon | latest confirmed | Indeterminate share | counts |
|---:|---|---:|---|
| 0.00 | Expansion | 3.4% | Recovery 24 / Expansion 34 / Slowdown 27 / Contraction 29 / Indeterminate 4 |
| 0.05 | Expansion | 5.1% | Recovery 19 / Expansion 33 / Slowdown 28 / Contraction 32 / Indeterminate 6 |
| 0.10 | Expansion | 10.2% | Recovery 23 / Expansion 34 / Slowdown 26 / Contraction 23 / Indeterminate 12 |
| 0.20 | Slowdown | 45.8% | Recovery 10 / Expansion 17 / Slowdown 20 / Contraction 17 / Indeterminate 54 |

## Ranking Tables

### lag=0

#### Recovery
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 1155 | KOSPI200 정보기술 | +2.07 | Recovery |
| 2 | 5044 | KRX 반도체 | +1.63 | Expansion |
| 3 | 5042 | KRX 산업재 | +0.39 | Expansion |
| 4 | 1168 | KOSPI200 금융 | +0.31 | Recovery |
| 5 | 1170 | KOSPI200 유틸리티 | -0.45 | Slowdown |
| 6 | 5045 | KRX 헬스케어 | -0.51 | Expansion |
| 7 | 1165 | KOSPI200 경기소비재 | -0.85 | Slowdown |
| 8 | 5048 | KRX 에너지화학 | -0.91 | Contraction |
| 9 | 5049 | KRX 철강 | -1.32 | Contraction |
| 10 | 1157 | KOSPI200 생활소비재 | -1.43 | Slowdown |
| 11 | 5046 | KRX 미디어통신 | -2.64 | Contraction |

#### Expansion
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 1155 | KOSPI200 정보기술 | +0.97 | Recovery |
| 2 | 5044 | KRX 반도체 | +0.94 | Expansion |
| 3 | 5049 | KRX 철강 | +0.28 | Contraction |
| 4 | 5045 | KRX 헬스케어 | +0.12 | Expansion |
| 5 | 5048 | KRX 에너지화학 | -0.02 | Contraction |
| 6 | 1168 | KOSPI200 금융 | -0.18 | Recovery |
| 7 | 5046 | KRX 미디어통신 | -0.23 | Contraction |
| 8 | 1165 | KOSPI200 경기소비재 | -0.57 | Slowdown |
| 9 | 5042 | KRX 산업재 | -0.74 | Expansion |
| 10 | 1170 | KOSPI200 유틸리티 | -1.02 | Slowdown |
| 11 | 1157 | KOSPI200 생활소비재 | -1.34 | Slowdown |

#### Slowdown
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 1170 | KOSPI200 유틸리티 | +2.44 | Slowdown |
| 2 | 5042 | KRX 산업재 | +1.00 | Expansion |
| 3 | 1157 | KOSPI200 생활소비재 | +0.93 | Slowdown |
| 4 | 1165 | KOSPI200 경기소비재 | +0.33 | Slowdown |
| 5 | 5046 | KRX 미디어통신 | +0.21 | Contraction |
| 6 | 1155 | KOSPI200 정보기술 | +0.03 | Recovery |
| 7 | 1168 | KOSPI200 금융 | -0.27 | Recovery |
| 8 | 5044 | KRX 반도체 | -0.44 | Expansion |
| 9 | 5049 | KRX 철강 | -0.49 | Contraction |
| 10 | 5048 | KRX 에너지화학 | -0.70 | Contraction |
| 11 | 5045 | KRX 헬스케어 | -1.96 | Expansion |

#### Contraction
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 5045 | KRX 헬스케어 | +1.45 | Expansion |
| 2 | 5044 | KRX 반도체 | +1.04 | Expansion |
| 3 | 5049 | KRX 철강 | +0.58 | Contraction |
| 4 | 1155 | KOSPI200 정보기술 | +0.15 | Recovery |
| 5 | 5048 | KRX 에너지화학 | +0.04 | Contraction |
| 6 | 5042 | KRX 산업재 | -0.01 | Expansion |
| 7 | 1168 | KOSPI200 금융 | -0.26 | Recovery |
| 8 | 1165 | KOSPI200 경기소비재 | -0.27 | Slowdown |
| 9 | 5046 | KRX 미디어통신 | -1.35 | Contraction |
| 10 | 1157 | KOSPI200 생활소비재 | -1.77 | Slowdown |
| 11 | 1170 | KOSPI200 유틸리티 | -3.41 | Slowdown |

### lag=1

#### Recovery
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 1155 | KOSPI200 정보기술 | +1.81 | Recovery |
| 2 | 1170 | KOSPI200 유틸리티 | +1.25 | Slowdown |
| 3 | 1168 | KOSPI200 금융 | +1.06 | Recovery |
| 4 | 5044 | KRX 반도체 | +0.75 | Expansion |
| 5 | 5049 | KRX 철강 | +0.32 | Contraction |
| 6 | 1157 | KOSPI200 생활소비재 | -0.16 | Slowdown |
| 7 | 1165 | KOSPI200 경기소비재 | -0.33 | Slowdown |
| 8 | 5045 | KRX 헬스케어 | -0.53 | Expansion |
| 9 | 5042 | KRX 산업재 | -0.65 | Expansion |
| 10 | 5048 | KRX 에너지화학 | -0.88 | Contraction |
| 11 | 5046 | KRX 미디어통신 | -1.11 | Contraction |

#### Expansion
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 5044 | KRX 반도체 | +1.56 | Expansion |
| 2 | 1155 | KOSPI200 정보기술 | +1.28 | Recovery |
| 3 | 5045 | KRX 헬스케어 | +0.21 | Expansion |
| 4 | 5042 | KRX 산업재 | -0.37 | Expansion |
| 5 | 5048 | KRX 에너지화학 | -0.55 | Contraction |
| 6 | 5046 | KRX 미디어통신 | -0.69 | Contraction |
| 7 | 5049 | KRX 철강 | -0.70 | Contraction |
| 8 | 1165 | KOSPI200 경기소비재 | -0.84 | Slowdown |
| 9 | 1168 | KOSPI200 금융 | -0.91 | Recovery |
| 10 | 1157 | KOSPI200 생활소비재 | -1.99 | Slowdown |
| 11 | 1170 | KOSPI200 유틸리티 | -2.53 | Slowdown |

#### Slowdown
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 1170 | KOSPI200 유틸리티 | +1.92 | Slowdown |
| 2 | 5042 | KRX 산업재 | +0.96 | Expansion |
| 3 | 1157 | KOSPI200 생활소비재 | +0.51 | Slowdown |
| 4 | 1155 | KOSPI200 정보기술 | +0.05 | Recovery |
| 5 | 1165 | KOSPI200 경기소비재 | -0.01 | Slowdown |
| 6 | 5049 | KRX 철강 | -0.03 | Contraction |
| 7 | 5046 | KRX 미디어통신 | -0.15 | Contraction |
| 8 | 1168 | KOSPI200 금융 | -0.32 | Recovery |
| 9 | 5044 | KRX 반도체 | -0.50 | Expansion |
| 10 | 5048 | KRX 에너지화학 | -1.04 | Contraction |
| 11 | 5045 | KRX 헬스케어 | -1.25 | Expansion |

#### Contraction
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 5044 | KRX 반도체 | +1.26 | Expansion |
| 2 | 5048 | KRX 에너지화학 | +0.95 | Contraction |
| 3 | 5045 | KRX 헬스케어 | +0.87 | Expansion |
| 4 | 5042 | KRX 산업재 | +0.34 | Expansion |
| 5 | 5049 | KRX 철강 | -0.01 | Contraction |
| 6 | 1168 | KOSPI200 금융 | -0.09 | Recovery |
| 7 | 1165 | KOSPI200 경기소비재 | -0.14 | Slowdown |
| 8 | 1155 | KOSPI200 정보기술 | -0.15 | Recovery |
| 9 | 1157 | KOSPI200 생활소비재 | -1.56 | Slowdown |
| 10 | 5046 | KRX 미디어통신 | -1.61 | Contraction |
| 11 | 1170 | KOSPI200 유틸리티 | -2.45 | Slowdown |

### lag=2

#### Recovery
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 1168 | KOSPI200 금융 | +1.69 | Recovery |
| 2 | 5049 | KRX 철강 | +1.12 | Contraction |
| 3 | 1155 | KOSPI200 정보기술 | +0.53 | Recovery |
| 4 | 5044 | KRX 반도체 | +0.22 | Expansion |
| 5 | 1165 | KOSPI200 경기소비재 | +0.04 | Slowdown |
| 6 | 5045 | KRX 헬스케어 | +0.01 | Expansion |
| 7 | 1170 | KOSPI200 유틸리티 | -0.51 | Slowdown |
| 8 | 5042 | KRX 산업재 | -0.57 | Expansion |
| 9 | 1157 | KOSPI200 생활소비재 | -0.73 | Slowdown |
| 10 | 5046 | KRX 미디어통신 | -0.73 | Contraction |
| 11 | 5048 | KRX 에너지화학 | -1.20 | Contraction |

#### Expansion
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 5045 | KRX 헬스케어 | +1.83 | Expansion |
| 2 | 1155 | KOSPI200 정보기술 | +1.69 | Recovery |
| 3 | 5044 | KRX 반도체 | +1.64 | Expansion |
| 4 | 5048 | KRX 에너지화학 | -0.09 | Contraction |
| 5 | 5042 | KRX 산업재 | -0.16 | Expansion |
| 6 | 5049 | KRX 철강 | -1.24 | Contraction |
| 7 | 1168 | KOSPI200 금융 | -1.47 | Recovery |
| 8 | 1165 | KOSPI200 경기소비재 | -1.56 | Slowdown |
| 9 | 5046 | KRX 미디어통신 | -1.70 | Contraction |
| 10 | 1170 | KOSPI200 유틸리티 | -1.78 | Slowdown |
| 11 | 1157 | KOSPI200 생활소비재 | -2.43 | Slowdown |

#### Slowdown
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 1170 | KOSPI200 유틸리티 | +1.80 | Slowdown |
| 2 | 5042 | KRX 산업재 | +1.20 | Expansion |
| 3 | 1157 | KOSPI200 생활소비재 | +0.82 | Slowdown |
| 4 | 1168 | KOSPI200 금융 | +0.30 | Recovery |
| 5 | 5049 | KRX 철강 | +0.20 | Contraction |
| 6 | 1155 | KOSPI200 정보기술 | +0.07 | Recovery |
| 7 | 5046 | KRX 미디어통신 | +0.03 | Contraction |
| 8 | 1165 | KOSPI200 경기소비재 | -0.16 | Slowdown |
| 9 | 5044 | KRX 반도체 | -0.35 | Expansion |
| 10 | 5048 | KRX 에너지화학 | -0.40 | Contraction |
| 11 | 5045 | KRX 헬스케어 | -1.00 | Expansion |

#### Contraction
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 5044 | KRX 반도체 | +1.41 | Expansion |
| 2 | 1155 | KOSPI200 정보기술 | +0.48 | Recovery |
| 3 | 1165 | KOSPI200 경기소비재 | +0.41 | Slowdown |
| 4 | 5048 | KRX 에너지화학 | +0.01 | Contraction |
| 5 | 5042 | KRX 산업재 | -0.16 | Expansion |
| 6 | 5049 | KRX 철강 | -0.30 | Contraction |
| 7 | 1168 | KOSPI200 금융 | -0.57 | Recovery |
| 8 | 1157 | KOSPI200 생활소비재 | -0.99 | Slowdown |
| 9 | 5046 | KRX 미디어통신 | -1.11 | Contraction |
| 10 | 5045 | KRX 헬스케어 | -1.48 | Expansion |
| 11 | 1170 | KOSPI200 유틸리티 | -1.59 | Slowdown |

## Reproduction
```bash
PYTHONIOENCODING=utf-8 python scripts/evaluate_regime_validity.py --path dashboard-parity --asof 2026-04-15
```
