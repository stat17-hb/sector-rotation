# 섹터-레짐 매핑 실증 검증 보고서 (dashboard-parity)

> 기준일: 2026-04-15
> path: `dashboard-parity`
> label column: `confirmed_regime`
> include provisional: `False`
> replay window: `2016-04` -> `2026-01` (118 points)
> Historical note: `docs/sector-mapping-validation.md`는 2026-02 historical snapshot입니다.

## Regime Counts
- Recovery 24 / Expansion 34 / Slowdown 27 / Contraction 29 / Indeterminate 4

## Ranking Tables

### Recovery
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

### Expansion
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

### Slowdown
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

### Contraction
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

## Cross-Regime Notes
- `1155` Recovery, Expansion, Contraction 상위권
- `1170` Recovery, Slowdown 상위권
- `5042` Recovery, Slowdown 상위권
- `5044` Recovery, Expansion, Contraction 상위권
- `5045` Expansion, Contraction 상위권
- `5048` Expansion, Contraction 상위권
- `5049` Expansion, Contraction 상위권

## Interpretation
- 이 표의 rank는 `lag0 nowcast empirical reference`입니다.
- `PIT` 기준 액션이나 매핑 변경 근거로 바로 쓰지 않습니다.
- static map은 별도 map gate를 통과하기 전까지 informational-only로 유지합니다.

## Reproduction
```bash
python scripts/validate_sector_mapping.py --path dashboard-parity --label-column confirmed_regime --asof 2026-04-15
```
