# 경기국면 판정 타당성 평가 보고서 (2026-02-25)

## 결론 (3줄)
- 최종 판정: **부분 타당** (D1~D4 합계 `2/4`).
- Point-in-time(1개월 지연) 적합률은 `2/8 = 25.0%`로 낮아, 시차 반영 시 성능 저하가 큽니다.
- 다만 Nowcast 기준 적합률은 `5/8 = 62.5%`이며, epsilon 0~0.1에서 최신 레짐은 `Expansion`으로 안정적입니다.

## 평가 범위 및 기준
- 범위: `국면 판정 로직 + 섹터 매핑 적합성`
- 기본 시나리오: Point-in-time (매크로 레짐 1개월 지연)
- 비교 시나리오: Nowcast(0개월), 스트레스(2개월)
- 기준일: `2026-02-25`
- 벤치마크: `KOSPI(1001)`
- 잠정치 정책: `is_provisional=True` 제외

## 시나리오별 핵심 지표
| 시나리오 | 관측월수 | 레짐 분포 (R/E/S/C) | 적합률(Top-half) |
|---|---:|---|---:|
| lag=0개월 | 31 | Recovery 3 / Expansion 19 / Slowdown 9 / Contraction 0 | 5/8 (62.5%) |
| lag=1개월 | 31 | Recovery 3 / Expansion 18 / Slowdown 10 / Contraction 0 | 2/8 (25.0%) |
| lag=2개월 | 31 | Recovery 3 / Expansion 17 / Slowdown 11 / Contraction 0 | 5/8 (62.5%) |

## Epsilon 민감도
| epsilon | 최신 레짐 | Indeterminate 비중 | 레짐 분포 (Recovery/Expansion/Slowdown/Contraction/Indeterminate) |
|---:|---|---:|---|
| 0.00 | Expansion | 12.1% | 3/23/25/0/7 |
| 0.05 | Expansion | 32.8% | 0/18/21/0/19 |
| 0.10 | Expansion | 53.4% | 0/11/16/0/31 |
| 0.20 | Indeterminate | 81.0% | 0/4/7/0/47 |

## 판정 근거 (D1~D4)
| 축 | 규칙 | 결과 | 점수 |
|---|---|---|---:|
| D1 표본 충족 | lag1에서 4개 레짐 모두 6개월 이상 | lag1 Contraction=0개월 | 0 |
| D2 PIT 강건성 | lag1>=50% and \|lag1-lag0\|<=20%p | lag0=62.5%, lag1=25.0% | 0 |
| D3 파라미터 강건성 | epsilon 0~0.1 최신 레짐 동일 + Indeterminate<=60% | 최신 레짐 동일(Expansion), 최대 53.4% | 1 |
| D4 Nowcast 유효성 | lag0 적합률 >= 60% | lag0=62.5% | 1 |

**최종 점수: 2/4 → 부분 타당**

## 보조 검증
- 레짐 전환 횟수: 전체 `16`회 / Indeterminate 제외 `10`회 (총 58포인트).
- 잠정치 포함/제외 비교: 포함 59포인트(최신 Expansion) vs 제외 58포인트(최신 Expansion).

## 리스크
- Contraction 관측월수가 0개월이라 표본 기반 검증이 불가능합니다.
- PIT 1개월 지연에서 적합률이 크게 하락해 실거래 관점 강건성이 약합니다.
- epsilon 상향 시 Indeterminate 비중이 급증해 운용 안정성이 저하될 수 있습니다.

## 개선안
### 단기(운영)
- 대시보드 기본 노출을 `PIT(1개월 지연)` 기준으로 전환하고 Nowcast는 참고 지표로 병기.
- Contraction 미관측 상태를 UI에 명시하고 해당 구간은 판정 유보 레이블 사용.
- 월말 리포트에 lag0/lag1 적합률 차이(누수 민감도)를 고정 KPI로 추가.

### 중기(모형)
- 데이터 기간을 확장해 Contraction 표본 확보 후 레짐별 재검증.
- 2지표(선행지수, CPI) 체계에서 금리/신용스프레드 보조지표를 포함한 합성 레짐 테스트.
- 고정 epsilon 대신 변동성 연동 임계치(rolling std 기반) 비교 실험.

## 시나리오별 레짐 순위표

### lag=0개월

#### Recovery
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 5045 | KRX 헬스케어 | +3.01 | Slowdown |
| 2 | 5046 | KRX 미디어통신 | +1.07 | Contraction |
| 3 | 1170 | KOSPI200 유틸리티 | +0.67 | Contraction |
| 4 | 5042 | KRX 산업재 | +0.62 | Expansion |
| 5 | 5044 | KRX 반도체 | +0.28 | Recovery |
| 6 | 1168 | KOSPI200 금융 | -0.28 | Expansion |
| 7 | 1165 | KOSPI200 경기소비재 | -0.47 | Expansion |
| 8 | 1155 | KOSPI200 정보기술 | -0.52 | Recovery |
| 9 | 1157 | KOSPI200 생활소비재 | -2.00 | Contraction |
| 10 | 5049 | KRX 철강 | -4.54 | Slowdown |
| 11 | 5048 | KRX 에너지화학 | -6.39 | Slowdown |

#### Expansion
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 5044 | KRX 반도체 | +2.07 | Recovery |
| 2 | 1155 | KOSPI200 정보기술 | +1.37 | Recovery |
| 3 | 5046 | KRX 미디어통신 | +1.21 | Contraction |
| 4 | 5042 | KRX 산업재 | +0.44 | Expansion |
| 5 | 1170 | KOSPI200 유틸리티 | +0.14 | Contraction |
| 6 | 1165 | KOSPI200 경기소비재 | +0.13 | Expansion |
| 7 | 1157 | KOSPI200 생활소비재 | +0.10 | Contraction |
| 8 | 5045 | KRX 헬스케어 | +0.02 | Slowdown |
| 9 | 1168 | KOSPI200 금융 | -0.97 | Expansion |
| 10 | 5049 | KRX 철강 | -1.02 | Slowdown |
| 11 | 5048 | KRX 에너지화학 | -1.48 | Slowdown |

#### Slowdown
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 5049 | KRX 철강 | +3.13 | Slowdown |
| 2 | 5044 | KRX 반도체 | +2.91 | Recovery |
| 3 | 5045 | KRX 헬스케어 | +1.82 | Slowdown |
| 4 | 5048 | KRX 에너지화학 | +0.43 | Slowdown |
| 5 | 1155 | KOSPI200 정보기술 | +0.12 | Recovery |
| 6 | 5042 | KRX 산업재 | +0.03 | Expansion |
| 7 | 1168 | KOSPI200 금융 | -0.53 | Expansion |
| 8 | 1170 | KOSPI200 유틸리티 | -0.67 | Contraction |
| 9 | 1165 | KOSPI200 경기소비재 | -1.51 | Expansion |
| 10 | 1157 | KOSPI200 생활소비재 | -1.99 | Contraction |
| 11 | 5046 | KRX 미디어통신 | -2.33 | Contraction |

#### Contraction
- 데이터 없음

### lag=1개월

#### Recovery
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 5046 | KRX 미디어통신 | +8.62 | Contraction |
| 2 | 5045 | KRX 헬스케어 | +6.03 | Slowdown |
| 3 | 1165 | KOSPI200 경기소비재 | +3.56 | Expansion |
| 4 | 1170 | KOSPI200 유틸리티 | +2.99 | Contraction |
| 5 | 1168 | KOSPI200 금융 | +2.52 | Expansion |
| 6 | 1157 | KOSPI200 생활소비재 | +0.48 | Contraction |
| 7 | 5042 | KRX 산업재 | +0.36 | Expansion |
| 8 | 5044 | KRX 반도체 | -2.78 | Recovery |
| 9 | 1155 | KOSPI200 정보기술 | -3.18 | Recovery |
| 10 | 5048 | KRX 에너지화학 | -3.79 | Slowdown |
| 11 | 5049 | KRX 철강 | -3.98 | Slowdown |

#### Expansion
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 5044 | KRX 반도체 | +2.33 | Recovery |
| 2 | 1155 | KOSPI200 정보기술 | +2.09 | Recovery |
| 3 | 5042 | KRX 산업재 | +0.53 | Expansion |
| 4 | 5045 | KRX 헬스케어 | +0.27 | Slowdown |
| 5 | 5046 | KRX 미디어통신 | -0.04 | Contraction |
| 6 | 1165 | KOSPI200 경기소비재 | -0.41 | Expansion |
| 7 | 1157 | KOSPI200 생활소비재 | -0.48 | Contraction |
| 8 | 1170 | KOSPI200 유틸리티 | -0.56 | Contraction |
| 9 | 5049 | KRX 철강 | -0.83 | Slowdown |
| 10 | 1168 | KOSPI200 금융 | -0.97 | Expansion |
| 11 | 5048 | KRX 에너지화학 | -1.90 | Slowdown |

#### Slowdown
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 5044 | KRX 반도체 | +1.18 | Recovery |
| 2 | 1170 | KOSPI200 유틸리티 | +1.17 | Contraction |
| 3 | 5049 | KRX 철강 | +0.90 | Slowdown |
| 4 | 5042 | KRX 산업재 | +0.08 | Expansion |
| 5 | 5046 | KRX 미디어통신 | -0.38 | Contraction |
| 6 | 1165 | KOSPI200 경기소비재 | -1.08 | Expansion |
| 7 | 5045 | KRX 헬스케어 | -1.15 | Slowdown |
| 8 | 1168 | KOSPI200 금융 | -1.15 | Expansion |
| 9 | 1155 | KOSPI200 정보기술 | -1.16 | Recovery |
| 10 | 5048 | KRX 에너지화학 | -1.44 | Slowdown |
| 11 | 1157 | KOSPI200 생활소비재 | -1.58 | Contraction |

#### Contraction
- 데이터 없음

### lag=2개월

#### Recovery
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 5045 | KRX 헬스케어 | +6.59 | Slowdown |
| 2 | 1165 | KOSPI200 경기소비재 | +2.97 | Expansion |
| 3 | 5044 | KRX 반도체 | +2.00 | Recovery |
| 4 | 1155 | KOSPI200 정보기술 | +1.64 | Recovery |
| 5 | 5046 | KRX 미디어통신 | +1.21 | Contraction |
| 6 | 5042 | KRX 산업재 | +0.70 | Expansion |
| 7 | 1168 | KOSPI200 금융 | +0.56 | Expansion |
| 8 | 5048 | KRX 에너지화학 | +0.29 | Slowdown |
| 9 | 1170 | KOSPI200 유틸리티 | -0.30 | Contraction |
| 10 | 1157 | KOSPI200 생활소비재 | -1.51 | Contraction |
| 11 | 5049 | KRX 철강 | -1.95 | Slowdown |

#### Expansion
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 1155 | KOSPI200 정보기술 | +1.58 | Recovery |
| 2 | 5044 | KRX 반도체 | +1.04 | Recovery |
| 3 | 5046 | KRX 미디어통신 | +0.88 | Contraction |
| 4 | 5045 | KRX 헬스케어 | +0.50 | Slowdown |
| 5 | 5042 | KRX 산업재 | +0.45 | Expansion |
| 6 | 1170 | KOSPI200 유틸리티 | +0.31 | Contraction |
| 7 | 1157 | KOSPI200 생활소비재 | +0.20 | Contraction |
| 8 | 1168 | KOSPI200 금융 | -0.13 | Expansion |
| 9 | 1165 | KOSPI200 경기소비재 | -0.22 | Expansion |
| 10 | 5049 | KRX 철강 | -1.23 | Slowdown |
| 11 | 5048 | KRX 에너지화학 | -2.85 | Slowdown |

#### Slowdown
| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |
|---:|---|---|---:|---|
| 1 | 5049 | KRX 철강 | +1.07 | Slowdown |
| 2 | 5044 | KRX 반도체 | +0.91 | Recovery |
| 3 | 1170 | KOSPI200 유틸리티 | +0.63 | Contraction |
| 4 | 5042 | KRX 산업재 | -0.04 | Expansion |
| 5 | 5048 | KRX 에너지화학 | -0.50 | Slowdown |
| 6 | 1165 | KOSPI200 경기소비재 | -0.51 | Expansion |
| 7 | 5045 | KRX 헬스케어 | -0.64 | Slowdown |
| 8 | 1157 | KOSPI200 생활소비재 | -0.86 | Contraction |
| 9 | 1168 | KOSPI200 금융 | -1.01 | Expansion |
| 10 | 5046 | KRX 미디어통신 | -1.15 | Contraction |
| 11 | 1155 | KOSPI200 정보기술 | -1.46 | Recovery |

#### Contraction
- 데이터 없음

## 재현 명령
```bash
PYTHONIOENCODING=utf-8 python scripts/evaluate_regime_validity.py --asof 2026-02-25
```
