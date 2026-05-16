# 변곡점/확률 중심 섹터 로테이션 대시보드 개선안

Date: 2026-05-16

## 결론

대시보드의 핵심 의사결정 문장은 다음처럼 바꾸는 것이 맞다.

- 상승 확률이 높고 손실 위험이 통제되는 섹터는 신규/증액 후보로 올린다.
- 하락 확률이 높거나 상승 확률 대비 손실 위험이 커진 섹터는 축소/회피 후보로 내린다.
- 기존 `Strong Buy / Watch / Hold / Avoid`는 최종 라벨로 유지하되, 그 앞단의 설명 변수를 `국면 적합도 × 모멘텀`에서 `상방 확률 × 하방 확률 × 변곡점 상태`로 확장한다.

현재 구현은 좋은 출발점이다. 이미 `src/signals/matrix.py`가 macro fit, relative strength, trend, hybrid momentum rank, RSI, flow overlay를 합쳐 액션을 만든다. 다만 사용자가 실제로 보고 싶은 질문인 "앞으로 오를 확률이 더 높은가, 내려갈 확률이 더 높은가"를 직접 수치화하지 않는다. 그래서 현 UI는 랭킹/액션 설명은 가능하지만, 진입/축소 타이밍을 확률적으로 비교하기 어렵다.

## 현재 구조 진단

현재 1차 신호:
- macro regime: `src/macro/regime.py`가 성장 방향과 물가 방향으로 Recovery, Expansion, Slowdown, Contraction을 분류한다.
- momentum: `src/indicators/momentum.py`와 `src/signals/matrix.py`가 RS, RS moving average, 20/60MA, 6M/12M ex-1M 상대수익률, 200DMA 필터를 계산한다.
- action matrix: KR은 momentum 중심, US/legacy는 macro fit과 momentum strong 조합으로 `Strong Buy / Watch / Hold / Avoid`를 만든다.
- flow overlay: `src/signals/flow.py`가 수급을 한 단계 upgrade/downgrade하는 보조 신호로 둔다.
- empirical fit: `src/signals/sector_fit.py`가 regime별 과거 초과수익 ranking artifact를 추가 설명으로 붙인다.

현재 UI:
- overview/decision-first stack은 현재 액션과 Top picks를 먼저 보여준다.
- analysis tab은 월간 절대수익률/초과수익률 heatmap과 cycle segment를 보여준다.
- flow tab은 KR 수급 또는 US ETF proxy flow를 보조 맥락으로 분리한다.

갭:
- "변곡점"이 명시적 상태값으로 없다.
- "상방 확률 / 하방 확률"이 calibrated probability로 없다.
- 현재 강한 섹터와 막 강해지는 섹터가 같은 화면/라벨에 섞인다.
- 회피 판단도 "왜 하락 확률이 커졌는지"보다 action label 중심이다.

## 외부 근거

섹터 로테이션은 단순 경기표만으로는 부족하고, 가격 모멘텀/상대강도/수급/거시 상태를 같이 봐야 한다.

- NBER의 Beber, Brandt, Kavajecz 연구는 섹터 간 포트폴리오 리밸런싱이 경기 상태에 대한 투자자 관점을 담고, 섹터 로테이션 footprint가 경제 변화와 미래 채권시장 수익률 예측에 정보를 가진다고 설명한다. Source: https://www.nber.org/papers/w16534
- Fidelity의 business-cycle sector framework는 섹터 성과가 경기 단계에 따라 회전하지만, 모든 사이클이 같지 않고 phase가 skip/retrace될 수 있다고 경고한다. 그래서 확정적 국면 매핑보다 probabilistic approach가 필요하다. Source: https://www.fidelity.com/webcontent/ap101883-markets_sectors-content/20.10.0/business_cycle/Business_Cycle_Sector_Approach_2020.pdf
- SSGA는 sector winner/loser 식별 접근으로 price momentum, macro indicator top-down, fundamental bottom-up을 함께 언급하고, business-cycle only rotation은 구현이 어렵다고 본다. Source: https://www.ssga.com/library-content/products/fund-docs/etfs/us/insights-investment-ideas/sector-business-cycle-analysis.pdf
- Moskowitz and Grinblatt의 industry momentum 연구는 산업/업종 모멘텀이 개별주 모멘텀 수익의 상당 부분을 설명한다고 본다. 현재 repo의 `hybrid_return_rank_v1` 방향과 맞다. Source: https://www.aqr.com/insights/research/journal-article/do-industries-explain-momentum

## 제안하는 신호 모델

섹터별로 매일/월말 다음 4개 값을 만든다.

1. `up_probability`
   - 다음 1M 또는 3M에 benchmark 대비 초과수익이 양수일 확률.
   - 후보 피처: 6M ex-1M relative return, 12M ex-1M relative return, RS-MA gap, RS slope, price/SMA200 gap, 20D volatility, 3M MDD, regime fit rank, flow z-score.

2. `down_probability`
   - 다음 1M 또는 3M에 benchmark 대비 초과수익이 특정 손실 임계값 이하일 확률.
   - 예: excess return < -3%, 또는 sector absolute return < -5%.
   - 후보 피처: RS breakdown, price below SMA200, volatility spike, MDD expansion, RSI overheat reversal, adverse flow, macro regime deterioration.

3. `turning_point_state`
   - `Bullish turn`, `Bearish turn`, `Continuation up`, `Continuation down`, `Neutral` 중 하나.
   - 핵심은 level이 아니라 변화율이다.
   - 예:
     - Bullish turn: RS below/near MA였던 섹터가 RS-MA gap 양전환 + RS slope 개선 + short momentum이 long momentum을 상회.
     - Bearish turn: RS-MA gap 음전환 + price trend 훼손 + flow adverse 또는 volatility expansion.

4. `edge_score`
   - `up_probability - down_probability`, 또는 risk-adjusted version.
   - 최종 액션은 이 값과 confidence로 결정한다.

권장 action mapping:
- `Strong Buy`: up >= 60%, down <= 30%, edge >= +25pp, data confidence high.
- `Watch`: up >= 50% 또는 Bullish turn이지만 confirmation 부족.
- `Hold`: 보유 섹터가 중립이거나 continuation up이지만 신규 매수 edge는 부족.
- `Avoid / Reduce`: down >= 50% 또는 Bearish turn 확인.
- `N/A`: 데이터 부족.

초기에는 머신러닝보다 walk-forward empirical calibration이 낫다.

- 각 월말 기준 피처를 만들고, 다음 1M/3M 초과수익 label을 붙인다.
- 피처를 분위수 bin으로 나눠 historical hit rate를 계산한다.
- 최근 n년 rolling window와 전체 window를 둘 다 저장한다.
- 확률은 `hit_rate`, 표본수, 최근성 가중치를 함께 보여준다.

이 방식은 설명 가능하고, 표본이 작아도 UI에서 "근거 빈약"을 드러낼 수 있다.

## 대시보드 개선안

### 1. 첫 화면을 "확률 보드"로 바꾸기

현재 Top picks를 유지하되, 컬럼을 다음처럼 바꾼다.

- Sector
- Action
- Up %
- Down %
- Edge
- Turning point
- Confidence
- Main evidence

사용자는 `Strong Buy`보다 `Up 64% / Down 22% / Edge +42pp`를 먼저 보게 된다.

### 2. 변곡점 전용 패널 추가

상단 또는 analysis tab에 `Rotation Inflection Board`를 둔다.

그룹:
- Emerging leaders: 아직 최상위 모멘텀은 아니지만 RS slope와 rank delta가 빠르게 개선.
- Deteriorating leaders: 여전히 상위권이지만 RS slope, flow, breadth가 꺾임.
- Confirmed leaders: 확률과 추세가 모두 강함.
- Avoid candidates: 하방 확률이 커지고 회복 근거가 약함.

핵심 컬럼:
- 1M rank change
- RS-MA gap change
- 20D RS slope
- 6M/12M momentum percentile
- flow state
- next 1M/3M hit rate

### 3. 섹터 상세 화면에 "상방/하방 근거 분해" 추가

섹터 클릭 시 다음을 보여준다.

- Probability gauge: Up, Down, Edge
- Bullish evidence: positive contributors
- Bearish evidence: negative contributors
- Trigger checklist:
  - RS crossed above/below MA
  - price above/below SMA200
  - rank improved/deteriorated by n places
  - flow supportive/adverse
  - RSI overheat/oversold
- Historical analogs:
  - 현재와 유사한 피처 조합의 과거 월말 사례 수
  - 다음 1M/3M 평균 초과수익
  - hit rate
  - worst drawdown

### 4. 국면 탭은 확정 label보다 "국면 전환 확률"로 개선

현재 regime은 confirmed regime label 중심이다. 이를 다음처럼 확장한다.

- Current regime probability: Recovery/Expansion/Slowdown/Contraction 각각 %
- Regime transition risk: 다음 국면 후보와 확률
- Sector implication: 현재 국면에서 유리한 섹터와, 다음 국면으로 넘어갈 때 유리해지는 섹터를 나눠 표시

처음에는 복잡한 macro model이 아니라 다음으로 충분하다.

- growth_dir/inflation_dir의 최근 3개월 방향 일관성
- direction magnitude z-score
- yield curve state
- policy/trade indicators availability
- raw regime과 confirmed regime mismatch 여부

### 5. 확률은 반드시 calibration 품질을 같이 보여주기

투자 확률은 위험하다. 그래서 화면에 다음 quality flag를 붙인다.

- sample count
- rolling window years
- Brier score 또는 calibration bucket error
- stale data 여부
- feature coverage

`Up 65%`만 보여주면 과신을 만든다. `Up 65%, n=42, calibration medium`처럼 보여야 한다.

## 구현 우선순위

### Phase 1: 확률 없이도 바로 가능한 UI 재정렬

- 기존 signal 필드로 pseudo probability를 만들지 말고, 우선 "edge proxy"와 "turning-point tags"만 계산한다.
- 추가 피처:
  - `rs_gap_pct`
  - `rs_gap_delta_20d`
  - `rs_slope_20d`
  - `mom_rank_delta_1m`
  - `price_sma200_gap_pct`
  - `risk_penalty`
- UI:
  - Top picks table에 `상승 근거`, `하락 경고`, `변곡 상태` 컬럼 추가.
  - Emerging/Deteriorating board 추가.

### Phase 2: historical hit-rate 확률

- `src/signals/probability.py` 추가.
- 월말 피처 snapshot 생성.
- next 1M/3M excess return label 생성.
- 분위수 bin별 hit rate와 drawdown risk 저장.
- `SectorSignal`에 `up_probability`, `down_probability`, `edge_score`, `probability_confidence`, `turning_point_state` 추가.
- 테스트:
  - lookahead bias 방지
  - label horizon alignment
  - 표본 부족 시 N/A
  - action mapping 고정

### Phase 3: calibrated model

- logistic regression 또는 isotonic calibration을 검토한다.
- 단, 표본수가 충분하기 전에는 모델보다 empirical binning을 우선한다.
- KR/US는 분리 학습한다.
- market regime별 nested calibration은 표본수 조건을 만족할 때만 켠다.

## 피해야 할 방향

- 확률을 임의 점수 스케일로 포장하지 않는다.
- `Strong Buy`를 더 화려하게 만드는 UI 개선에 머물지 않는다.
- 단일 국면 매핑을 절대 규칙처럼 보여주지 않는다.
- flow를 1차 신호로 격상하지 않는다. 현재 repo 문서의 결론처럼 수급은 conviction/timing overlay로 유지한다.
- 변곡점 신호를 너무 민감하게 만들지 않는다. 최소 2개 이상의 독립 신호가 같은 방향이어야 한다.

## 최종 권장 설계

최상단은 다음 3문장에 답해야 한다.

1. 지금 신규 매수 후보는 무엇인가?
   - `up_probability`, `edge_score`, `bullish turn/continuation` 기준.

2. 지금 줄여야 할 보유 섹터는 무엇인가?
   - `down_probability`, `bearish turn`, `risk penalty` 기준.

3. 어떤 변곡점이 새로 발생했는가?
   - rank delta, RS slope, trend break, flow shift를 근거로 표시.

현재 대시보드는 "현재 강한 섹터를 랭킹하는 도구"에 가깝다. 개선 목표는 "앞으로 유리한 방향으로 확률이 기울고 있는 섹터를 조기에 찾고, 불리한 확률로 기우는 섹터를 줄이는 의사결정 도구"가 되어야 한다.
