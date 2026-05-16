# 투자자 수급과 섹터 로테이션 중요도

Date: 2026-05-12

Source artifacts:
- Mission: `.omx/specs/autoresearch-investor-flow-sector-rotation/mission.md`
- Validation: `.omx/specs/autoresearch-investor-flow-sector-rotation/result.json`
- Original report: `.omx/specs/autoresearch-investor-flow-sector-rotation/report.md`

## 결론

투자자 수급은 섹터 로테이션에서 중요한 2차 신호다.
하지만 1차 결정 변수로 두기에는 노이즈와 반전 위험이 크다.

권장 위치:
- 1차: 섹터 상대강도, 추세, 변동성, 경기/금리/달러 같은 매크로 국면
- 2차: 외국인/기관/ETF flow 같은 수급
- 3차: 과열, 혼잡도, 유동성, 이벤트 리스크

실무 가중치 감각:
- 정량 모델: 수급 10-25%
- 재량 판단: "매수/회피를 결정"보다 "확신도와 타이밍을 조정"
- 데이터 신뢰도가 낮은 KR 비공식 수급: 5-15% 또는 reference-only

## 왜 1차 신호는 아닌가

섹터 로테이션의 강한 근거는 업종/섹터 수익률의 지속성이다.
Moskowitz and Grinblatt의 업종 모멘텀 연구는 과거 승자 업종을 사고 패자 업종을 파는 전략이 강한 성과를 보였고, 개별 종목 모멘텀 상당 부분을 업종 모멘텀이 설명한다고 본다.

즉 "어느 섹터가 이미 시장 대비 강한가"가 로테이션의 기본 축이다.
수급은 그 움직임이 실제 자금 이동으로 지지되는지 확인하는 역할에 더 가깝다.

## 왜 그래도 중요한가

펀드/ETF flow는 실제 가격 압력을 만든다는 연구가 많다.
ETF flow 연구는 flow shock과 관련된 가격 변화 중 일부가 단기 가격 압력이고 며칠 뒤 반전된다고 보고한다.
뮤추얼펀드 flow 연구도 aggregate flow가 시장 수익률과 동시에 움직이고, 가격 변화 일부가 10거래일 안에 되돌려진다는 증거를 제시한다.

따라서 수급은 두 가지를 알려준다.
- 추세 확인: 강한 섹터에 외국인/기관/ETF 자금이 붙는지
- 과열 경고: 너무 빠른 유입이 단기 가격 압력일 뿐인지

## 좋은 사용법

수급은 단독 매수 신호가 아니라 overlay로 쓰는 것이 낫다.

권장 규칙:
- 가격/RS가 강하고 수급도 순유입이면 conviction을 올린다.
- 가격은 강한데 수급이 약하면 추격 매수 비중을 줄인다.
- 가격은 약한데 수급만 강하면 관찰 후보로 둔다.
- 과도한 유입 뒤 수익률이 둔화되면 반전/혼잡 리스크로 본다.
- 개인 순매수 급증은 시장/섹터에 따라 contrarian 신호로 별도 검증한다.

현재 repo의 `src/signals/flow.py`는 이 방향과 대체로 맞다.
수급 프로필을 `foreign_lead`, `institutional_confirmation`, `contrarian_retail`로 나누고, base action을 한 단계만 조정한다.
테스트도 disabled overlay가 base action을 유지하도록 고정한다.

## KR 시장에서의 해석

KR에서는 외국인/기관/개인 수급이 정보 가치가 있을 수 있다.
특히 외국인 수급은 환율, 반도체/수출주, 글로벌 risk-on/off와 연결된다.
기관 수급은 연기금/투신 리밸런싱의 영향을 받을 수 있다.
개인 수급은 단기 과열 또는 반대매매/테마 쏠림의 흔적일 수 있다.

단, 이 repo의 KRX 수급 문서는 해당 수집 경로를 연구/프로토타입 수준으로 명시한다.
공식 Open API로 보장된 데이터가 아니고, 접근 차단/세션/정책 리스크가 있다.
그래서 KR 수급은 화면에 보여도 "최종 액션 반영 여부"를 분명히 구분해야 한다.

## 의사결정 룰

가장 실용적인 판단:
- 수급은 중요하다.
- 그러나 가격/상대강도/추세보다 위에 두면 안 된다.
- 수급이 price momentum과 같은 방향일 때만 신뢰도를 크게 올린다.
- 수급과 가격이 충돌하면 가격을 우선하고, 수급은 경고/대기 신호로 둔다.
- 데이터 신뢰도가 낮거나 stale이면 최종 액션에는 반영하지 않는다.

## Project Implications

현재 구현 방향은 유지하는 편이 타당하다.

- `src/signals/flow.py`: 수급은 profile별 overlay로 유지한다.
- `tests/test_investor_flow_scoring.py`: disabled overlay가 base action을 유지하는 계약은 중요하다.
- `docs/krx-unofficial-investor-flow.md`: KR 수급의 비공식/프로토타입 제약은 계속 사용자-facing copy와 액션 반영 조건에 반영되어야 한다.

향후 개선은 수급 신호 자체를 더 강하게 만드는 것보다, 신뢰도와 stale/reference-only 상태를 더 명확히 나누는 쪽이 우선이다.

## Sources

- Moskowitz, Tobias J. and Mark Grinblatt, "Do Industries Explain Momentum?", Journal of Finance, 1999. AQR summary: https://www.aqr.com/Insights/Research/Journal-Article/Do-Industries-Explain-Momentum
- RePEc abstract for "Do Industries Explain Momentum?": https://ideas.repec.org/a/bla/jfinan/v54y1999i4p1249-1290.html
- Staer, Arsenio, "Fund Flows and Underlying Returns: The Case of ETFs", 2017: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2158468
- Ben-Rephael, Azi, Shmuel Kandel, and Avi Wohl, "The Price Pressure of Aggregate Mutual Fund Flows", 2008: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1295986
- Ben-David, Itzhak, Jiacui Li, Andrea Rossi, and Yang Song, "Ratings-Driven Demand and Systematic Price Fluctuations", NBER Working Paper 28103: https://www.nber.org/papers/w28103
- Suominen, Matti and Eeli Tuovinen, "Price Pressures from Daily Mutual Fund Flows", 2025 working paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5378243
