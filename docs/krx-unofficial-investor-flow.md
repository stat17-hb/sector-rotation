# KRX 비공식 투자자 수급 수집 노트

## 상태

- 목적: KRX 웹 통계 화면 뒤의 비공식 백엔드 호출을 **연구/프로토타입 수준**에서 재현
- 현재 구현물:
  - `python/collectors/krx_investor_flow/`
  - `scripts/krx_investor_flow_probe.py`
  - `tests/krx/krx_investor_flow/`
- 운영 판정: **adopt-with-caution 후보**
  - 기술적 재현 가능성은 높아 보이지만
  - 공식 Open API로 보장되지 않고
  - 정책/안정성 리스크가 큼

## 확인한 근거

### 공식 페이지
- KRX OPEN API 메인: `https://openapi.krx.co.kr/contents/OPP/MAIN/main/index.cmd`
- KRX OPEN API 서비스 목록: `https://openapi.krx.co.kr/contents/OPP/INFO/service/OPPINFO004.cmd`
- KRX Data Marketplace 메인: `https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd`
- KRX 안내문(disclaimer): `https://data.krx.co.kr/templets/mdc/disclaimer_p1.jsp`

### 공식/공식에 가까운 사실
- Data Marketplace에는 `주식 > 거래실적 > 투자자별 거래실적`, `투자자별 거래실적(개별종목)` 메뉴가 공개되어 있다.
- 공개 Open API 카탈로그에서는 같은 범주의 표준 공개 API가 명확하지 않다.
- KRX 안내문은 웹 데이터가 `AS IS`로 제공되며, 사전 허가 없는 재생산/전송/배포를 금지하고, 네트워크 절차/정책을 방해하지 말라고 명시한다.

### 비공식 호출 가설
- 공개된 브라우저 트레이스 분석에서 아래 패턴이 반복적으로 관찰된다.
  - Endpoint: `POST https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd`
  - Market-level `bld`: `dbms/MDC/STAT/standard/MDCSTAT02202`
  - Stock-level `bld` 후보:
    - `dbms/MDC/STAT/standard/MDCSTAT02302`
    - `dbms/MDC/STAT/표준/MDCSTAT02302`
- 응답은 JSON이며 `block1` 배열 형태일 가능성이 높다.

## 구현 원칙

1. **probe layer와 normalize layer를 분리**한다.
2. KRX 호출 가정(엔드포인트, `bld`, 헤더)은 `config.py`에 고정한다.
3. 운영 채택 전에는 반드시 다음을 재검증한다.
   - referer / origin / XHR header 의존성
   - 쿠키/세션 의존성
   - OTP 또는 종목코드 finder 선행 조회 필요 여부
   - 빈 응답 / HTML 에러 / 차단 응답 패턴

## 예시 사용법

### 시장 단위

```bash
python scripts/krx_investor_flow_probe.py \
  --start-date 20260401 \
  --end-date 20260410 \
  market \
  --market-id ALL
```

### 개별 종목 단위

```bash
python scripts/krx_investor_flow_probe.py \
  --start-date 20260401 \
  --end-date 20260410 \
  stock \
  --isu-cd KR7005930003 \
  --ticker 005930
```

## 운영 전 체크리스트

- [ ] 실제 브라우저 네트워크 요청과 payload 일치 확인
- [ ] 2회 이상 동일 조건 재현 성공
- [ ] `block1` 외 응답 스키마 변형 대응 필요 여부 확인
- [ ] KRX 이용정책/약관 리스크 검토
- [ ] 호출 속도 제한 및 차단 징후 점검

## 현재 한계

- 이 세션에서는 PowerShell 초기화 오류(`8009001d`) 때문에 로컬 테스트/실행 검증을 완료하지 못했다.
- 따라서 이 구현은 **실행 전 검증이 필요한 프로토타입**이다.
