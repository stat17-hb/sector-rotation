# Korea Sector Rotation Dashboard

한국 주식시장 섹터 로테이션 분석을 위한 Streamlit 대시보드

---

## 개요

거시경제 국면(경기 사이클)에 따라 유망 섹터를 탐지하고 모멘텀 신호를 결합하여 투자 액션(매수 / 관망 / 회피)을 제시하는 대시보드입니다.

**주요 기능**

- 4단계 경기 국면 분류 (Recovery / Expansion / Slowdown / Contraction)
- KRX 섹터 지수 가격 데이터 자동 수집 (pykrx)
- 한국은행 ECOS API — 기준금리, 국고채 3년물, USD/KRW
- 통계청 KOSIS API — CPI YoY, 경기선행지수
- 섹터별 상대강도(RS), RSI, 이동평균 모멘텀 지표
- 매크로 국면 × 모멘텀 상태 → 투자 신호 매트릭스
- API 미연결 시 샘플 데이터 폴백 및 경고 배너

---

## 경기 국면 & 섹터 매핑

| 국면 | 조건 | 주요 섹터 |
|------|------|-----------|
| **Recovery** | 성장 ↑, 물가/금리 ↓ | KRX 반도체, KOSPI200 IT |
| **Expansion** | 성장 ↑, 물가/금리 ↑ | KRX 산업재, KOSPI200 금융, 경기소비재 |
| **Slowdown** | 성장 ↓, 물가/금리 ↑ | KRX 에너지화학, 철강, KOSPI200 유틸리티 |
| **Contraction** | 성장 ↓, 물가/금리 ↓ | KRX 헬스케어, 미디어통신, KOSPI200 생활소비재 |

---

## 프로젝트 구조

```
sector-rotation/
├── app.py                      # Streamlit SPA 진입점
├── requirements.txt
├── environment.yml             # conda 환경 정의
├── config/
│   ├── settings.yml            # 알고리즘 파라미터 (RS 기간, RSI 설정 등)
│   ├── sector_map.yml          # 국면 → KRX 섹터 코드 매핑
│   └── macro_series.yml        # ECOS/KOSIS 시리즈 ID 및 레이블
├── src/
│   ├── contracts/              # DataFrame 스키마 검증
│   ├── data_sources/           # KRX, ECOS, KOSIS 데이터 수집
│   ├── transforms/             # 영업일 계산, 리샘플링
│   ├── indicators/             # 모멘텀(RS, SMA), RSI 계산
│   ├── macro/                  # 거시경제 국면 분류
│   ├── signals/                # 신호 매트릭스 및 RSI/FX 필터
│   └── ui/                     # CSS, 컴포넌트, 데이터 상태
├── data/
│   ├── raw/                    # 원천 데이터 (parquet)
│   ├── curated/                # 정제 데이터
│   └── features/               # 피처 데이터
├── scripts/
│   └── run_streamlit.bat       # Windows 실행 스크립트
└── tests/                      # pytest 테스트 모음
```

---

## 설치 및 실행

### 1. 환경 구성

**conda 사용 (권장)**

```bash
conda env create -f environment.yml
conda activate sector-rotation
```

**pip 사용**

```bash
pip install -r requirements.txt
```

### 2. API 키 설정

ECOS(한국은행)와 KOSIS(통계청) API를 사용하려면 키가 필요합니다.

`.streamlit/secrets.toml` 파일을 생성하고 아래와 같이 입력하세요:

```toml
ECOS_API_KEY = "your_ecos_key_here"
KOSIS_API_KEY = "your_kosis_key_here"
```

- ECOS API 신청: https://ecos.bok.or.kr/api/
- KOSIS API 신청: https://kosis.kr/openapi/

> API 키 없이 실행하면 샘플 데이터로 동작하며, 대시보드 상단에 경고 배너가 표시됩니다.

### 3. 실행

**Windows (conda 환경 자동 활성화)**

```bat
scripts\run_streamlit.bat
```

**직접 실행**

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## 주요 파라미터 (`config/settings.yml`)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `rs_ma_period` | 20 | 상대강도 이동평균 기간 |
| `ma_fast` | 20 | 단기 이동평균 기간 |
| `ma_slow` | 60 | 장기 이동평균 기간 |
| `rsi_period` | 14 | RSI 계산 기간 |
| `rsi_overbought` | 70 | RSI 과매수 임계값 |
| `rsi_oversold` | 30 | RSI 과매도 임계값 |
| `fx_shock_pct` | 3.0 | FX 충격 필터 임계값 (%) |
| `price_years` | 3 | 가격 데이터 수집 기간 (년) |
| `cache_ttl` | 21600 | 캐시 유효 시간 (초, 기본 6시간) |

---

## 데이터 흐름

```
KRX (pykrx) ──────────────┐
ECOS API (한국은행) ───────┤──▶ 캐시(parquet) ──▶ 지표 계산 ──▶ 신호 매트릭스 ──▶ 대시보드
KOSIS API (통계청) ────────┘
```

데이터 갱신 버튼별 캐시 범위:

| 버튼 | 삭제 대상 | 재수집 |
|------|-----------|--------|
| 시장데이터 갱신 | `curated/sector_prices.parquet` | pykrx (실시간) |
| 매크로데이터 갱신 | `curated/macro_monthly.parquet` | ECOS + KOSIS API |
| 전체 재계산 | `features/*.parquet` | API 미호출, 기존 curated 재계산 |

---

## 테스트

```bash
pytest tests/ -v
```

---

## 의존성

| 라이브러리 | 용도 |
|-----------|------|
| `streamlit` | 웹 대시보드 |
| `pykrx` | KRX 섹터/지수 데이터 |
| `pandas` / `numpy` | 데이터 처리 |
| `plotly` | 인터랙티브 차트 |
| `ta` | 기술적 지표 (RSI 등) |
| `requests` | ECOS/KOSIS API 호출 |
| `pyyaml` | 설정 파일 파싱 |
| `pyarrow` | parquet 파일 처리 |
