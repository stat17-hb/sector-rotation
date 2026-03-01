# API 키 설정 가이드

대시보드가 실제 데이터(LIVE 모드)로 동작하려면 두 개의 API 키가 필요합니다.
키 없이도 실행은 가능하며, 이 경우 SAMPLE 모드(합성 데이터)로 동작합니다.

---

## 1. ECOS API 키 (한국은행 경제통계시스템)

**제공 데이터**: 기준금리, 국고채 3년, USD/KRW 환율

### 발급 절차

1. [https://ecos.bok.or.kr/api/](https://ecos.bok.or.kr/api/) 접속
2. 우측 상단 **회원가입** → 로그인
3. **API 키 신청** 메뉴 클릭
4. 사용 목적 입력 후 신청 (즉시 발급)
5. **마이페이지 → API 키 관리** 에서 키 확인

---

## 2. KOSIS API 키 (통계청 국가통계포털)

**제공 데이터**: CPI 전년동월비, 수출증감률, 경기선행지수순환변동치

### 발급 절차

1. [https://kosis.kr/openapi/](https://kosis.kr/openapi/) 접속
2. **회원가입** → 로그인
3. 상단 메뉴 **OpenAPI → 인증키 신청**
4. 서비스명/용도 입력 후 신청 (즉시 발급)
5. **마이페이지 → OpenAPI 신청 현황** 에서 키 확인

---

## 3. 키 설정 방법

프로젝트 루트에서 아래 명령 실행:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

`.streamlit/secrets.toml` 파일을 열어 키 입력:

```toml
ECOS_API_KEY = "여기에_ECOS_키_입력"
KOSIS_API_KEY = "여기에_KOSIS_키_입력"
```

> **주의**: `secrets.toml`은 `.gitignore`에 추가하여 Git에 커밋하지 마세요.

---

## 4. 환경변수로 설정 (선택)

`secrets.toml` 대신 환경변수를 사용할 수도 있습니다:

```bash
# Windows (PowerShell)
$env:ECOS_API_KEY = "여기에_키_입력"
$env:KOSIS_API_KEY = "여기에_키_입력"

# Windows (CMD)
set ECOS_API_KEY=여기에_키_입력
set KOSIS_API_KEY=여기에_키_입력
```

---

## 5. 대시보드 실행

```bash
conda activate sector-rotation
python -m streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속.

---

## 6. 데이터 상태 확인

대시보드 우측 사이드바 하단에 데이터 상태가 표시됩니다:

| 상태 | 의미 |
|------|------|
| **LIVE** | API에서 실시간 데이터 수신 중 |
| **CACHED** | 로컬 캐시(`data/curated/`) 데이터 사용 중 |
| **SAMPLE** | API 키 없음 또는 네트워크 오류 — 합성 데이터 사용 |

SAMPLE 모드에서는 상단에 빨간색 경고 배너가 표시되며,
"시장데이터 갱신" / "매크로데이터 갱신" 버튼으로 재시도할 수 있습니다.

---

## 7. API 사용 한도

| API | 무료 한도 | 비고 |
|-----|-----------|------|
| ECOS | 일 10,000건 | 일반적 대시보드 사용에 충분 |
| KOSIS | 일 10,000건 | 일반적 대시보드 사용에 충분 |

대시보드는 6시간 캐시(`CACHE_TTL=21600`)를 사용하므로 API 호출 빈도가 낮습니다.

---

## 8. Key Rotation and Cache Invalidation

When you update `ECOS_API_KEY` or `KOSIS_API_KEY`, run one of these immediately:

1. Click `매크로데이터 갱신` in the sidebar, or
2. Restart Streamlit server after `conda activate sector-rotation` (`python -m streamlit run app.py`).

Reason: macro loader cache can reuse failed responses. This app now fingerprints keys in cache tokens, but manual refresh is still the fastest reset path.

---

## 9. Error Code Interpretation (Do Not Overfit to Key Expiry)

- `ECOS ERROR-100`: can be invalid key, but also request schema mismatch (missing/invalid item code path).
- `KOSIS err=21` (and `err=20`): usually request variable mismatch for table-specific `objL*` parameters; not always key expiry.

Always verify request parameters and connectivity before rotating keys again.

---

## 10. API Preflight Status Guide

The app now runs a cached preflight check for `ECOS`, `KOSIS`, and `KRX` endpoints.

- `OK`: endpoint reachable (HTTP 2xx/3xx/4xx reachable)
- `TIMEOUT`: request timed out
- `DNS_FAIL`: DNS resolution issue
- `SOCKET_BLOCKED`: local/network policy blocked outbound socket (e.g., WinError 10013)
- `HTTP_ERROR`: endpoint returned 5xx or non-timeout request failure

Use preflight status to distinguish key issues from network/security restrictions.

---

## 11. Verified KOSIS Parameter Pattern (Important)

Some KOSIS tables do not use region/item codes the way they look at first glance.
For several macro series, the stable request pattern is:

- `itmId` = aggregate item code (for example `T` or `T1`)
- `objL1` = desired sub-dimension/member (for example `T10` or `A03`)

Working examples in this project:

- `cpi_yoy`: `orgId=101`, `tblId=DT_1J22003`, `itmId=T`, `objL1=T10`
- `leading_index`: `orgId=101`, `tblId=DT_1C8015`, `itmId=T1`, `objL1=A03`

If `itmId` is set directly to a member code (for example `T10`) and `objL1` is left as `ALL`, KOSIS often returns `err=21`.

---

## 12. Optional Macro Series

`config/macro_series.yml` supports `enabled: false` per series.

- Disabled series are skipped during live fetch.
- This allows non-critical series (for example `export_growth`) to be optional.
- Regime computation continues as long as required series (`cpi_yoy`, `leading_index`) are available.

---

## 13. KRX OpenAPI Key and Provider

KRX price loading now supports provider switching.

Add to `.streamlit/secrets.toml` (or environment variables):

```toml
KRX_OPENAPI_KEY = "your_krx_openapi_key_here"
KRX_PROVIDER = "AUTO" # AUTO | OPENAPI | PYKRX
```

Runtime behavior:

- `AUTO` with key -> `OPENAPI` path first
- `AUTO` without key -> `PYKRX` path
- `OPENAPI` without key -> explicit warning (`KRX_OPENAPI_KEY not configured`) and cache fallback

When key/provider changes, market-price cache is invalidated by key fingerprint token.

OpenAPI reference:
- https://openapi.krx.co.kr/contents/OPN/01/0104/01040100/OPN01040100.jsp
