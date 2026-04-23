# KR Sector Authority Remap Audit (2026-04-20)

## Decision

- Canonical target family: plain `KRX + sector`
- Migration gate: do not finalize canonical output until raw finder identity and discovery precedence are repaired
- Legacy families during transition: keep as non-canonical aliases with provenance

## Concrete mismatches observed

- `5042`
  - current config assumption: `KRX 산업재`
  - current official finder result: `KRX 100`
  - implication: this code cannot remain a canonical sector mapping

## Mixed-family rows still encoded in config

- `1155` `KOSPI200 정보기술`
- `1168` `KOSPI200 금융`
- `1157` `KOSPI200 생활소비재`
- `1165` `KOSPI200 경기소비재`
- `1170` `KOSPI200 유틸리티`
- `5044` `KRX 반도체`
- `5045` `KRX 헬스케어`
- `5046` `KRX 미디어통신`
- `5048` `KRX 에너지화학`
- `5049` `KRX 철강`

## Repair requirements

- preserve raw finder identity before canonical selection
- prefer `source_market=KRX` over duplicate `테마` rows after identity preservation
- stop letting local config names outrank official metadata at runtime
- exclude `KRX 100`, `KRX 300`, `TMI`, `KRX 300 *`, `KOSPI 200 *`, `KOSDAQ 150 *` from canonical sector output
