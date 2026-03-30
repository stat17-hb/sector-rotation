# Railway Deployment Guide (Streamlit)

This project can be deployed on Railway with `railway.toml` + `requirements.txt`.

## 1) Pre-check

- Entry point: `app.py`
- Start command is already defined in `railway.toml`.
- Python version is pinned by `runtime.txt` (`3.11`).

## 2) Push repository

```bash
git add railway.toml runtime.txt docs/railway-deploy.md
git commit -m "chore: add railway deployment setup"
git push origin main
```

## 3) Create Railway service

1. Open Railway dashboard.
2. `New Project` -> `Deploy from GitHub repo`.
3. Select this repository.
4. Railway will read `railway.toml` and build automatically.

## 4) Set environment variables

In Railway service variables, add:

- `ECOS_API_KEY` = your ECOS key
- `KOSIS_API_KEY` = your KOSIS key
- `KRX_OPENAPI_KEY` = your KRX OpenAPI key (optional but recommended)
- `KRX_PROVIDER` = `AUTO` (default), `OPENAPI`, or `PYKRX`
- `FRED_API_KEY` = your FRED key for US macro data

Bootstrap tuning (optional):

| Variable | Default | Description |
| --- | --- | --- |
| `SKIP_BOOTSTRAP` | `0` | Set to `1` to skip bootstrap entirely (e.g. local dev with pre-populated warehouse) |
| `BOOTSTRAP_PRICE_YEARS` | `3` | How many years of price history to fetch on cold start |
| `BOOTSTRAP_MACRO_YEARS` | `10` | How many years of macro history to fetch on cold start |
| `BOOTSTRAP_CHUNK_MONTHS` | `1` | pykrx fetch chunk size (months) — reduce if API rate-limits |

Notes:
- Do not set `PORT` manually. Railway injects it automatically.
- Without API keys, the app still runs in `SAMPLE` fallback mode.
- Mount a persistent volume for the repository `data/` directory so `data/warehouse.duckdb` and cache artifacts survive redeploys.
- Recommended target path is the app `data/` directory used at runtime (for Railway/Nixpacks this is typically `/app/data`).
- KR market uses `ECOS/KOSIS + KRX`; US market uses `FRED + yfinance`.

## 5) Cold-start bootstrap

On first deploy (or when a new Railway volume is attached), the DuckDB warehouse
file will be absent. `scripts/start.sh` detects this automatically:

```text
warehouse absent or < 8 KB -> run bootstrap_warehouse.py -> start Streamlit
warehouse already populated -> start Streamlit immediately
```

Bootstrap typically takes **1-3 minutes** depending on pykrx API latency.
During bootstrap the Railway health-check may show the service as "starting"; this is normal.

If bootstrap fails (network timeout, API quota), the app starts in **SAMPLE mode**
and a yellow warning banner is shown. Set `SKIP_BOOTSTRAP=1` and trigger a redeploy
once the API issue is resolved.

## 6) Deploy and verify

1. Trigger deploy (usually automatic after connect/push).
2. Check build logs for successful `pip install -r requirements.txt`.
3. Check runtime logs for `[start.sh] Bootstrap complete.` or `Warehouse present`.
4. Check runtime logs for Streamlit startup message.
5. Open generated Railway public URL.

## 7) Troubleshooting

- If deploy exits immediately:
  - Verify `railway.toml` exists at repo root.
  - Verify `scripts/start.sh` is executable (check `git ls-files --stage scripts/start.sh`).
- If bootstrap fails with `IOException`:
  - A stale DuckDB lock file may be present, or another process may still hold the file. Stop the writer, then delete `data/warehouse.duckdb` and redeploy if needed.
- If macro data stays fallback:
  - Recheck `ECOS_API_KEY`, `KOSIS_API_KEY`, `FRED_API_KEY`.
  - Confirm keys are valid and not rate-limited/revoked.
- If market prices stay cached/sample under KRX auth enforcement:
  - Recheck `KRX_OPENAPI_KEY` and `KRX_PROVIDER`.
  - Confirm your KRX account has approval for the index API service.
- If build fails due Python version mismatch:
  - Update `runtime.txt` to the exact version you want and redeploy.
