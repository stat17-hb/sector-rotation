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

Notes:
- Do not set `PORT` manually. Railway injects it automatically.
- Without API keys, the app still runs in `SAMPLE` fallback mode.

## 5) Deploy and verify

1. Trigger deploy (usually automatic after connect/push).
2. Check build logs for successful `pip install -r requirements.txt`.
3. Check runtime logs for Streamlit startup message.
4. Open generated Railway public URL.

## 6) Troubleshooting

- If deploy exits immediately:
  - Verify `railway.toml` exists at repo root.
  - Verify start command includes `--server.address 0.0.0.0` and `--server.port $PORT`.
- If macro data stays fallback:
  - Recheck `ECOS_API_KEY`, `KOSIS_API_KEY`.
  - Confirm keys are valid and not rate-limited/revoked.
- If build fails due Python version mismatch:
  - Update `runtime.txt` to the exact version you want and redeploy.
