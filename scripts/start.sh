#!/usr/bin/env bash
# Startup script for Railway (and any Linux deployment).
#
# 1. If the DuckDB warehouse is empty (cold start / new volume mount),
#    run a lightweight bootstrap to pre-populate market and macro data.
# 2. Then launch Streamlit.
#
# Set SKIP_BOOTSTRAP=1 to bypass the bootstrap step (useful for local dev
# where the warehouse is already populated).

set -euo pipefail

WAREHOUSE_PATH="data/warehouse.duckdb"

run_bootstrap() {
    echo "[start.sh] Cold start detected — running warehouse bootstrap..."
    python scripts/bootstrap_warehouse.py \
        --prices-years "${BOOTSTRAP_PRICE_YEARS:-3}" \
        --macro-years  "${BOOTSTRAP_MACRO_YEARS:-10}" \
        --market-chunk-months "${BOOTSTRAP_CHUNK_MONTHS:-1}" \
        && echo "[start.sh] Bootstrap complete." \
        || echo "[start.sh] Bootstrap finished with errors — app will start in SAMPLE mode."
}

if [ "${SKIP_BOOTSTRAP:-0}" != "1" ]; then
    # Bootstrap if warehouse file is absent or smaller than 8 KB (empty schema only)
    if [ ! -f "$WAREHOUSE_PATH" ] || [ "$(wc -c < "$WAREHOUSE_PATH")" -lt 8192 ]; then
        run_bootstrap
    else
        echo "[start.sh] Warehouse present — skipping bootstrap."
    fi
else
    echo "[start.sh] SKIP_BOOTSTRAP=1 — skipping bootstrap."
fi

echo "[start.sh] Starting Streamlit..."
exec streamlit run app.py \
    --server.headless true \
    --server.address 0.0.0.0 \
    --server.port "${PORT:-8501}"
