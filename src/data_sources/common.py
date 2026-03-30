"""Shared helpers for external data-source loaders."""
from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests


REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
BACKOFF_BASE = 2
RETRYABLE_HTTP_STATUS_CODES = {429, 503}


def load_secret_or_env(name: str) -> str:
    """Load a secret from Streamlit first, then environment variables."""
    try:
        import streamlit as st  # type: ignore[import]

        value = str(st.secrets.get(name, "")).strip()
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def request_json_with_retry(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout_sec: int = REQUEST_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    backoff_base: int = BACKOFF_BASE,
    client_name: str = "HTTP",
    non_json_prefix: str | None = None,
) -> object:
    """HTTP GET with timeout, retry, and exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout_sec)
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError as exc:
                if non_json_prefix is None:
                    raise
                text_preview = (resp.text or "")[:200]
                raise ValueError(f"{non_json_prefix}: {text_preview!r}") from exc
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff_base ** (attempt + 1))
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in RETRYABLE_HTTP_STATUS_CODES:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(backoff_base ** (attempt + 1))
            else:
                raise

    assert last_exc is not None
    raise last_exc


def normalize_month_token(value: str) -> str:
    """Normalize a YYYYMM-like input into six digits."""
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) < 6:
        raise ValueError(f"Invalid month token: {value!r}")
    return digits[:6]


def shift_month_token(value: str, months: int) -> str:
    """Shift a YYYYMM token by N months."""
    period = pd.Period(normalize_month_token(value), freq="M")
    return str((period + int(months)).strftime("%Y%m"))
