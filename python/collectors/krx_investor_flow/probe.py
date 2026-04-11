"""HTTP probe helpers for unofficial KRX investor-flow collection."""

from __future__ import annotations

import json
from http.cookiejar import CookieJar
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

from .config import (
    DEFAULT_ENDPOINT_URL,
    MAIN_INDEX_REFERER,
    MARKET_PRESET,
    STOCK_PRESET,
    build_headers,
    build_market_payload,
    build_stock_payload,
)
from .normalize import normalize_investor_flow_rows
from .schemas import ProbeArtifacts


class KrxProbeError(RuntimeError):
    """Raised when the unofficial KRX probe cannot produce a usable response."""


class KrxInvestorFlowClient:
    """Small stdlib-only client for KRX investor-flow web probes."""

    def __init__(self, *, endpoint_url: str = DEFAULT_ENDPOINT_URL, timeout: int = 20) -> None:
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        self._opener = build_opener(HTTPCookieProcessor(CookieJar()))

    def _open(self, request: Request) -> bytes:
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                return response.read()
        except HTTPError as exc:  # pragma: no cover - network dependent
            body = exc.read().decode("utf-8", errors="replace")
            raise KrxProbeError(f"KRX probe HTTP error {exc.code}: {body[:300]}") from exc
        except URLError as exc:  # pragma: no cover - network dependent
            raise KrxProbeError(f"KRX probe transport error: {exc}") from exc

    def prewarm(self, referer: str) -> None:
        self._open(Request(MAIN_INDEX_REFERER, headers=build_headers(referer=MAIN_INDEX_REFERER)))
        self._open(Request(referer, headers=build_headers(referer=referer)))

    def post_form(self, *, payload: dict[str, str], referer: str, prewarm: bool = True) -> dict:
        if prewarm:
            self.prewarm(referer)

        request = Request(
            self.endpoint_url,
            data=urlencode(payload).encode("utf-8"),
            headers=build_headers(referer=referer),
            method="POST",
        )
        raw_body = self._open(request)
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            snippet = raw_body.decode("utf-8", errors="replace")[:300]
            raise KrxProbeError(
                f"KRX probe returned a non-JSON response. First 300 chars: {snippet!r}"
            ) from exc

    def fetch_market(
        self,
        *,
        start_date: str,
        end_date: str,
        market_id: str = "ALL",
        prewarm: bool = True,
    ) -> ProbeArtifacts:
        payload = build_market_payload(start_date=start_date, end_date=end_date, market_id=market_id)
        response = self.post_form(payload=payload, referer=MARKET_PRESET.referer, prewarm=prewarm)
        return ProbeArtifacts(
            mode="market",
            payload=payload,
            response=response,
            normalized_rows=normalize_investor_flow_rows(
                response,
                default_trade_date=end_date,
                default_market=market_id,
            ),
            endpoint_url=self.endpoint_url,
            referer=MARKET_PRESET.referer,
            bld=payload.get("bld"),
            notes=[
                "Unofficial KRX web-backend probe.",
                "Investigate headers/cookies/anti-bot requirements before operational use.",
            ],
        )

    def fetch_stock(
        self,
        *,
        start_date: str,
        end_date: str,
        isu_cd: str,
        ticker: str | None = None,
        bld_candidates: tuple[str, ...] | None = None,
        prewarm: bool = True,
    ) -> ProbeArtifacts:
        candidate_blds = bld_candidates or tuple(STOCK_PRESET.bld_candidates)
        last_error: Exception | None = None
        for bld in candidate_blds:
            base_payload = build_stock_payload(
                start_date=start_date,
                end_date=end_date,
                isu_cd=isu_cd,
                ticker=ticker,
                bld=bld,
            )
            payload_candidates: list[dict[str, str]] = [base_payload]

            if "isuCd" in base_payload:
                finder_only_payload = dict(base_payload)
                finder_only_payload.pop("isuCd", None)
                payload_candidates.append(finder_only_payload)

            if "inqTpVal" in base_payload:
                alt_inq_key_payload = dict(base_payload)
                alt_inq_key_payload["inqTpCd"] = alt_inq_key_payload.pop("inqTpVal")
                payload_candidates.append(alt_inq_key_payload)

            minimal_payload = {"bld": bld, "locale": "ko_KR", "csvxls_isNo": "false"}
            if ticker:
                minimal_payload["tboxisuCd_finder_stkisu0_25"] = f"{ticker}/\\\\isuCd"
            minimal_payload["isuCd"] = isu_cd
            minimal_payload["inqTpVal"] = "2"
            payload_candidates.append(minimal_payload)

            seen_payloads: set[tuple[tuple[str, str], ...]] = set()
            for payload in payload_candidates:
                payload_key = tuple(sorted(payload.items()))
                if payload_key in seen_payloads:
                    continue
                seen_payloads.add(payload_key)
                try:
                    response = self.post_form(payload=payload, referer=STOCK_PRESET.referer, prewarm=prewarm)
                except Exception as exc:  # pragma: no cover - network dependent
                    last_error = exc
                    continue

                return ProbeArtifacts(
                    mode="stock",
                    payload=payload,
                    response=response,
                    normalized_rows=normalize_investor_flow_rows(response, default_trade_date=end_date),
                    endpoint_url=self.endpoint_url,
                    referer=STOCK_PRESET.referer,
                    bld=bld,
                    notes=[
                        "Unofficial KRX stock-level investor-flow probe.",
                        "The stock probe may require an isuCd discovered from a separate KRX finder endpoint.",
                    ],
                )

        message = "All stock probe payload candidates failed."
        if last_error is not None:
            message += f" Last error: {last_error}"
        raise KrxProbeError(message)
