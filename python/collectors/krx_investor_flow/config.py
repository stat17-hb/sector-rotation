"""Configuration for the KRX investor-flow prototype collector.

This module intentionally keeps the unofficial endpoint assumptions in one place
so the probe can be updated or disabled quickly when KRX changes the web flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

DEFAULT_ENDPOINT_URL = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

# The public web UI has historically been reachable under both http and https.
# We default to https, but keep the form-POST assumptions configurable.
DEFAULT_ORIGIN = "http://data.krx.co.kr"
MAIN_INDEX_REFERER = "http://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd"
STOCK_STATS_REFERER = "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201"

# Public menu routes gathered from KRX Data Marketplace snippets and secondary
# browser-trace writeups. They are best-effort defaults, not guaranteed.
MARKET_INVESTOR_FLOW_REFERER = STOCK_STATS_REFERER
STOCK_INVESTOR_FLOW_REFERER = STOCK_STATS_REFERER

MARKET_INVESTOR_FLOW_BLD = "dbms/MDC/STAT/standard/MDCSTAT02202"
STOCK_INVESTOR_FLOW_BLD_CANDIDATES = (
    "dbms/MDC/STAT/standard/MDCSTAT02302",
    "dbms/MDC/STAT/표준/MDCSTAT02302",
)

DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Host": "data.krx.co.kr",
    "Pragma": "no-cache",
    "X-Requested-With": "XMLHttpRequest",
}


@dataclass(frozen=True)
class ProbePreset:
    """Best-effort description of a KRX web-backed probe target."""

    name: str
    referer: str
    default_payload: Mapping[str, str]
    bld_candidates: Iterable[str]


MARKET_PRESET = ProbePreset(
    name="market",
    referer=MARKET_INVESTOR_FLOW_REFERER,
    default_payload={
        "locale": "ko_KR",
        "inqTpCd": "2",
        "trdVolVal": "2",
        "askBid": "3",
        "mktId": "ALL",
        "money": "3",
        "csvxls_isNo": "false",
    },
    bld_candidates=(MARKET_INVESTOR_FLOW_BLD,),
)

STOCK_PRESET = ProbePreset(
    name="stock",
    referer=STOCK_INVESTOR_FLOW_REFERER,
    default_payload={
        "locale": "ko_KR",
        "inqTpVal": "2",
        "csvxls_isNo": "false",
    },
    bld_candidates=STOCK_INVESTOR_FLOW_BLD_CANDIDATES,
)


def build_headers(*, referer: str | None = None, origin: str = DEFAULT_ORIGIN) -> Dict[str, str]:
    headers = dict(DEFAULT_HEADERS)
    if origin:
        headers["Origin"] = origin
    if referer:
        headers["Referer"] = referer
    return headers


def build_market_payload(*, start_date: str, end_date: str, market_id: str = "ALL") -> Dict[str, str]:
    payload = dict(MARKET_PRESET.default_payload)
    payload.update(
        {
            "bld": MARKET_INVESTOR_FLOW_BLD,
            "strtDd": start_date,
            "endDd": end_date,
            "mktId": market_id,
        }
    )
    return payload


def build_stock_payload(
    *,
    start_date: str,
    end_date: str,
    isu_cd: str,
    ticker: str | None = None,
    bld: str | None = None,
    finder_field_key: str = "tboxisuCd_finder_stkisu0_25",
) -> Dict[str, str]:
    payload = dict(STOCK_PRESET.default_payload)
    payload.update(
        {
            "bld": bld or STOCK_INVESTOR_FLOW_BLD_CANDIDATES[0],
            "strtDd": start_date,
            "endDd": end_date,
            "isuCd": isu_cd,
        }
    )
    if ticker:
        payload[finder_field_key] = f"{ticker}/\\isuCd"
    return payload
