"""Normalize unofficial KRX investor-flow responses to a canonical schema."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional

from .schemas import InvestorFlowRow

DATE_KEYS = ("TRD_DD", "trdDd", "일자", "trade_date")
MARKET_KEYS = ("MKT_NM", "mktNm", "시장구분", "market")
INVESTOR_KEYS = ("INVST_TP_NM", "invstTpNm", "투자자구분", "investor_type")

# In KRX naming, BID is typically the buy side and ASK is the sell side.
BUY_KEYS = (
    "BID_TRDVAL",
    "BID_TRDVOL",
    "buy_amount",
    "매수거래대금",
    "매수",
)
SELL_KEYS = (
    "ASK_TRDVAL",
    "ASK_TRDVOL",
    "sell_amount",
    "매도거래대금",
    "매도",
)
NET_KEYS = (
    "NETPRPS_TRDVAL",
    "NETPRPS_TRDVOL",
    "net_buy_amount",
    "순매수거래대금",
    "순매수",
)
TICKER_KEYS = ("ISU_SRT_CD", "isuSrtCd", "short_code", "ticker", "종목코드")
TICKER_NAME_KEYS = ("ISU_ABBRV", "isuAbbrv", "codeName", "ticker_name", "종목명")


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    text = text.replace(",", "").replace("+", "").replace(" ", "")
    if text in {"", "-"}:
        return None
    if text.startswith("-"):
        negative = True
        text = text[1:]
    if not text.isdigit():
        return None
    number = int(text)
    return -number if negative else number


def _first_present(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _extract_rows(response: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    preferred_keys = ("block1", "output", "OutBlock_1", "result")
    for key in preferred_keys:
        candidate = response.get(key)
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, Mapping)]
    for value in response.values():
        if isinstance(value, list) and value and all(isinstance(item, Mapping) for item in value):
            return list(value)
    return []


def normalize_investor_flow_rows(
    response: Mapping[str, Any],
    *,
    default_trade_date: str | None = None,
    default_market: str | None = None,
) -> List[InvestorFlowRow]:
    rows = _extract_rows(response)
    normalized: List[InvestorFlowRow] = []
    for raw_row in rows:
        normalized.append(
            InvestorFlowRow(
                trade_date=_first_present(raw_row, DATE_KEYS) or default_trade_date,
                market=_first_present(raw_row, MARKET_KEYS) or default_market,
                investor_type=_first_present(raw_row, INVESTOR_KEYS),
                buy_amount=_coerce_int(_first_present(raw_row, BUY_KEYS)),
                sell_amount=_coerce_int(_first_present(raw_row, SELL_KEYS)),
                net_buy_amount=_coerce_int(_first_present(raw_row, NET_KEYS)),
                ticker=_first_present(raw_row, TICKER_KEYS),
                ticker_name=_first_present(raw_row, TICKER_NAME_KEYS),
                raw=dict(raw_row),
            )
        )
    return normalized
