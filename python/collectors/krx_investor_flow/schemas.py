"""Schema objects for the KRX investor-flow prototype collector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InvestorFlowRow:
    trade_date: Optional[str]
    market: Optional[str]
    investor_type: Optional[str]
    buy_amount: Optional[int]
    sell_amount: Optional[int]
    net_buy_amount: Optional[int]
    ticker: Optional[str] = None
    ticker_name: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeArtifacts:
    mode: str
    payload: Dict[str, str]
    response: Dict[str, Any]
    normalized_rows: List[InvestorFlowRow]
    endpoint_url: str
    referer: str
    bld: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "payload": self.payload,
            "response": self.response,
            "normalized_rows": [row.to_dict() for row in self.normalized_rows],
            "endpoint_url": self.endpoint_url,
            "referer": self.referer,
            "bld": self.bld,
            "notes": list(self.notes),
        }
