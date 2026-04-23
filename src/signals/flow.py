"""Investor-flow scoring and post-processing for sector signals."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import pandas as pd


FLOW_PROFILE_IDS: tuple[str, ...] = (
    "foreign_lead",
    "institutional_confirmation",
    "contrarian_retail",
)
FLOW_STATE_IDS: tuple[str, ...] = ("supportive", "neutral", "adverse", "unavailable")
INVESTOR_GROUPS: tuple[str, ...] = ("foreign", "institutional", "retail")
INVESTOR_LABEL_TO_GROUP: dict[str, str] = {
    "외국인": "foreign",
    "기관합계": "institutional",
    "개인": "retail",
}
PROFILE_CONFIG: dict[str, dict[str, Any]] = {
    "foreign_lead": {
        "score_group": "foreign",
        "score_sign": 1.0,
    },
    "institutional_confirmation": {
        "score_group": "institutional",
        "score_sign": 1.0,
    },
    "contrarian_retail": {
        "score_group": "retail",
        "score_sign": -1.0,
    },
}
UPGRADE_MAP: dict[str, str] = {
    "Hold": "Watch",
    "Watch": "Strong Buy",
}
DOWNGRADE_MAP: dict[str, str] = {
    "Strong Buy": "Watch",
    "Watch": "Hold",
    "Hold": "Avoid",
}


@dataclass(frozen=True)
class InvestorComponentSummary:
    state: str = "unavailable"
    latest_ratio: float | None = None
    short_mean: float | None = None
    long_mean: float | None = None
    zscore: float | None = None


@dataclass(frozen=True)
class SectorFlowSummary:
    sector_code: str
    sector_name: str
    flow_profile: str
    flow_state: str
    flow_score: float
    flow_reason: str
    foreign: InvestorComponentSummary = dataclasses.field(default_factory=InvestorComponentSummary)
    institutional: InvestorComponentSummary = dataclasses.field(default_factory=InvestorComponentSummary)
    retail: InvestorComponentSummary = dataclasses.field(default_factory=InvestorComponentSummary)
    latest_date: str = ""


def normalize_flow_profile(value: str | None) -> str:
    normalized = str(value or FLOW_PROFILE_IDS[0]).strip().lower()
    return normalized if normalized in FLOW_PROFILE_IDS else FLOW_PROFILE_IDS[0]


def _classify_component(series: pd.Series, *, short_window: int = 20, long_window: int = 60) -> InvestorComponentSummary:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < long_window:
        return InvestorComponentSummary()

    latest_ratio = float(values.iloc[-1])
    short_mean = float(values.tail(short_window).mean())
    long_sample = values.tail(long_window)
    long_mean = float(long_sample.mean())
    long_std = float(long_sample.std(ddof=0))
    zscore = 0.0 if long_std == 0 else (short_mean - long_mean) / long_std

    if zscore >= 0.5:
        state = "supportive"
    elif zscore <= -0.5:
        state = "adverse"
    else:
        state = "neutral"

    return InvestorComponentSummary(
        state=state,
        latest_ratio=latest_ratio,
        short_mean=short_mean,
        long_mean=long_mean,
        zscore=float(zscore),
    )


def summarize_sector_investor_flow(
    flow_frame: pd.DataFrame,
    *,
    flow_profile: str,
    short_window: int = 20,
    long_window: int = 60,
) -> dict[str, SectorFlowSummary]:
    profile = normalize_flow_profile(flow_profile)
    if flow_frame.empty:
        return {}

    normalized = flow_frame.copy()
    normalized.index = pd.DatetimeIndex(normalized.index)
    normalized = normalized.sort_index()

    summaries: dict[str, SectorFlowSummary] = {}
    for sector_code, sector_rows in normalized.groupby(normalized["sector_code"].astype(str)):
        sector_name = str(sector_rows["sector_name"].iloc[-1])
        component_summaries: dict[str, InvestorComponentSummary] = {}
        for investor_label, group_key in INVESTOR_LABEL_TO_GROUP.items():
            investor_rows = sector_rows[sector_rows["investor_type"].astype(str) == investor_label]
            component_summaries[group_key] = _classify_component(
                investor_rows["net_flow_ratio"],
                short_window=short_window,
                long_window=long_window,
            )

        config = PROFILE_CONFIG[profile]
        score_group = str(config["score_group"])
        score_sign = float(config["score_sign"])
        score_component = component_summaries[score_group]
        if score_component.state == "unavailable" or score_component.zscore is None:
            flow_state = "unavailable"
            flow_score = 0.0
        else:
            flow_score = float(score_component.zscore) * score_sign
            if flow_score >= 0.5:
                flow_state = "supportive"
            elif flow_score <= -0.5:
                flow_state = "adverse"
            else:
                flow_state = "neutral"

        foreign_state = component_summaries["foreign"].state
        institutional_state = component_summaries["institutional"].state
        retail_state = component_summaries["retail"].state
        flow_reason = (
            f"profile={profile}, score_group={score_group}, sigma={flow_score:+.2f}, "
            f"foreign={foreign_state}, institutional={institutional_state}, retail={retail_state}"
        )
        summaries[str(sector_code)] = SectorFlowSummary(
            sector_code=str(sector_code),
            sector_name=sector_name,
            flow_profile=profile,
            flow_state=flow_state,
            flow_score=float(flow_score),
            flow_reason=flow_reason,
            foreign=component_summaries["foreign"],
            institutional=component_summaries["institutional"],
            retail=component_summaries["retail"],
            latest_date=sector_rows.index.max().strftime("%Y-%m-%d"),
        )
    return summaries


def _flow_adjustment_label(base_action: str, adjusted_action: str, *, flow_state: str) -> str:
    if flow_state == "unavailable":
        return "experimental unavailable"
    if base_action == adjusted_action:
        return "none"
    if UPGRADE_MAP.get(base_action) == adjusted_action:
        return "upgrade"
    if DOWNGRADE_MAP.get(base_action) == adjusted_action:
        return "downgrade"
    return "none"


def apply_flow_to_signal(signal, summary: SectorFlowSummary | None, *, flow_profile: str):
    from src.signals.matrix import ACTION_VALUES

    base_action = str(getattr(signal, "action", "N/A"))
    adjusted_action = base_action
    flow_state = "unavailable"
    flow_score = 0.0
    flow_reason = "experimental unavailable"
    foreign_state = "unavailable"
    institutional_state = "unavailable"
    retail_state = "unavailable"
    foreign_ratio = float("nan")
    institutional_ratio = float("nan")
    retail_ratio = float("nan")
    foreign_z = float("nan")
    institutional_z = float("nan")
    retail_z = float("nan")

    if base_action == "N/A":
        summary = None

    if summary is not None:
        flow_state = summary.flow_state
        flow_score = float(summary.flow_score)
        flow_reason = str(summary.flow_reason)
        foreign_state = summary.foreign.state
        institutional_state = summary.institutional.state
        retail_state = summary.retail.state
        foreign_ratio = summary.foreign.latest_ratio if summary.foreign.latest_ratio is not None else float("nan")
        institutional_ratio = summary.institutional.latest_ratio if summary.institutional.latest_ratio is not None else float("nan")
        retail_ratio = summary.retail.latest_ratio if summary.retail.latest_ratio is not None else float("nan")
        foreign_z = summary.foreign.zscore if summary.foreign.zscore is not None else float("nan")
        institutional_z = summary.institutional.zscore if summary.institutional.zscore is not None else float("nan")
        retail_z = summary.retail.zscore if summary.retail.zscore is not None else float("nan")
        if flow_state == "supportive" and base_action in UPGRADE_MAP:
            adjusted_action = UPGRADE_MAP[base_action]
        elif flow_state == "adverse" and base_action in DOWNGRADE_MAP:
            adjusted_action = DOWNGRADE_MAP[base_action]

    if adjusted_action not in ACTION_VALUES:
        adjusted_action = base_action

    return dataclasses.replace(
        signal,
        action=adjusted_action,
        base_action=base_action,
        flow_adjusted_action=adjusted_action,
        flow_adjustment=_flow_adjustment_label(base_action, adjusted_action, flow_state=flow_state),
        flow_profile=normalize_flow_profile(flow_profile),
        flow_state=flow_state,
        flow_score=flow_score,
        flow_reason=flow_reason,
        foreign_flow_state=foreign_state,
        institutional_flow_state=institutional_state,
        retail_flow_state=retail_state,
        foreign_flow_ratio=foreign_ratio,
        institutional_flow_ratio=institutional_ratio,
        retail_flow_ratio=retail_ratio,
        foreign_flow_z=foreign_z,
        institutional_flow_z=institutional_z,
        retail_flow_z=retail_z,
    )


def apply_flow_overlay(
    signals: list,
    *,
    flow_frame: pd.DataFrame,
    flow_profile: str,
    enabled: bool,
    short_window: int = 20,
    long_window: int = 60,
) -> tuple[list, dict[str, SectorFlowSummary]]:
    profile = normalize_flow_profile(flow_profile)
    if not enabled or flow_frame.empty:
        updated = [apply_flow_to_signal(signal, None, flow_profile=profile) for signal in signals]
        return updated, {}

    summary_map = summarize_sector_investor_flow(
        flow_frame,
        flow_profile=profile,
        short_window=short_window,
        long_window=long_window,
    )
    updated = [
        apply_flow_to_signal(signal, summary_map.get(str(getattr(signal, "index_code", ""))), flow_profile=profile)
        for signal in signals
    ]
    return updated, summary_map


__all__ = [
    "FLOW_PROFILE_IDS",
    "FLOW_STATE_IDS",
    "InvestorComponentSummary",
    "SectorFlowSummary",
    "apply_flow_overlay",
    "apply_flow_to_signal",
    "normalize_flow_profile",
    "summarize_sector_investor_flow",
]
