"""
Regime validity evaluation runner.

Outputs:
- legacy path: `docs/regime-validity-<YYYY-MM-DD>.md`
- dashboard parity path: `docs/regime-validity-dashboard-parity-<YYYY-MM-DD>.md`
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.validate_sector_mapping import (
        DASHBOARD_RUNTIME_WINDOW_MONTHS,
        REGIMES,
        align_with_regimes,
        build_current_assignment,
        build_sector_name_map,
        compute_monthly_excess_returns,
        extract_regime_history,
        get_all_sector_codes,
        load_configs,
        load_macro,
        load_sector_prices_wide,
        load_settings,
        normalize_validation_path,
        resolve_label_column,
        summarize_regime_history,
    )
except ImportError:
    from validate_sector_mapping import (
        DASHBOARD_RUNTIME_WINDOW_MONTHS,
        REGIMES,
        align_with_regimes,
        build_current_assignment,
        build_sector_name_map,
        compute_monthly_excess_returns,
        extract_regime_history,
        get_all_sector_codes,
        load_configs,
        load_macro,
        load_sector_prices_wide,
        load_settings,
        normalize_validation_path,
        resolve_label_column,
        summarize_regime_history,
    )
from src.macro.regime import classify_regime, compute_regime_history
from src.macro.series_utils import (
    build_regime_history_from_macro,
    build_regime_inflation_series,
    extract_macro_series,
    filter_macro_provisional_rows,
)
from src.transforms.resample import apply_confirmation_filter, compute_3ma_direction


LAGS = (0, 1, 2)
EPSILONS = (0.0, 0.05, 0.1, 0.2)
CLASSIFIER_BASELINE_ASSIGNMENT = {
    "5044": "Recovery",
    "1155": "Recovery",
    "5042": "Expansion",
    "1168": "Expansion",
    "1165": "Expansion",
    "5048": "Slowdown",
    "5049": "Slowdown",
    "5045": "Slowdown",
    "1170": "Contraction",
    "1157": "Contraction",
    "5046": "Contraction",
}


@dataclass
class RankRow:
    lag_months: int
    regime: str
    rank: int
    code: str
    sector_name: str
    avg_excess_pct: float
    assigned_regime: str


def _ordered_counts(raw_counts: dict[str, int]) -> dict[str, int]:
    ordered = {name: int(raw_counts.get(name, 0)) for name in REGIMES}
    ordered["Indeterminate"] = int(raw_counts.get("Indeterminate", 0))
    return ordered


def _render_counts(counts: dict[str, int]) -> str:
    ordered = _ordered_counts(counts)
    return (
        f"Recovery {ordered['Recovery']} / "
        f"Expansion {ordered['Expansion']} / "
        f"Slowdown {ordered['Slowdown']} / "
        f"Contraction {ordered['Contraction']} / "
        f"Indeterminate {ordered['Indeterminate']}"
    )


def _checkpoint_counts_from_todo() -> dict[str, int] | None:
    todo_path = PROJECT_ROOT / "tasks" / "todo.md"
    if not todo_path.exists():
        return None
    matches = re.findall(r"confirmed_counts\s+(\{[^}]+\})", todo_path.read_text(encoding="utf-8"))
    if not matches:
        return None
    payload = ast.literal_eval(matches[-1])
    return _ordered_counts({str(key): int(value) for key, value in dict(payload).items()})


def _count_delta(current: dict[str, int], checkpoint: dict[str, int] | None) -> dict[str, int] | None:
    if checkpoint is None:
        return None
    ordered_current = _ordered_counts(current)
    return {
        key: int(ordered_current.get(key, 0)) - int(checkpoint.get(key, 0))
        for key in ordered_current
    }


def _lag_scenario_metrics(
    *,
    lag: int,
    regime_hist: pd.DataFrame,
    label_col: str,
    excess: pd.DataFrame,
    sector_names: dict[str, str],
    assignment: dict[str, str],
) -> tuple[dict, list[RankRow]]:
    shifted = regime_hist.copy()
    shifted["_active_regime"] = shifted[label_col].shift(lag) if lag > 0 else shifted[label_col]
    combined = align_with_regimes(excess, shifted[["_active_regime"]], label_col="_active_regime")
    counts = _ordered_counts(combined["regime"].value_counts().to_dict())

    fit_total = 0
    fit_hits = 0
    ranking_rows: list[RankRow] = []
    for regime in REGIMES:
        subset = combined[combined["regime"] == regime]
        if subset.empty:
            continue
        sector_cols = [column for column in subset.columns if column != "regime"]
        means = subset[sector_cols].mean().sort_values(ascending=False)
        total = len(means)
        for rank, (code, value) in enumerate(means.items(), start=1):
            code_str = str(code)
            assigned_regime = assignment.get(code_str, "-")
            ranking_rows.append(
                RankRow(
                    lag_months=lag,
                    regime=regime,
                    rank=rank,
                    code=code_str,
                    sector_name=sector_names.get(code_str, code_str),
                    avg_excess_pct=float(value * 100.0),
                    assigned_regime=assigned_regime,
                )
            )
            if assigned_regime == regime:
                fit_total += 1
                if rank <= total // 2:
                    fit_hits += 1

    fit_rate = (fit_hits / fit_total * 100.0) if fit_total else 0.0
    return (
        {
            "lag_months": lag,
            "observed_months": int(len(combined)),
            "counts": counts,
            "fit_hits": int(fit_hits),
            "fit_total": int(fit_total),
            "fit_rate_pct": float(fit_rate),
        },
        ranking_rows,
    )


def _epsilon_metrics(
    *,
    macro_df: pd.DataFrame,
    macro_series_cfg: dict,
    settings: dict,
    include_provisional: bool,
    market_id: str,
) -> list[dict]:
    rows: list[dict] = []
    for epsilon in EPSILONS:
        tuned_settings = dict(settings)
        tuned_settings["epsilon"] = float(epsilon)
        hist = build_regime_history_from_macro(
            macro_df=macro_df,
            macro_series_cfg=macro_series_cfg,
            settings=tuned_settings,
            market_id=market_id,
            include_provisional=include_provisional,
            window_months=None,
        )
        if hist.empty:
            continue
        confirmed_counts = _ordered_counts(hist["confirmed_regime"].value_counts().to_dict())
        indeterminate_count = confirmed_counts["Indeterminate"]
        rows.append(
            {
                "epsilon": float(epsilon),
                "latest_regime": str(hist["confirmed_regime"].iloc[-1]),
                "total_points": int(len(hist)),
                "indeterminate_count": int(indeterminate_count),
                "indeterminate_share_pct": float(indeterminate_count / len(hist) * 100.0),
                "counts": confirmed_counts,
            }
        )
    return rows


def _raw_vs_confirmed_divergence(regime_hist: pd.DataFrame) -> dict[str, object]:
    if regime_hist.empty:
        return {"points": 0, "count": 0, "pct": 0.0}
    raw = regime_hist["regime"].astype(str)
    confirmed = regime_hist["confirmed_regime"].astype(str)
    count = int((raw != confirmed).sum())
    points = int(len(regime_hist))
    return {
        "points": points,
        "count": count,
        "pct": float(count / points * 100.0) if points else 0.0,
    }


def _stage1_gate(
    *,
    lag_scenarios: list[dict],
    divergence: dict[str, object],
) -> dict[str, object]:
    by_lag = {int(row["lag_months"]): row for row in lag_scenarios}
    lag0 = float(by_lag[0]["fit_rate_pct"])
    lag1 = float(by_lag[1]["fit_rate_pct"])
    triggers: list[str] = []
    if float(divergence["pct"]) > 15.0:
        triggers.append(f"confirmed-vs-raw divergence {divergence['pct']:.1f}% > 15%")
    if lag1 < 35.0 and (lag0 - lag1) >= 20.0:
        triggers.append(f"PIT fit {lag1:.1f}% < 35% with nowcast gap {lag0 - lag1:.1f}%p")
    return {
        "open": bool(triggers),
        "triggers": triggers,
        "lag0_fit_pct": lag0,
        "lag1_fit_pct": lag1,
    }


def _transition_count(labels: pd.Series) -> int:
    if labels.empty:
        return 0
    return int((labels.astype(str) != labels.astype(str).shift(1)).sum() - 1)


def _stage1_experiment_metrics(
    *,
    macro_df: pd.DataFrame,
    macro_series_cfg: dict,
    settings: dict,
    include_provisional: bool,
    excess: pd.DataFrame,
    sector_names: dict[str, str],
    assignment: dict[str, str],
) -> dict[str, object]:
    baseline_settings = dict(settings)
    baseline_settings["carry_single_flat_regime"] = False
    baseline_hist = build_regime_history_from_macro(
        macro_df=macro_df,
        macro_series_cfg=macro_series_cfg,
        settings=baseline_settings,
        market_id="KR",
        include_provisional=include_provisional,
        window_months=None,
    )
    trial_settings = dict(settings)
    trial_settings["carry_single_flat_regime"] = True
    trial_hist = build_regime_history_from_macro(
        macro_df=macro_df,
        macro_series_cfg=macro_series_cfg,
        settings=trial_settings,
        market_id="KR",
        include_provisional=include_provisional,
        window_months=None,
    )
    baseline_lag = _lag_scenario_metrics(
        lag=1,
        regime_hist=baseline_hist,
        label_col="confirmed_regime",
        excess=excess,
        sector_names=sector_names,
        assignment=assignment,
    )[0]
    trial_lag = _lag_scenario_metrics(
        lag=1,
        regime_hist=trial_hist,
        label_col="confirmed_regime",
        excess=excess,
        sector_names=sector_names,
        assignment=assignment,
    )[0]
    changed_months = int(
        (baseline_hist["confirmed_regime"].astype(str) != trial_hist["confirmed_regime"].astype(str)).sum()
    )
    changed_pct = float(changed_months / len(trial_hist) * 100.0) if len(trial_hist) else 0.0
    baseline_transitions = _transition_count(baseline_hist["confirmed_regime"])
    trial_transitions = _transition_count(trial_hist["confirmed_regime"])
    latest_same = bool(
        not baseline_hist.empty
        and not trial_hist.empty
        and str(baseline_hist["confirmed_regime"].iloc[-1]) == str(trial_hist["confirmed_regime"].iloc[-1])
    )
    accepted = bool(
        changed_pct <= 5.0
        and float(trial_lag["fit_rate_pct"]) + 5.0 >= float(baseline_lag["fit_rate_pct"])
        and latest_same
        and trial_transitions >= (baseline_transitions - 1)
    )
    return {
        "enabled": bool(settings.get("carry_single_flat_regime", False)),
        "accepted": accepted,
        "changed_months": changed_months,
        "changed_pct": changed_pct,
        "baseline_lag1_fit_pct": float(baseline_lag["fit_rate_pct"]),
        "trial_lag1_fit_pct": float(trial_lag["fit_rate_pct"]),
        "baseline_transitions": baseline_transitions,
        "trial_transitions": trial_transitions,
        "latest_same": latest_same,
    }


def _prepare_regime_inputs(
    *,
    macro_df: pd.DataFrame,
    macro_series_cfg: dict,
    include_provisional: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    filtered = filter_macro_provisional_rows(
        macro_df,
        include_provisional=include_provisional,
    )
    growth = extract_macro_series(filtered, macro_series_cfg, "leading_index")
    inflation = build_regime_inflation_series(
        macro_df=filtered,
        macro_series_cfg=macro_series_cfg,
        market_id="KR",
    )
    aligned = pd.concat({"growth": growth, "inflation": inflation}, axis=1, join="inner").dropna()
    growth_dir = compute_3ma_direction(aligned["growth"], epsilon=0.0)
    inflation_dir = compute_3ma_direction(aligned["inflation"], epsilon=0.0)
    return aligned, growth_dir, inflation_dir


def _build_bounded_bridge_history(
    *,
    growth_dir: pd.Series,
    inflation_dir: pd.Series,
    confirmation_periods: int,
    lookback_months: int,
) -> tuple[pd.DataFrame, int]:
    raw_labels = [classify_regime(g, i) for g, i in zip(growth_dir, inflation_dir)]
    bridged_labels: list[str] = []
    prior_nonflat: list[tuple[int, str]] = []
    affected_months = 0

    for idx, (raw_label, growth_label, inflation_label) in enumerate(
        zip(raw_labels, growth_dir, inflation_dir)
    ):
        bridged = raw_label
        if raw_label == "Indeterminate" and ((growth_label == "Flat") ^ (inflation_label == "Flat")):
            eligible = [value for pos, value in prior_nonflat if idx - pos <= lookback_months]
            if eligible:
                affected_months += 1
                bridged = eligible[-1]
        if raw_label != "Indeterminate":
            prior_nonflat.append((idx, raw_label))
        bridged_labels.append(bridged)

    result = pd.DataFrame({"regime": bridged_labels}, index=growth_dir.index)
    result["confirmed_regime"] = apply_confirmation_filter(result["regime"], n=confirmation_periods)
    return result, affected_months


def _macro_fit_action_churn(
    *,
    baseline_regimes: pd.Series,
    candidate_regimes: pd.Series,
    excess: pd.DataFrame,
    assignment: dict[str, str],
) -> dict[str, object]:
    months = excess.copy()
    months.index = months.index.to_period("M")
    comparable_months = months.index.intersection(baseline_regimes.index).intersection(candidate_regimes.index)
    changed_action_months = 0
    comparable_replay_months = 0

    for period in comparable_months:
        baseline_label = str(baseline_regimes.loc[period])
        candidate_label = str(candidate_regimes.loc[period])
        for code, assigned_regime in assignment.items():
            comparable_replay_months += 1
            baseline_fit = baseline_label != "Indeterminate" and assigned_regime == baseline_label
            candidate_fit = candidate_label != "Indeterminate" and assigned_regime == candidate_label
            if baseline_fit != candidate_fit:
                changed_action_months += 1

    churn_pct = (
        float(changed_action_months / comparable_replay_months * 100.0)
        if comparable_replay_months
        else 0.0
    )
    return {
        "changed_action_months": changed_action_months,
        "comparable_replay_months": comparable_replay_months,
        "unexplained_churn_pct": churn_pct,
    }


def _pre_registered_bridge_metrics(
    *,
    macro_df: pd.DataFrame,
    macro_series_cfg: dict,
    settings: dict,
    include_provisional: bool,
    current_hist: pd.DataFrame,
    excess: pd.DataFrame,
    assignment: dict[str, str],
) -> dict[str, object]:
    aligned, growth_dir, inflation_dir = _prepare_regime_inputs(
        macro_df=macro_df,
        macro_series_cfg=macro_series_cfg,
        include_provisional=include_provisional,
    )
    if current_hist.empty or aligned.empty:
        return {
            "name": "flat_to_prior_nonflat_bridge",
            "lookback_months": 3,
            "affected_months": 0,
            "changed_confirmed_months_vs_current": 0,
            "changed_confirmed_pct_vs_current": 0.0,
            "candidate_lag1_fit_pct": 0.0,
            "candidate_transition_count": 0,
            "latest_same_as_current": True,
            "unexplained_churn_formula": "changed_action_months / comparable_replay_months",
            "changed_action_months": 0,
            "comparable_replay_months": 0,
            "unexplained_churn_pct": 0.0,
            "strong_enough": False,
            "justification": "No aligned replay window available.",
        }

    aligned = aligned.loc[current_hist.index.min():current_hist.index.max()]
    growth_dir = growth_dir.loc[aligned.index]
    inflation_dir = inflation_dir.loc[aligned.index]
    candidate_hist, affected_months = _build_bounded_bridge_history(
        growth_dir=growth_dir,
        inflation_dir=inflation_dir,
        confirmation_periods=int(settings.get("confirmation_periods", 2)),
        lookback_months=3,
    )
    current_scoped = current_hist.loc[candidate_hist.index]
    changed_confirmed = int(
        (current_scoped["confirmed_regime"].astype(str) != candidate_hist["confirmed_regime"].astype(str)).sum()
    )
    changed_confirmed_pct = (
        float(changed_confirmed / len(candidate_hist) * 100.0) if len(candidate_hist) else 0.0
    )
    candidate_lag1 = _lag_scenario_metrics(
        lag=1,
        regime_hist=candidate_hist,
        label_col="confirmed_regime",
        excess=excess,
        sector_names={},
        assignment=assignment,
    )[0]
    churn = _macro_fit_action_churn(
        baseline_regimes=current_scoped["confirmed_regime"].astype(str),
        candidate_regimes=candidate_hist["confirmed_regime"].astype(str),
        excess=excess,
        assignment=assignment,
    )
    candidate_transition_count = _transition_count(candidate_hist["confirmed_regime"])
    latest_same = bool(
        str(current_scoped["confirmed_regime"].iloc[-1]) == str(candidate_hist["confirmed_regime"].iloc[-1])
    )
    strong_enough = bool(
        affected_months > 0
        and changed_confirmed > 0
        and float(candidate_lag1["fit_rate_pct"]) >= 30.3
        and churn["unexplained_churn_pct"] <= 5.0
        and latest_same
    )
    justification = (
        "Candidate changes no confirmed-regime months versus the current accepted path."
        if changed_confirmed == 0
        else "Candidate changes confirmed-regime months, but thresholds still determine whether it is viable."
    )
    return {
        "name": "flat_to_prior_nonflat_bridge",
        "lookback_months": 3,
        "affected_months": affected_months,
        "changed_confirmed_months_vs_current": changed_confirmed,
        "changed_confirmed_pct_vs_current": changed_confirmed_pct,
        "candidate_lag1_fit_pct": float(candidate_lag1["fit_rate_pct"]),
        "candidate_transition_count": candidate_transition_count,
        "latest_same_as_current": latest_same,
        "unexplained_churn_formula": "changed_action_months / comparable_replay_months",
        "changed_action_months": int(churn["changed_action_months"]),
        "comparable_replay_months": int(churn["comparable_replay_months"]),
        "unexplained_churn_pct": float(churn["unexplained_churn_pct"]),
        "strong_enough": strong_enough,
        "justification": justification,
    }


def _classifier_risk_report_path(asof: str) -> Path:
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir / f"regime-classifier-risk-closure-{asof}.md"


def _build_classifier_risk_report_text(
    *,
    asof: str,
    replay_window_start: str,
    replay_window_end: str,
    divergence: dict[str, object],
    gate: dict[str, object],
    current_transition_count: int,
    latest_confirmed_regime: str,
    current_experiment: dict[str, object],
    preregistered_bridge: dict[str, object],
) -> str:
    lines: list[str] = []
    lines.append(f"# Regime Classifier Risk Closure ({asof})")
    lines.append("")
    lines.append(f"- replay window: `{replay_window_start}` -> `{replay_window_end}`")
    lines.append(f"- divergence: `{divergence['count']}/{divergence['points']} = {divergence['pct']:.1f}%`")
    lines.append(f"- baseline lag0 fit: `{gate['lag0_fit_pct']:.1f}%`")
    lines.append(f"- baseline lag1/PIT fit: `{gate['lag1_fit_pct']:.1f}%`")
    lines.append(f"- baseline transition count: `{current_transition_count}`")
    lines.append(f"- latest confirmed regime: `{latest_confirmed_regime}`")
    lines.append("")
    lines.append("## Current Accepted Experiment")
    lines.append(
        f"- `carry_single_flat_regime`: `{'ACCEPTED' if current_experiment['accepted'] else 'REVIEW NEEDED'}`"
    )
    lines.append(
        f"- lag1/PIT: `{current_experiment['baseline_lag1_fit_pct']:.1f}% -> {current_experiment['trial_lag1_fit_pct']:.1f}%`"
    )
    lines.append(
        f"- changed confirmed months: `{current_experiment['changed_months']}/{divergence['points']} = {current_experiment['changed_pct']:.1f}%`"
    )
    lines.append("")
    lines.append("## Pre-Registered Candidate")
    lines.append(f"- name: `{preregistered_bridge['name']}`")
    lines.append(f"- lookback: `{preregistered_bridge['lookback_months']} months`")
    lines.append(f"- affected months: `{preregistered_bridge['affected_months']}`")
    lines.append(
        f"- changed confirmed months vs current: `{preregistered_bridge['changed_confirmed_months_vs_current']}/{divergence['points']} = {preregistered_bridge['changed_confirmed_pct_vs_current']:.1f}%`"
    )
    lines.append(
        f"- candidate lag1/PIT fit: `{preregistered_bridge['candidate_lag1_fit_pct']:.1f}%`"
    )
    lines.append(
        f"- unexplained churn formula: `{preregistered_bridge['unexplained_churn_formula']}`"
    )
    lines.append(
        f"- unexplained churn: `{preregistered_bridge['changed_action_months']}/{preregistered_bridge['comparable_replay_months']} = {preregistered_bridge['unexplained_churn_pct']:.1f}%`"
    )
    lines.append(
        f"- latest confirmed regime unchanged: `{preregistered_bridge['latest_same_as_current']}`"
    )
    lines.append(f"- rationale: {preregistered_bridge['justification']}")
    lines.append("")
    lines.append("## Decision")
    decision = (
        "Freeze classifier semantics now. The named pre-registration case is not strong enough to justify another classifier change on this fixed replay window."
        if not preregistered_bridge["strong_enough"]
        else "Run the pre-registered classifier experiment and evaluate it against the fixed thresholds before any further semantic change."
    )
    lines.append(f"- {decision}")
    lines.append("- `lag0` nowcast and `lag1/PIT` remain explicitly separated.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("```bash")
    lines.append(f"python scripts/evaluate_regime_validity.py --path dashboard-parity --asof {asof}")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _append_rank_table(lines: list[str], rows: list[RankRow]) -> None:
    lines.append("| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |")
    lines.append("|---:|---|---|---:|---|")
    for row in rows:
        lines.append(
            f"| {row.rank} | {row.code} | {row.sector_name} | {row.avg_excess_pct:+.2f} | {row.assigned_regime} |"
        )


def _report_paths(path: str, asof: str) -> tuple[Path, Path]:
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    if path == "dashboard-parity":
        stem = f"regime-validity-dashboard-parity-{asof}"
    else:
        stem = f"regime-validity-{asof}"
    return docs_dir / f"{stem}.md", docs_dir / f"{stem}-rankings.csv"


def _current_alias_paths() -> tuple[Path, Path]:
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    return (
        docs_dir / "regime-validity-dashboard-parity-current.md",
        docs_dir / "regime-validity-dashboard-parity-current-rankings.csv",
    )


def _build_report_text(
    *,
    asof: str,
    path: str,
    label_col: str,
    include_provisional: bool,
    primary_summary: dict[str, object],
    legacy_summary: dict[str, object],
    runtime_summary: dict[str, object],
    checkpoint_counts: dict[str, int] | None,
    checkpoint_delta: dict[str, int] | None,
    lag_scenarios: list[dict],
    epsilon_rows: list[dict],
    divergence: dict[str, object],
    gate: dict[str, object],
    stage1_experiment: dict[str, object],
    ranking_rows: list[RankRow],
) -> str:
    by_lag = {int(row["lag_months"]): row for row in lag_scenarios}
    lines: list[str] = []
    title = "경기국면 판정 타당성 평가 보고서"
    if path == "dashboard-parity":
        title = "경기국면 판정 타당성 평가 보고서 (dashboard-parity)"
    lines.append(f"# {title} ({asof})")
    lines.append("")
    lines.append("> Authority source: pinned curated snapshot + this parity artifact")
    lines.append("> `tasks/todo.md` is used only as a reconciliation checkpoint.")
    lines.append("")
    if path == "dashboard-parity":
        lines.append("## Historical Note")
        lines.append("- `docs/regime-validity-2026-02-25.md` and `docs/sector-mapping-validation.md` are historical snapshots from the older raw-regime path.")
        lines.append("- This report uses the current dashboard-derived regime-construction rules on the current curated artifacts.")
        lines.append("")

    lines.append("## Parity Snapshot")
    lines.append(f"- path: `{path}`")
    lines.append(f"- label column: `{label_col}`")
    lines.append(f"- include provisional: `{include_provisional}`")
    lines.append(f"- replay window: `{primary_summary['start']}` -> `{primary_summary['end']}` ({primary_summary['points']} points)")
    lines.append(f"- primary counts: `{_render_counts(primary_summary['counts'])}`")
    lines.append("")
    lines.append("## Comparison")
    lines.append("| Slice | Window | Counts |")
    lines.append("|---|---|---|")
    lines.append(f"| Primary `{path}` | {primary_summary['start']} -> {primary_summary['end']} | {_render_counts(primary_summary['counts'])} |")
    lines.append(f"| Legacy raw path | {legacy_summary['start']} -> {legacy_summary['end']} | {_render_counts(legacy_summary['counts'])} |")
    lines.append(f"| Dashboard runtime 60M reference | {runtime_summary['start']} -> {runtime_summary['end']} | {_render_counts(runtime_summary['counts'])} |")
    if checkpoint_counts is not None:
        lines.append(f"| `tasks/todo.md` checkpoint | refreshed note | {_render_counts(checkpoint_counts)} |")
    lines.append("")
    if checkpoint_delta is not None:
        lines.append("## Checkpoint Delta")
        delta_text = ", ".join(f"{key} {value:+d}" for key, value in checkpoint_delta.items())
        lines.append(f"- primary vs `tasks/todo.md`: `{delta_text}`")
        lines.append("")

    lines.append("## Gate Summary")
    lines.append(f"- raw vs confirmed divergence: `{divergence['count']}/{divergence['points']} = {divergence['pct']:.1f}%`")
    lines.append(f"- lag0 fit: `{gate['lag0_fit_pct']:.1f}%`")
    lines.append(f"- lag1 fit: `{gate['lag1_fit_pct']:.1f}%`")
    if gate["open"]:
        lines.append("- Stage 1 gate: `OPEN`")
        for trigger in gate["triggers"]:
            lines.append(f"  - {trigger}")
    else:
        lines.append("- Stage 1 gate: `CLOSED`")
    if stage1_experiment["enabled"]:
        verdict = "ACCEPTED" if stage1_experiment["accepted"] else "REVIEW NEEDED"
        lines.append(f"- Stage 1 experiment: `carry_single_flat_regime` enabled ({verdict})")
        lines.append(
            f"  - changed months: `{stage1_experiment['changed_months']}/{divergence['points']} = {stage1_experiment['changed_pct']:.1f}%`"
        )
        lines.append(
            f"  - lag1 fit: `{stage1_experiment['baseline_lag1_fit_pct']:.1f}% -> {stage1_experiment['trial_lag1_fit_pct']:.1f}%`"
        )
        lines.append(
            f"  - transitions: `{stage1_experiment['baseline_transitions']} -> {stage1_experiment['trial_transitions']}`"
        )
        lines.append(f"  - latest confirmed regime stable: `{stage1_experiment['latest_same']}`")
    lines.append("")
    if path == "dashboard-parity":
        lines.append("## Decision")
        lines.append("- Stage 1R default posture: `freeze classifier semantics and improve reporting/wording only`.")
        lines.append("- The only named exception is `flat_to_prior_nonflat_bridge`, and it should open only if pre-registration on this fixed replay window is strong enough.")
        lines.append("- `lag0` nowcast and `lag1/PIT` evidence must stay separated.")
        lines.append("")

    lines.append("## Lag Scenarios")
    lines.append("| Scenario | Observed months | Counts (R/E/S/C/I) | Top-half fit |")
    lines.append("|---|---:|---|---:|")
    for lag in LAGS:
        row = by_lag[lag]
        lines.append(
            f"| lag={lag} | {row['observed_months']} | {_render_counts(row['counts'])} | {row['fit_hits']}/{row['fit_total']} ({row['fit_rate_pct']:.1f}%) |"
        )
    lines.append("")

    lines.append("## Epsilon Sensitivity")
    lines.append("| epsilon | latest confirmed | Indeterminate share | counts |")
    lines.append("|---:|---|---:|---|")
    for row in epsilon_rows:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['latest_regime']} | {row['indeterminate_share_pct']:.1f}% | {_render_counts(row['counts'])} |"
        )
    lines.append("")

    lines.append("## Ranking Tables")
    for lag in LAGS:
        lines.append("")
        lines.append(f"### lag={lag}")
        for regime in REGIMES:
            lines.append("")
            lines.append(f"#### {regime}")
            rows = [row for row in ranking_rows if row.lag_months == lag and row.regime == regime]
            if not rows:
                lines.append("- 데이터 없음")
                continue
            _append_rank_table(lines, rows)
    lines.append("")
    lines.append("## Reproduction")
    lines.append("```bash")
    if path == "dashboard-parity":
        lines.append(f"PYTHONIOENCODING=utf-8 python scripts/evaluate_regime_validity.py --path dashboard-parity --asof {asof}")
    else:
        lines.append(f"PYTHONIOENCODING=utf-8 python scripts/evaluate_regime_validity.py --asof {asof}")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def run(
    asof: str,
    *,
    path: str,
    label_column: str,
    include_provisional: bool,
    window_months: int | None,
    runtime_window_months: int,
) -> tuple[Path, Path, dict[str, object]]:
    normalized_path = normalize_validation_path(path)
    sector_map, macro_series_cfg = load_configs()
    settings = load_settings()
    sector_names = build_sector_name_map(sector_map)
    assignment = build_current_assignment(sector_map)
    classifier_assignment = dict(CLASSIFIER_BASELINE_ASSIGNMENT)

    all_codes = get_all_sector_codes(sector_map)
    wide = load_sector_prices_wide(all_codes)
    macro = load_macro()
    excess = compute_monthly_excess_returns(wide)

    primary_label_col = resolve_label_column(normalized_path, label_column)
    primary_hist = extract_regime_history(
        macro,
        macro_series_cfg,
        settings=settings,
        path=normalized_path,
        include_provisional=include_provisional,
        window_months=window_months,
    )
    legacy_hist = extract_regime_history(
        macro,
        macro_series_cfg,
        settings=settings,
        path="legacy",
        include_provisional=False,
        window_months=window_months,
    )
    runtime_hist = extract_regime_history(
        macro,
        macro_series_cfg,
        settings=settings,
        path="dashboard-parity",
        include_provisional=True,
        window_months=runtime_window_months,
    )

    primary_summary = summarize_regime_history(primary_hist, label_col=primary_label_col)
    legacy_summary = summarize_regime_history(legacy_hist, label_col="regime")
    runtime_summary = summarize_regime_history(runtime_hist, label_col="confirmed_regime")

    checkpoint_counts = _checkpoint_counts_from_todo()
    checkpoint_delta = _count_delta(primary_summary["counts"], checkpoint_counts)

    lag_scenarios: list[dict] = []
    ranking_rows: list[RankRow] = []
    for lag in LAGS:
        metrics, rows = _lag_scenario_metrics(
            lag=lag,
            regime_hist=primary_hist,
            label_col=primary_label_col,
            excess=excess,
            sector_names=sector_names,
            assignment=assignment,
        )
        lag_scenarios.append(metrics)
        ranking_rows.extend(rows)

    classifier_lag_scenarios: list[dict] = []
    for lag in LAGS:
        classifier_metrics, _ = _lag_scenario_metrics(
            lag=lag,
            regime_hist=primary_hist,
            label_col=primary_label_col,
            excess=excess,
            sector_names=sector_names,
            assignment=classifier_assignment,
        )
        classifier_lag_scenarios.append(classifier_metrics)

    epsilon_rows = _epsilon_metrics(
        macro_df=macro,
        macro_series_cfg=macro_series_cfg,
        settings=settings,
        include_provisional=include_provisional,
        market_id="KR",
    )
    divergence = _raw_vs_confirmed_divergence(primary_hist)
    gate = _stage1_gate(lag_scenarios=lag_scenarios, divergence=divergence)
    classifier_gate = _stage1_gate(lag_scenarios=classifier_lag_scenarios, divergence=divergence)
    stage1_experiment = _stage1_experiment_metrics(
        macro_df=macro,
        macro_series_cfg=macro_series_cfg,
        settings=settings,
        include_provisional=include_provisional,
        excess=excess,
        sector_names=sector_names,
        assignment=classifier_assignment,
    )
    preregistered_bridge = _pre_registered_bridge_metrics(
        macro_df=macro,
        macro_series_cfg=macro_series_cfg,
        settings=settings,
        include_provisional=include_provisional,
        current_hist=primary_hist,
        excess=excess,
        assignment=classifier_assignment,
    )

    report_path, ranking_csv_path = _report_paths(normalized_path, asof)
    report_text = _build_report_text(
        asof=asof,
        path=normalized_path,
        label_col=primary_label_col,
        include_provisional=include_provisional,
        primary_summary=primary_summary,
        legacy_summary=legacy_summary,
        runtime_summary=runtime_summary,
        checkpoint_counts=checkpoint_counts,
        checkpoint_delta=checkpoint_delta,
        lag_scenarios=lag_scenarios,
        epsilon_rows=epsilon_rows,
        divergence=divergence,
        gate=gate,
        stage1_experiment=stage1_experiment,
        ranking_rows=ranking_rows,
    )
    report_path.write_text(report_text, encoding="utf-8")
    ranking_frame = pd.DataFrame([row.__dict__ for row in ranking_rows])
    ranking_frame.to_csv(ranking_csv_path, index=False, encoding="utf-8-sig")
    if normalized_path == "dashboard-parity":
        current_report_path, current_ranking_path = _current_alias_paths()
        current_report_path.write_text(report_text, encoding="utf-8")
        ranking_frame.to_csv(current_ranking_path, index=False, encoding="utf-8-sig")
        classifier_risk_path = _classifier_risk_report_path(asof)
        classifier_risk_path.write_text(
            _build_classifier_risk_report_text(
                asof=asof,
                replay_window_start=str(primary_summary["start"]),
                replay_window_end=str(primary_summary["end"]),
                divergence=divergence,
                gate=classifier_gate,
                current_transition_count=int(_transition_count(primary_hist["confirmed_regime"])),
                latest_confirmed_regime=str(primary_hist["confirmed_regime"].iloc[-1]),
                current_experiment=stage1_experiment,
                preregistered_bridge=preregistered_bridge,
            ),
            encoding="utf-8",
        )

    return report_path, ranking_csv_path, {
        "path": normalized_path,
        "label_col": primary_label_col,
        "primary_summary": primary_summary,
        "legacy_summary": legacy_summary,
        "runtime_summary": runtime_summary,
        "checkpoint_counts": checkpoint_counts,
        "checkpoint_delta": checkpoint_delta,
        "divergence": divergence,
        "gate": gate,
        "classifier_gate": classifier_gate,
        "stage1_experiment": stage1_experiment,
        "preregistered_bridge": preregistered_bridge,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate regime validity.")
    parser.add_argument("--asof", default=pd.Timestamp.now().date().isoformat(), help="Report label in YYYY-MM-DD.")
    parser.add_argument("--path", default="legacy", choices=["legacy", "dashboard-parity"], help="Validation path.")
    parser.add_argument(
        "--label-column",
        default="auto",
        choices=["auto", "regime", "confirmed_regime"],
        help="Label column for lag scenarios and fit calculations.",
    )
    parser.add_argument(
        "--include-provisional",
        action="store_true",
        help="Keep provisional macro rows in the primary path.",
    )
    parser.add_argument(
        "--window-months",
        type=int,
        default=None,
        help="Optional trailing month window for the primary and legacy paths.",
    )
    parser.add_argument(
        "--runtime-window-months",
        type=int,
        default=DASHBOARD_RUNTIME_WINDOW_MONTHS,
        help="Trailing month window for the dashboard runtime reference slice.",
    )
    args = parser.parse_args()

    report_path, ranking_csv_path, summary = run(
        args.asof,
        path=args.path,
        label_column=args.label_column,
        include_provisional=args.include_provisional,
        window_months=args.window_months,
        runtime_window_months=args.runtime_window_months,
    )
    print(f"report: {report_path}")
    print(f"rankings: {ranking_csv_path}")
    print(f"path: {summary['path']}")
    print(f"label_col: {summary['label_col']}")
    print(
        "replay_window: "
        f"{summary['primary_summary']['start']} -> {summary['primary_summary']['end']} "
        f"({summary['primary_summary']['points']} points)"
    )
    print(f"primary_counts: {summary['primary_summary']['counts']}")
    print(f"runtime_60m_counts: {summary['runtime_summary']['counts']}")
    print(f"legacy_counts: {summary['legacy_summary']['counts']}")
    print(f"divergence_pct: {summary['divergence']['pct']:.1f}")
    print(f"stage1_gate: {'OPEN' if summary['gate']['open'] else 'CLOSED'}")
    if summary["stage1_experiment"]["enabled"]:
        print(
            "stage1_experiment: "
            f"{'ACCEPTED' if summary['stage1_experiment']['accepted'] else 'REVIEW NEEDED'} "
            f"(changed_pct={summary['stage1_experiment']['changed_pct']:.1f}, "
            f"lag1={summary['stage1_experiment']['baseline_lag1_fit_pct']:.1f}->"
            f"{summary['stage1_experiment']['trial_lag1_fit_pct']:.1f})"
        )
    print(
        "pre_registered_bridge: "
        f"{'STRONG' if summary['preregistered_bridge']['strong_enough'] else 'WEAK'} "
        f"(affected={summary['preregistered_bridge']['affected_months']}, "
        f"changed_vs_current={summary['preregistered_bridge']['changed_confirmed_months_vs_current']}, "
        f"churn={summary['preregistered_bridge']['unexplained_churn_pct']:.1f})"
    )


if __name__ == "__main__":
    main()
