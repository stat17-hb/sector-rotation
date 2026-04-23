"""Momentum methodology comparison runner.

Outputs:
- dated path: `docs/momentum-method-comparison-<YYYY-MM-DD>.md`
- dated csv: `docs/momentum-method-comparison-<YYYY-MM-DD>.csv`
- optional current aliases via `--update-current`
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.markets import load_market_configs
from src.data_sources.warehouse import read_market_prices
from src.signals.matrix import build_signal_table


LEGACY_METHOD = "legacy_rs_ma_v0"
HYBRID_METHOD = "hybrid_return_rank_v1"


@dataclass
class ComparisonMetrics:
    whipsaws_legacy: int
    whipsaws_hybrid: int
    whipsaw_reduction_pct: float
    median_rank_rho: float
    p10_rank_rho: float
    pass_whipsaw: bool
    pass_rank_stability: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate legacy vs hybrid momentum methods.")
    parser.add_argument("--market", default="KR")
    parser.add_argument("--benchmark-code", default="")
    parser.add_argument("--asof", required=True)
    parser.add_argument("--window-months", type=int, default=36)
    parser.add_argument("--update-current", action="store_true")
    return parser.parse_args()


def _all_sector_codes(sector_map: dict) -> list[str]:
    codes: list[str] = []
    for regime_data in sector_map.get("regimes", {}).values():
        for item in regime_data.get("sectors", []):
            code = str(item.get("code", "")).strip()
            if code and code not in codes:
                codes.append(code)
    return codes


def _month_end_checkpoints(benchmark_series: pd.Series, *, asof: str, window_months: int) -> list[pd.Timestamp]:
    frame = benchmark_series.sort_index().dropna()
    frame = frame.loc[frame.index <= pd.Timestamp(asof)]
    if frame.empty:
        return []
    checkpoints = frame.groupby(frame.index.to_period("M")).apply(lambda series: series.index[-1])
    if checkpoints.empty:
        return []
    return list(pd.DatetimeIndex(checkpoints.iloc[-window_months:]))


def _dummy_macro_result(checkpoint: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "growth_dir": ["Up"],
            "inflation_dir": ["Down"],
            "regime": ["Recovery"],
            "confirmed_regime": ["Recovery"],
        },
        index=pd.DatetimeIndex([checkpoint]),
    )


def _run_method(
    *,
    sector_prices: pd.DataFrame,
    benchmark_code: str,
    sector_map: dict,
    settings: dict,
    checkpoint: pd.Timestamp,
    method: str,
) -> list[dict[str, object]]:
    runtime_settings = dict(settings)
    runtime_settings["momentum_method"] = method
    sliced = sector_prices.loc[sector_prices.index <= checkpoint].copy()
    bench_mask = sliced["index_code"].astype(str) == str(benchmark_code)
    benchmark_series = sliced.loc[bench_mask, "close"].copy()
    signals = build_signal_table(
        sector_prices=sliced,
        benchmark_prices=benchmark_series,
        macro_result=_dummy_macro_result(checkpoint),
        sector_map=sector_map,
        settings=runtime_settings,
        fx_change_pct=None,
    )
    rows: list[dict[str, object]] = []
    for signal in signals:
        rows.append(
            {
                "checkpoint": checkpoint.strftime("%Y-%m-%d"),
                "method": method,
                "sector_code": str(signal.index_code),
                "sector_name": str(signal.sector_name),
                "action": str(signal.action),
                "momentum_strong": bool(signal.momentum_strong),
                "mom_score": getattr(signal, "mom_score", float("nan")),
                "mom_rank": getattr(signal, "mom_rank", None),
                "mom_percentile": getattr(signal, "mom_percentile", float("nan")),
                "eligible": str(signal.action) != "N/A",
            }
        )
    return rows


def _count_whipsaws(method_rows: pd.DataFrame) -> int:
    total = 0
    for _, frame in method_rows.groupby("sector_code"):
        valid = frame.sort_values("checkpoint")
        valid = valid[valid["eligible"] == True]
        if len(valid) < 3:
            continue
        values = list(valid["momentum_strong"].astype(bool))
        for idx in range(1, len(values) - 1):
            prev_value = values[idx - 1]
            curr_value = values[idx]
            next_values = values[idx + 1 : idx + 3]
            if curr_value != prev_value and prev_value in next_values:
                total += 1
    return total


def _rank_stability(method_rows: pd.DataFrame) -> tuple[float, float]:
    frame = method_rows.copy()
    frame = frame[frame["eligible"] == True]
    frame = frame.dropna(subset=["mom_rank"])
    if frame.empty:
        return float("nan"), float("nan")

    checkpoints = sorted(frame["checkpoint"].unique())
    rhos: list[float] = []
    for prev_cp, next_cp in zip(checkpoints, checkpoints[1:]):
        prev_rank = (
            frame.loc[frame["checkpoint"] == prev_cp, ["sector_code", "mom_rank"]]
            .set_index("sector_code")["mom_rank"]
            .astype("float64")
        )
        next_rank = (
            frame.loc[frame["checkpoint"] == next_cp, ["sector_code", "mom_rank"]]
            .set_index("sector_code")["mom_rank"]
            .astype("float64")
        )
        common = prev_rank.index.intersection(next_rank.index)
        if len(common) < 2:
            continue
        rho = prev_rank.loc[common].rank(method="average").corr(
            next_rank.loc[common].rank(method="average")
        )
        if pd.notna(rho):
            rhos.append(float(rho))
    if not rhos:
        return float("nan"), float("nan")
    return float(pd.Series(rhos).median()), float(pd.Series(rhos).quantile(0.10))


def _compute_metrics(rows: pd.DataFrame) -> ComparisonMetrics:
    legacy_rows = rows.loc[rows["method"] == LEGACY_METHOD].copy()
    hybrid_rows = rows.loc[rows["method"] == HYBRID_METHOD].copy()
    whipsaws_legacy = _count_whipsaws(legacy_rows)
    whipsaws_hybrid = _count_whipsaws(hybrid_rows)
    if whipsaws_legacy > 0:
        reduction = (whipsaws_legacy - whipsaws_hybrid) / whipsaws_legacy * 100.0
    else:
        reduction = 0.0
    median_rho, p10_rho = _rank_stability(hybrid_rows)
    return ComparisonMetrics(
        whipsaws_legacy=whipsaws_legacy,
        whipsaws_hybrid=whipsaws_hybrid,
        whipsaw_reduction_pct=float(reduction),
        median_rank_rho=median_rho,
        p10_rank_rho=p10_rho,
        pass_whipsaw=bool(reduction >= 15.0),
        pass_rank_stability=bool(pd.notna(median_rho) and pd.notna(p10_rho) and median_rho >= 0.70 and p10_rho >= 0.40),
    )


def _render_report(
    *,
    market: str,
    benchmark_code: str,
    asof: str,
    window_months: int,
    settings: dict,
    rows: pd.DataFrame,
    metrics: ComparisonMetrics,
) -> str:
    eligible_counts = (
        rows.loc[rows["eligible"] == True]
        .groupby(["checkpoint", "method"])
        .size()
        .reset_index(name="eligible_sector_count")
    )
    hybrid_approved = metrics.pass_whipsaw and metrics.pass_rank_stability
    lines: list[str] = []
    lines.append(f"# Momentum Method Comparison ({asof})")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- market_id: `{market}`")
    lines.append(f"- benchmark_code: `{benchmark_code}`")
    lines.append(f"- methods: `{LEGACY_METHOD}`, `{HYBRID_METHOD}`")
    lines.append(f"- window_months: `{window_months}`")
    lines.append(f"- momentum_skip_recent_days: `{settings.get('momentum_skip_recent_days', 21)}`")
    lines.append(f"- momentum_lookback_6m_days: `{settings.get('momentum_lookback_6m_days', 126)}`")
    lines.append(f"- momentum_lookback_12m_days: `{settings.get('momentum_lookback_12m_days', 252)}`")
    lines.append(f"- momentum_rank_threshold_pct: `{settings.get('momentum_rank_threshold_pct', 0.60)}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append(f"- legacy whipsaws: `{metrics.whipsaws_legacy}`")
    lines.append(f"- hybrid whipsaws: `{metrics.whipsaws_hybrid}`")
    lines.append(f"- whipsaw reduction: `{metrics.whipsaw_reduction_pct:.1f}%`")
    lines.append(f"- hybrid median adjacent-month Spearman rho: `{metrics.median_rank_rho:.2f}`")
    lines.append(f"- hybrid 10th-percentile rho: `{metrics.p10_rank_rho:.2f}`")
    lines.append(f"- pass_whipsaw: `{metrics.pass_whipsaw}`")
    lines.append(f"- pass_rank_stability: `{metrics.pass_rank_stability}`")
    lines.append("")
    lines.append("## Monthly Eligible Sector Counts")
    if eligible_counts.empty:
        lines.append("- No eligible checkpoints.")
    else:
        for _, row in eligible_counts.iterrows():
            lines.append(
                f"- {row['checkpoint']} | {row['method']} | eligible sectors: {int(row['eligible_sector_count'])}"
            )
    lines.append("")
    lines.append("## Decision")
    lines.append(
        "- Hybrid cutover is approved."
        if hybrid_approved
        else "- Hybrid cutover remains evidence-only. Metrics did not yet clear the configured gate."
    )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("```bash")
    lines.append(
        f"python scripts/evaluate_momentum_method.py --market {market} --benchmark-code {benchmark_code} --asof {asof} --window-months {window_months} --update-current"
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    market = str(args.market or "KR").strip().upper() or "KR"
    if market != "KR":
        print("Phase 1 momentum comparison currently supports KR only.", file=sys.stderr)
        return 2
    settings, sector_map, _macro_series, profile = load_market_configs(market)
    benchmark_code = str(args.benchmark_code or settings.get("benchmark_code") or profile.benchmark_code).strip()
    asof = str(args.asof)

    all_codes = _all_sector_codes(sector_map)
    if benchmark_code not in all_codes:
        all_codes.append(benchmark_code)

    start = (pd.Timestamp(asof) - pd.Timedelta(days=1200)).strftime("%Y%m%d")
    end = pd.Timestamp(asof).strftime("%Y%m%d")
    sector_prices = read_market_prices(all_codes, start, end, market=market)
    if sector_prices.empty:
        print("No market prices available in warehouse for the requested range.", file=sys.stderr)
        return 1

    benchmark_series = sector_prices.loc[sector_prices["index_code"].astype(str) == benchmark_code, "close"].copy()
    checkpoints = _month_end_checkpoints(benchmark_series, asof=asof, window_months=int(args.window_months))
    if not checkpoints:
        print("No usable month-end checkpoints were found.", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        rows.extend(
            _run_method(
                sector_prices=sector_prices,
                benchmark_code=benchmark_code,
                sector_map=sector_map,
                settings=settings,
                checkpoint=checkpoint,
                method=LEGACY_METHOD,
            )
        )
        rows.extend(
            _run_method(
                sector_prices=sector_prices,
                benchmark_code=benchmark_code,
                sector_map=sector_map,
                settings=settings,
                checkpoint=checkpoint,
                method=HYBRID_METHOD,
            )
        )

    result_df = pd.DataFrame(rows)
    metrics = _compute_metrics(result_df)

    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    dated_md = docs_dir / f"momentum-method-comparison-{pd.Timestamp(asof).date()}.md"
    dated_csv = docs_dir / f"momentum-method-comparison-{pd.Timestamp(asof).date()}.csv"
    report = _render_report(
        market=market,
        benchmark_code=benchmark_code,
        asof=asof,
        window_months=int(args.window_months),
        settings=settings,
        rows=result_df,
        metrics=metrics,
    )
    dated_md.write_text(report, encoding="utf-8")
    result_df.to_csv(dated_csv, index=False, encoding="utf-8-sig")

    if args.update_current:
        shutil.copyfile(dated_md, docs_dir / "momentum-method-comparison-current.md")
        shutil.copyfile(dated_csv, docs_dir / "momentum-method-comparison-current.csv")

    hybrid_approved = metrics.pass_whipsaw and metrics.pass_rank_stability
    print(f"market={market} benchmark={benchmark_code} window_months={int(args.window_months)}")
    print(
        "whipsaws "
        f"legacy={metrics.whipsaws_legacy} hybrid={metrics.whipsaws_hybrid} "
        f"reduction_pct={metrics.whipsaw_reduction_pct:.1f}"
    )
    print(
        "rank_stability "
        f"median_rho={metrics.median_rank_rho:.2f} "
        f"p10_rho={metrics.p10_rank_rho:.2f}"
    )
    print(f"gate_state={'approved' if hybrid_approved else 'evidence-only'}")
    print(dated_md)
    print(dated_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
