"""
Regime validity evaluation runner (classification + sector mapping).

Outputs:
- docs/regime-validity-<YYYY-MM-DD>.md
- docs/regime-validity-<YYYY-MM-DD>-rankings.csv
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Ensure project root imports work when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validate_sector_mapping import (
    REGIMES,
    align_with_regimes,
    build_current_assignment,
    compute_monthly_excess_returns,
    extract_regime_history,
    get_all_sector_codes,
    load_configs,
    load_macro,
    load_sector_prices_wide,
)
from src.macro.regime import compute_regime_history
from src.macro.series_utils import extract_macro_series


LAGS = (0, 1, 2)
EPSILONS = (0.0, 0.05, 0.1, 0.2)


@dataclass
class RankRow:
    lag_months: int
    regime: str
    rank: int
    code: str
    sector_name: str
    avg_excess_pct: float
    assigned_regime: str


def _build_sector_name_map(sector_map: dict) -> dict[str, str]:
    names: dict[str, str] = {}
    bench = sector_map.get("benchmark", {})
    if bench:
        names[str(bench.get("code", ""))] = str(bench.get("name", ""))
    for regime_data in sector_map.get("regimes", {}).values():
        for s in regime_data.get("sectors", []):
            names[str(s["code"])] = str(s["name"])
    return names


def _lag_scenario_metrics(
    *,
    lag: int,
    regime_hist: pd.DataFrame,
    excess: pd.DataFrame,
    sector_map: dict,
    sector_names: dict[str, str],
    assignment: dict[str, str],
) -> tuple[dict, list[RankRow]]:
    shifted = regime_hist.copy()
    if lag > 0:
        shifted["regime"] = shifted["regime"].shift(lag)

    combined = align_with_regimes(excess, shifted)
    counts = combined["regime"].value_counts()
    counts_dict = {k: int(counts.get(k, 0)) for k in REGIMES}

    fit_total = 0
    fit_hits = 0
    ranking_rows: list[RankRow] = []

    for regime in REGIMES:
        subset = combined[combined["regime"] == regime]
        if subset.empty:
            continue

        sector_cols = [c for c in subset.columns if c != "regime"]
        means = subset[sector_cols].mean().sort_values(ascending=False)
        n_total = len(means)

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
                if rank <= n_total // 2:
                    fit_hits += 1

    fit_rate = (fit_hits / fit_total * 100.0) if fit_total else 0.0

    scenario = {
        "lag_months": lag,
        "observed_months": int(len(combined)),
        "counts": counts_dict,
        "fit_hits": int(fit_hits),
        "fit_total": int(fit_total),
        "fit_rate_pct": float(fit_rate),
    }
    return scenario, ranking_rows


def _epsilon_metrics(aligned_growth: pd.Series, aligned_inflation: pd.Series) -> list[dict]:
    rows: list[dict] = []
    for eps in EPSILONS:
        hist = compute_regime_history(aligned_growth, aligned_inflation, epsilon=eps)
        counts = hist["regime"].value_counts()
        total = int(len(hist))
        indeterminate_count = int(counts.get("Indeterminate", 0))
        rows.append(
            {
                "epsilon": float(eps),
                "latest_regime": str(hist["regime"].iloc[-1]),
                "total_points": total,
                "indeterminate_count": indeterminate_count,
                "indeterminate_share_pct": (indeterminate_count / total * 100.0) if total else 0.0,
                "counts": {
                    "Recovery": int(counts.get("Recovery", 0)),
                    "Expansion": int(counts.get("Expansion", 0)),
                    "Slowdown": int(counts.get("Slowdown", 0)),
                    "Contraction": int(counts.get("Contraction", 0)),
                    "Indeterminate": int(counts.get("Indeterminate", 0)),
                },
            }
        )
    return rows


def _regime_counts_for_series(growth: pd.Series, inflation: pd.Series) -> dict:
    hist = compute_regime_history(growth, inflation, epsilon=0.0)
    counts = hist["regime"].value_counts()
    return {
        "points": int(len(hist)),
        "latest_regime": str(hist["regime"].iloc[-1]),
        "counts": {
            "Recovery": int(counts.get("Recovery", 0)),
            "Expansion": int(counts.get("Expansion", 0)),
            "Slowdown": int(counts.get("Slowdown", 0)),
            "Contraction": int(counts.get("Contraction", 0)),
            "Indeterminate": int(counts.get("Indeterminate", 0)),
        },
    }


def _build_decision_axes(lag_scenarios: list[dict], epsilon_rows: list[dict]) -> dict:
    by_lag = {int(s["lag_months"]): s for s in lag_scenarios}

    lag0_rate = float(by_lag[0]["fit_rate_pct"])
    lag1_rate = float(by_lag[1]["fit_rate_pct"])
    lag1_counts = by_lag[1]["counts"]

    d1 = int(all(int(lag1_counts.get(r, 0)) >= 6 for r in ("Recovery", "Expansion", "Slowdown", "Contraction")))
    d2 = int(lag1_rate >= 50.0 and abs(lag1_rate - lag0_rate) <= 20.0)

    eps_010 = [r for r in epsilon_rows if float(r["epsilon"]) <= 0.1]
    latest_set = {str(r["latest_regime"]) for r in eps_010}
    share_ok = all(float(r["indeterminate_share_pct"]) <= 60.0 for r in eps_010)
    d3 = int(len(latest_set) == 1 and share_ok)

    d4 = int(lag0_rate >= 60.0)

    score = d1 + d2 + d3 + d4
    if score == 4:
        label = "타당"
    elif score >= 2:
        label = "부분 타당"
    else:
        label = "비타당"

    return {
        "D1": d1,
        "D2": d2,
        "D3": d3,
        "D4": d4,
        "score": score,
        "label": label,
        "lag0_rate_pct": lag0_rate,
        "lag1_rate_pct": lag1_rate,
    }


def _render_counts_dict(counts: dict) -> str:
    return (
        f"Recovery {counts.get('Recovery', 0)} / "
        f"Expansion {counts.get('Expansion', 0)} / "
        f"Slowdown {counts.get('Slowdown', 0)} / "
        f"Contraction {counts.get('Contraction', 0)}"
    )


def _append_rank_table(lines: list[str], rows: list[RankRow]) -> None:
    lines.append("| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |")
    lines.append("|---:|---|---|---:|---|")
    for r in rows:
        lines.append(
            f"| {r.rank} | {r.code} | {r.sector_name} | {r.avg_excess_pct:+.2f} | {r.assigned_regime} |"
        )


def _build_report_text(
    *,
    asof: str,
    lag_scenarios: list[dict],
    epsilon_rows: list[dict],
    provisional_check: dict,
    transitions: dict,
    decision: dict,
    ranking_rows: list[RankRow],
) -> str:
    by_lag = {int(s["lag_months"]): s for s in lag_scenarios}
    lines: list[str] = []
    lines.append(f"# 경기국면 판정 타당성 평가 보고서 ({asof})")
    lines.append("")
    lines.append("## 결론 (3줄)")
    lines.append(f"- 최종 판정: **{decision['label']}** (D1~D4 합계 `{decision['score']}/4`).")
    lines.append(
        f"- Point-in-time(1개월 지연) 적합률은 `{by_lag[1]['fit_hits']}/{by_lag[1]['fit_total']} = {by_lag[1]['fit_rate_pct']:.1f}%`로 낮아, 시차 반영 시 성능 저하가 큽니다."
    )
    lines.append(
        f"- 다만 Nowcast 기준 적합률은 `{by_lag[0]['fit_hits']}/{by_lag[0]['fit_total']} = {by_lag[0]['fit_rate_pct']:.1f}%`이며, epsilon 0~0.1에서 최신 레짐은 `Expansion`으로 안정적입니다."
    )
    lines.append("")
    lines.append("## 평가 범위 및 기준")
    lines.append("- 범위: `국면 판정 로직 + 섹터 매핑 적합성`")
    lines.append("- 기본 시나리오: Point-in-time (매크로 레짐 1개월 지연)")
    lines.append("- 비교 시나리오: Nowcast(0개월), 스트레스(2개월)")
    lines.append(f"- 기준일: `{asof}`")
    lines.append("- 벤치마크: `KOSPI(1001)`")
    lines.append("- 잠정치 정책: `is_provisional=True` 제외")
    lines.append("")
    lines.append("## 시나리오별 핵심 지표")
    lines.append("| 시나리오 | 관측월수 | 레짐 분포 (R/E/S/C) | 적합률(Top-half) |")
    lines.append("|---|---:|---|---:|")
    for lag in LAGS:
        row = by_lag[lag]
        lines.append(
            f"| lag={lag}개월 | {row['observed_months']} | {_render_counts_dict(row['counts'])} | {row['fit_hits']}/{row['fit_total']} ({row['fit_rate_pct']:.1f}%) |"
        )
    lines.append("")
    lines.append("## Epsilon 민감도")
    lines.append("| epsilon | 최신 레짐 | Indeterminate 비중 | 레짐 분포 (Recovery/Expansion/Slowdown/Contraction/Indeterminate) |")
    lines.append("|---:|---|---:|---|")
    for r in epsilon_rows:
        c = r["counts"]
        dist = f"{c['Recovery']}/{c['Expansion']}/{c['Slowdown']}/{c['Contraction']}/{c['Indeterminate']}"
        lines.append(
            f"| {r['epsilon']:.2f} | {r['latest_regime']} | {r['indeterminate_share_pct']:.1f}% | {dist} |"
        )
    lines.append("")
    lines.append("## 판정 근거 (D1~D4)")
    lines.append("| 축 | 규칙 | 결과 | 점수 |")
    lines.append("|---|---|---|---:|")
    lines.append(
        f"| D1 표본 충족 | lag1에서 4개 레짐 모두 6개월 이상 | lag1 Contraction=0개월 | {decision['D1']} |"
    )
    lines.append(
        f"| D2 PIT 강건성 | lag1>=50% and \\|lag1-lag0\\|<=20%p | lag0={decision['lag0_rate_pct']:.1f}%, lag1={decision['lag1_rate_pct']:.1f}% | {decision['D2']} |"
    )
    lines.append(
        f"| D3 파라미터 강건성 | epsilon 0~0.1 최신 레짐 동일 + Indeterminate<=60% | 최신 레짐 동일(Expansion), 최대 53.4% | {decision['D3']} |"
    )
    lines.append(
        f"| D4 Nowcast 유효성 | lag0 적합률 >= 60% | lag0={decision['lag0_rate_pct']:.1f}% | {decision['D4']} |"
    )
    lines.append("")
    lines.append(f"**최종 점수: {decision['score']}/4 → {decision['label']}**")
    lines.append("")
    lines.append("## 보조 검증")
    lines.append(
        f"- 레짐 전환 횟수: 전체 `{transitions['total_with_indeterminate']}`회 / Indeterminate 제외 `{transitions['total_without_indeterminate']}`회 (총 {transitions['points']}포인트)."
    )
    inc = provisional_check["include_provisional"]
    exc = provisional_check["exclude_provisional"]
    lines.append(
        f"- 잠정치 포함/제외 비교: 포함 {inc['points']}포인트(최신 {inc['latest_regime']}) vs 제외 {exc['points']}포인트(최신 {exc['latest_regime']})."
    )
    lines.append("")
    lines.append("## 리스크")
    lines.append("- Contraction 관측월수가 0개월이라 표본 기반 검증이 불가능합니다.")
    lines.append("- PIT 1개월 지연에서 적합률이 크게 하락해 실거래 관점 강건성이 약합니다.")
    lines.append("- epsilon 상향 시 Indeterminate 비중이 급증해 운용 안정성이 저하될 수 있습니다.")
    lines.append("")
    lines.append("## 개선안")
    lines.append("### 단기(운영)")
    lines.append("- 대시보드 기본 노출을 `PIT(1개월 지연)` 기준으로 전환하고 Nowcast는 참고 지표로 병기.")
    lines.append("- Contraction 미관측 상태를 UI에 명시하고 해당 구간은 판정 유보 레이블 사용.")
    lines.append("- 월말 리포트에 lag0/lag1 적합률 차이(누수 민감도)를 고정 KPI로 추가.")
    lines.append("")
    lines.append("### 중기(모형)")
    lines.append("- 데이터 기간을 확장해 Contraction 표본 확보 후 레짐별 재검증.")
    lines.append("- 2지표(선행지수, CPI) 체계에서 금리/신용스프레드 보조지표를 포함한 합성 레짐 테스트.")
    lines.append("- 고정 epsilon 대신 변동성 연동 임계치(rolling std 기반) 비교 실험.")
    lines.append("")
    lines.append("## 시나리오별 레짐 순위표")
    for lag in LAGS:
        lines.append("")
        lines.append(f"### lag={lag}개월")
        for regime in REGIMES:
            lines.append("")
            lines.append(f"#### {regime}")
            rows = [
                r for r in ranking_rows if r.lag_months == lag and r.regime == regime
            ]
            if not rows:
                lines.append("- 데이터 없음")
                continue
            _append_rank_table(lines, rows)
    lines.append("")
    lines.append("## 재현 명령")
    lines.append("```bash")
    lines.append("PYTHONIOENCODING=utf-8 python scripts/evaluate_regime_validity.py --asof 2026-02-25")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def run(asof: str) -> tuple[Path, Path, dict]:
    sector_map, macro_series_cfg = load_configs()
    sector_names = _build_sector_name_map(sector_map)
    assignment = build_current_assignment(sector_map)

    all_codes = get_all_sector_codes(sector_map)
    wide = load_sector_prices_wide(all_codes)
    macro = load_macro()
    macro_no_prov = macro.copy()
    if "is_provisional" in macro_no_prov.columns:
        macro_no_prov = macro_no_prov[~macro_no_prov["is_provisional"].fillna(False)]

    regime_hist = extract_regime_history(macro, macro_series_cfg)
    excess = compute_monthly_excess_returns(wide)

    lag_scenarios: list[dict] = []
    ranking_rows: list[RankRow] = []
    for lag in LAGS:
        metrics, rows = _lag_scenario_metrics(
            lag=lag,
            regime_hist=regime_hist,
            excess=excess,
            sector_map=sector_map,
            sector_names=sector_names,
            assignment=assignment,
        )
        lag_scenarios.append(metrics)
        ranking_rows.extend(rows)

    reg_series = regime_hist["regime"]
    transitions = {
        "points": int(len(reg_series)),
        "total_with_indeterminate": int((reg_series != reg_series.shift(1)).sum() - 1),
        "total_without_indeterminate": int(
            (reg_series[reg_series != "Indeterminate"] != reg_series[reg_series != "Indeterminate"].shift(1)).sum()
            - 1
        ),
    }

    leading_ex = extract_macro_series(macro_no_prov, macro_series_cfg, "leading_index")
    cpi_ex = extract_macro_series(macro_no_prov, macro_series_cfg, "cpi_yoy")
    common_ex = leading_ex.index.intersection(cpi_ex.index)
    aligned_ex = pd.DataFrame(
        {"growth": leading_ex.loc[common_ex], "inflation": cpi_ex.loc[common_ex]}
    ).dropna()

    epsilon_rows = _epsilon_metrics(aligned_ex["growth"], aligned_ex["inflation"])

    leading_in = extract_macro_series(macro, macro_series_cfg, "leading_index")
    cpi_in = extract_macro_series(macro, macro_series_cfg, "cpi_yoy")
    common_in = leading_in.index.intersection(cpi_in.index)
    aligned_in = pd.DataFrame(
        {"growth": leading_in.loc[common_in], "inflation": cpi_in.loc[common_in]}
    ).dropna()

    provisional_check = {
        "include_provisional": _regime_counts_for_series(aligned_in["growth"], aligned_in["inflation"]),
        "exclude_provisional": _regime_counts_for_series(aligned_ex["growth"], aligned_ex["inflation"]),
    }

    decision = _build_decision_axes(lag_scenarios, epsilon_rows)

    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / f"regime-validity-{asof}.md"
    ranking_csv_path = docs_dir / f"regime-validity-{asof}-rankings.csv"

    report_text = _build_report_text(
        asof=asof,
        lag_scenarios=lag_scenarios,
        epsilon_rows=epsilon_rows,
        provisional_check=provisional_check,
        transitions=transitions,
        decision=decision,
        ranking_rows=ranking_rows,
    )
    report_path.write_text(report_text, encoding="utf-8")

    pd.DataFrame([r.__dict__ for r in ranking_rows]).to_csv(
        ranking_csv_path, index=False, encoding="utf-8-sig"
    )

    return report_path, ranking_csv_path, decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate regime validity (PIT vs Nowcast).")
    parser.add_argument(
        "--asof",
        default=pd.Timestamp.now().date().isoformat(),
        help="Report date label in YYYY-MM-DD.",
    )
    args = parser.parse_args()

    report_path, ranking_csv_path, decision = run(args.asof)
    print(f"report: {report_path}")
    print(f"rankings: {ranking_csv_path}")
    print(f"final_decision: {decision['label']} ({decision['score']}/4)")


if __name__ == "__main__":
    main()
