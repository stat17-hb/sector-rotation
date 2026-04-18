"""
섹터-레짐 매핑 실증 검증 스크립트

현재 `config/sector_map.yml`의 섹터 배정이 실제 한국 시장 데이터와
얼마나 일치하는지 검증한다.

지원 경로:
- `legacy`: 2026-02 문서와 동일한 오래된 raw-regime 검증 경로
- `dashboard-parity`: dashboard 규칙(`confirmed_regime`, confirmation, KR inflation stitch)
  에 맞춘 현재 canonical 검증 경로
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.macro.regime import compute_regime_history
from src.macro.series_utils import (
    build_regime_history_from_macro,
    extract_macro_series,
    filter_macro_provisional_rows,
)
from src.transforms.resample import to_monthly_last

CURATED_DIR = PROJECT_ROOT / "data" / "curated"
CONFIG_DIR = PROJECT_ROOT / "config"
BENCHMARK_CODE = "1001"
DASHBOARD_RUNTIME_WINDOW_MONTHS = 60
REGIMES = ["Recovery", "Expansion", "Slowdown", "Contraction"]


def load_configs() -> tuple[dict, dict]:
    with open(CONFIG_DIR / "sector_map.yml", encoding="utf-8") as f:
        sector_map = yaml.safe_load(f)
    with open(CONFIG_DIR / "macro_series.yml", encoding="utf-8") as f:
        macro_series_cfg = yaml.safe_load(f)
    return sector_map, macro_series_cfg


def load_settings() -> dict:
    with open(CONFIG_DIR / "settings.yml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_sector_name_map(sector_map: dict) -> dict[str, str]:
    names: dict[str, str] = {BENCHMARK_CODE: sector_map["benchmark"]["name"]}
    for regime_data in sector_map["regimes"].values():
        for sector in regime_data.get("sectors", []):
            names[str(sector["code"])] = sector["name"]
    return names


def build_current_assignment(sector_map: dict) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for regime, regime_data in sector_map["regimes"].items():
        for sector in regime_data.get("sectors", []):
            assignment[str(sector["code"])] = regime
    return assignment


def get_all_sector_codes(sector_map: dict) -> list[str]:
    codes = [BENCHMARK_CODE]
    for regime_data in sector_map["regimes"].values():
        for sector in regime_data.get("sectors", []):
            codes.append(str(sector["code"]))
    return codes


def load_sector_prices_wide(all_codes: list[str]) -> pd.DataFrame:
    path = CURATED_DIR / "sector_prices.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\n"
            "app.py를 먼저 실행하여 데이터를 수집하세요: streamlit run app.py"
        )

    frame = pd.read_parquet(path)
    frame["index_code"] = frame["index_code"].astype(str)
    frame = frame[frame["index_code"].isin(all_codes)]
    if frame.empty:
        raise ValueError("sector_prices.parquet에 분석 가능한 섹터 데이터가 없습니다.")

    wide = frame.pivot_table(index=frame.index, columns="index_code", values="close", aggfunc="last")
    wide.columns.name = None
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def load_macro() -> pd.DataFrame:
    path = CURATED_DIR / "macro_monthly.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\n"
            "app.py를 먼저 실행하여 데이터를 수집하세요: streamlit run app.py"
        )
    return pd.read_parquet(path)


def normalize_validation_path(path: str | None) -> str:
    normalized = str(path or "legacy").strip().lower()
    if normalized not in {"legacy", "dashboard-parity"}:
        raise ValueError(f"Unsupported validation path: {path!r}")
    return normalized


def resolve_label_column(path: str, label_column: str | None = None) -> str:
    normalized = str(label_column or "auto").strip().lower()
    if normalized in {"regime", "confirmed_regime"}:
        return normalized
    if normalized != "auto":
        raise ValueError(f"Unsupported label column: {label_column!r}")
    return "confirmed_regime" if normalize_validation_path(path) == "dashboard-parity" else "regime"


def extract_regime_history(
    macro_df: pd.DataFrame,
    macro_series_cfg: dict,
    *,
    settings: dict | None = None,
    path: str = "legacy",
    include_provisional: bool = False,
    window_months: int | None = None,
    market_id: str = "KR",
) -> pd.DataFrame:
    """Return regime history for the requested validation path."""
    normalized_path = normalize_validation_path(path)
    if normalized_path == "dashboard-parity":
        return build_regime_history_from_macro(
            macro_df=macro_df,
            macro_series_cfg=macro_series_cfg,
            settings=settings or load_settings(),
            market_id=market_id,
            include_provisional=include_provisional,
            window_months=window_months,
        )

    filtered_macro = filter_macro_provisional_rows(
        macro_df,
        include_provisional=include_provisional,
    )
    leading = extract_macro_series(filtered_macro, macro_series_cfg, "leading_index")
    cpi = extract_macro_series(filtered_macro, macro_series_cfg, "cpi_yoy")
    if leading.empty:
        raise ValueError("leading_index 데이터를 추출하지 못했습니다.")
    if cpi.empty:
        raise ValueError("cpi_yoy 데이터를 추출하지 못했습니다.")

    common = leading.index.intersection(cpi.index)
    regime_hist = compute_regime_history(
        leading.loc[common],
        cpi.loc[common],
        epsilon=0.0,
        confirmation_periods=1,
    )
    if window_months is not None and window_months > 0 and len(regime_hist) > window_months:
        regime_hist = regime_hist.iloc[-window_months:]
    return regime_hist


def compute_monthly_excess_returns(wide_prices: pd.DataFrame) -> pd.DataFrame:
    monthly = to_monthly_last(wide_prices)
    returns = monthly.pct_change()
    if BENCHMARK_CODE not in returns.columns:
        raise ValueError(f"벤치마크({BENCHMARK_CODE}) 데이터가 없습니다.")
    benchmark = returns[BENCHMARK_CODE]
    sector_cols = [code for code in returns.columns if code != BENCHMARK_CODE]
    return returns[sector_cols].sub(benchmark, axis=0)


def align_with_regimes(
    excess_returns: pd.DataFrame,
    regime_hist: pd.DataFrame,
    *,
    label_col: str = "regime",
    drop_indeterminate: bool = True,
) -> pd.DataFrame:
    """Join monthly excess returns with a selected regime label column."""
    if label_col not in regime_hist.columns:
        raise KeyError(f"Label column {label_col!r} missing from regime history")

    excess_copy = excess_returns.copy()
    excess_copy.index = excess_copy.index.to_period("M")

    regime_copy = regime_hist[[label_col]].rename(columns={label_col: "regime"}).copy()
    if isinstance(regime_copy.index, pd.DatetimeIndex):
        regime_copy.index = regime_copy.index.to_period("M")

    combined = excess_copy.join(regime_copy, how="inner")
    combined = combined.dropna(subset=["regime"])
    if drop_indeterminate:
        combined = combined[combined["regime"] != "Indeterminate"]
    return combined


def summarize_regime_history(regime_hist: pd.DataFrame, *, label_col: str) -> dict[str, object]:
    if regime_hist.empty:
        return {
            "points": 0,
            "start": "",
            "end": "",
            "counts": {},
            "label_col": label_col,
        }
    labels = regime_hist[label_col].astype(str)
    return {
        "points": int(len(regime_hist)),
        "start": str(regime_hist.index.min()),
        "end": str(regime_hist.index.max()),
        "counts": labels.value_counts().to_dict(),
        "label_col": label_col,
    }


def _render_counts(counts: dict[str, int]) -> str:
    return (
        f"Recovery {int(counts.get('Recovery', 0))} / "
        f"Expansion {int(counts.get('Expansion', 0))} / "
        f"Slowdown {int(counts.get('Slowdown', 0))} / "
        f"Contraction {int(counts.get('Contraction', 0))} / "
        f"Indeterminate {int(counts.get('Indeterminate', 0))}"
    )


def print_regime_rankings(
    combined: pd.DataFrame,
    sector_names: dict[str, str],
    current_assignment: dict[str, str],
) -> tuple[dict[str, pd.Series], list[dict[str, object]]]:
    rankings: dict[str, pd.Series] = {}
    ranking_rows: list[dict[str, object]] = []
    for regime in REGIMES:
        subset = combined[combined["regime"] == regime]
        observed_months = len(subset)
        print(f"\n{'=' * 65}")
        print(f"  {regime}  (관측 월수: {observed_months}개월)")
        print(f"{'=' * 65}")

        if observed_months == 0:
            print("  데이터 없음")
            continue

        sector_cols = [column for column in subset.columns if column != "regime"]
        means = subset[sector_cols].mean().sort_values(ascending=False)
        rankings[regime] = means

        total = len(means)
        print(f"{'순위':<4} {'코드':<6} {'섹터명':<26} {'평균 초과수익':>12}  배정")
        print("-" * 65)
        for rank, (code, value) in enumerate(means.items(), start=1):
            name = sector_names.get(code, code)
            assigned = current_assignment.get(code, "미배정")
            tier = "상위" if rank <= total // 2 else "하위"
            marker = "OK" if assigned == regime else "--"
            ranking_rows.append(
                {
                    "regime": regime,
                    "rank": int(rank),
                    "code": str(code),
                    "sector_name": name,
                    "avg_excess_pct": float(value * 100.0),
                    "assigned_regime": assigned,
                    "is_top_half": bool(rank <= total // 2),
                }
            )
            print(
                f"{rank:<4} {code:<6} {name:<26} {value * 100:>+10.2f}%  "
                f"{marker} {assigned} ({tier})"
            )
    return rankings, ranking_rows


def build_rankings_from_combined(
    combined: pd.DataFrame,
    current_assignment: dict[str, str],
) -> list[dict[str, object]]:
    ranking_rows: list[dict[str, object]] = []
    for regime in REGIMES:
        subset = combined[combined["regime"] == regime]
        if subset.empty:
            continue
        sector_cols = [column for column in subset.columns if column != "regime"]
        means = subset[sector_cols].mean().sort_values(ascending=False)
        total = len(means)
        for rank, (code, value) in enumerate(means.items(), start=1):
            ranking_rows.append(
                {
                    "regime": regime,
                    "rank": int(rank),
                    "code": str(code),
                    "avg_excess_pct": float(value * 100.0),
                    "assigned_regime": current_assignment.get(str(code), "-"),
                    "is_top_half": bool(rank <= total // 2),
                }
            )
    return ranking_rows


def build_lagged_combined(
    *,
    excess: pd.DataFrame,
    regime_hist: pd.DataFrame,
    label_col: str,
    lag_months: int,
) -> pd.DataFrame:
    shifted = regime_hist.copy()
    shifted["_active_regime"] = shifted[label_col].shift(lag_months) if lag_months > 0 else shifted[label_col]
    return align_with_regimes(excess, shifted[["_active_regime"]], label_col="_active_regime")


def build_lag1_candidate_map(
    *,
    ranking_rows: list[dict[str, object]],
    capacity_by_regime: dict[str, int],
) -> dict[str, list[str]]:
    ranking_frame = pd.DataFrame(ranking_rows)
    if ranking_frame.empty:
        return {regime: [] for regime in REGIMES}

    top_half_by_regime: dict[str, set[str]] = {}
    for regime in REGIMES:
        rows = ranking_frame[ranking_frame["regime"] == regime].sort_values("rank")
        cutoff = len(rows) // 2
        top_half_by_regime[regime] = set(rows.head(cutoff)["code"].astype(str))

    codes = sorted(ranking_frame["code"].astype(str).unique().tolist())
    regime_order = [regime for regime in REGIMES if capacity_by_regime.get(regime, 0) > 0]
    best_score: int | None = None
    best_key: tuple[tuple[str, ...], ...] | None = None
    best_assignment: dict[str, list[str]] | None = None

    def _search(idx: int, remaining: list[str], current: dict[str, tuple[str, ...]]) -> None:
        nonlocal best_score, best_key, best_assignment
        if idx == len(regime_order):
            hits = 0
            for regime, assigned_codes in current.items():
                for code in assigned_codes:
                    if code in top_half_by_regime[regime]:
                        hits += 1
            key = tuple(tuple(current.get(regime, ())) for regime in regime_order)
            if best_score is None or hits > best_score or (hits == best_score and (best_key is None or key < best_key)):
                best_score = hits
                best_key = key
                best_assignment = {regime: list(assigned_codes) for regime, assigned_codes in current.items()}
            return

        regime = regime_order[idx]
        capacity = capacity_by_regime[regime]
        for combo in combinations(remaining, capacity):
            current[regime] = combo
            _search(idx + 1, [code for code in remaining if code not in combo], current)
            current.pop(regime, None)

    _search(0, codes, {})
    if best_assignment is None:
        return {regime: [] for regime in REGIMES}
    for regime in REGIMES:
        best_assignment.setdefault(regime, [])
    return best_assignment


def print_summary(rankings: dict[str, pd.Series], sector_map: dict, sector_names: dict[str, str]) -> None:
    print(f"\n{'=' * 65}")
    print("  현재 배정 적합성 요약")
    print(f"{'=' * 65}")

    total = 0
    hits = 0
    for regime in REGIMES:
        if regime not in rankings:
            continue
        means = rankings[regime]
        total_count = len(means)
        assigned_codes = [
            str(sector["code"])
            for sector in sector_map["regimes"].get(regime, {}).get("sectors", [])
        ]
        for code in assigned_codes:
            if code not in means.index:
                continue
            rank = means.index.get_loc(code) + 1
            is_top = rank <= total_count // 2
            total += 1
            if is_top:
                hits += 1
            name = sector_names.get(code, code)
            tier = "상위" if is_top else "하위"
            print(f"  [{regime:12s}] {name:<26} → 실증 순위 {rank:2d}/{total_count} ({tier})")

    if total <= 0:
        return
    fit_rate = hits / total * 100
    print(f"\n  배정 적합률: {hits}/{total} ({fit_rate:.0f}%)")
    if fit_rate >= 70:
        print("  판정: 현재 매핑이 실증적으로 대체로 지지됨")
    elif fit_rate >= 50:
        print("  판정: 현재 매핑이 부분적으로 지지됨 (일부 재검토 권장)")
    else:
        print("  판정: 현재 매핑이 실증적으로 지지되지 않음 (전면 재검토 필요)")


def build_sector_fit_risk_summary(
    *,
    rankings: list[dict[str, object]],
    sector_map: dict,
    regime_summary: dict[str, object],
    current_matches_candidate: bool = False,
    candidate_top_half_hits: int | None = None,
    candidate_overlap_rate: float | None = None,
    candidate_generic_overlap: bool | None = None,
) -> dict[str, object]:
    assignments = build_current_assignment(sector_map)
    total = 0
    hits = 0
    top_half_memberships = 0
    shared_top_half_memberships = 0
    top_half_by_code: dict[str, set[str]] = {}

    for row in rankings:
        if bool(row["is_top_half"]):
            top_half_memberships += 1
            top_half_by_code.setdefault(str(row["code"]), set()).add(str(row["regime"]))
        if assignments.get(str(row["code"])) == str(row["regime"]):
            total += 1
            if bool(row["is_top_half"]):
                hits += 1

    for regimes in top_half_by_code.values():
        if len(regimes) > 1:
            shared_top_half_memberships += len(regimes)

    overlap_rate = (
        float(shared_top_half_memberships / top_half_memberships)
        if top_half_memberships
        else 0.0
    )
    sample_counts = {
        regime: int(regime_summary["counts"].get(regime, 0))
        for regime in REGIMES
    }
    candidate_proposed = (
        candidate_top_half_hits is not None
        and candidate_overlap_rate is not None
        and candidate_generic_overlap is not None
    )
    gate_open = bool(
        candidate_proposed
        and candidate_top_half_hits >= 5
        and candidate_top_half_hits > hits
        and candidate_overlap_rate < overlap_rate
        and not candidate_generic_overlap
    )
    if current_matches_candidate and candidate_proposed and candidate_top_half_hits >= 5:
        decision = "current mapping accepted"
        reason = (
            f"Current mapping matches the reproducible lag1 candidate and that candidate "
            f"achieves {candidate_top_half_hits}/{total} top-half hits with overlap_rate {candidate_overlap_rate:.2f}."
        )
    elif gate_open:
        decision = "map follow-up justified later"
        reason = (
            f"Candidate remap would improve top-half hits to {candidate_top_half_hits}/{total} "
            f"with lower overlap_rate {candidate_overlap_rate:.2f}."
        )
    else:
        decision = "map remains unchanged"
        reason = (
            "No candidate remap is proposed in this pass; baseline fit remains "
            f"{hits}/{total} and overlap_rate is {overlap_rate:.2f}."
            if not candidate_proposed
            else (
                f"Candidate remap does not clear the gate "
                f"(hits {candidate_top_half_hits}/{total}, overlap_rate {candidate_overlap_rate:.2f}, "
                f"generic_overlap={candidate_generic_overlap})."
            )
        )
    return {
        "top_half_hits": int(hits),
        "top_half_total": int(total),
        "fit_rate_pct": float(hits / total * 100.0) if total else 0.0,
        "sample_counts": sample_counts,
        "top_half_memberships": int(top_half_memberships),
        "shared_top_half_memberships": int(shared_top_half_memberships),
        "overlap_rate": float(overlap_rate),
        "shared_leaders": sorted(
            code for code, regimes in top_half_by_code.items() if len(regimes) > 1
        ),
        "candidate_proposed": bool(candidate_proposed),
        "map_gate_open": gate_open,
        "map_gate_decision": decision,
        "map_gate_reason": reason,
    }


def summarize_candidate_assignment(
    *,
    candidate_assignment: dict[str, list[str]],
    ranking_rows: list[dict[str, object]],
) -> dict[str, object]:
    top_half_sets = {
        regime: {str(row["code"]) for row in ranking_rows if row["regime"] == regime and row["is_top_half"]}
        for regime in REGIMES
    }
    hits = 0
    total = 0
    top_half_memberships = 0
    shared_top_half_memberships = 0
    for regime, codes in candidate_assignment.items():
        for code in codes:
            total += 1
            if code in top_half_sets[regime]:
                hits += 1
            memberships = sum(code in top_half_sets[name] for name in REGIMES)
            top_half_memberships += memberships
            if memberships >= 2:
                shared_top_half_memberships += memberships
    overlap_rate = (
        float(shared_top_half_memberships / top_half_memberships)
        if top_half_memberships
        else 0.0
    )
    return {
        "candidate_top_half_hits": int(hits),
        "candidate_top_half_total": int(total),
        "candidate_fit_rate_pct": float(hits / total * 100.0) if total else 0.0,
        "candidate_overlap_rate": float(overlap_rate),
        "candidate_generic_overlap": bool(shared_top_half_memberships > top_half_memberships / 2) if top_half_memberships else False,
    }


def _report_path(path: str, asof: str) -> Path:
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    if path == "dashboard-parity":
        return docs_dir / f"sector-mapping-validation-dashboard-parity-{asof}.md"
    return docs_dir / f"sector-mapping-validation-{asof}.md"


def _sector_fit_risk_report_path(asof: str) -> Path:
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir / f"sector-fit-risk-closure-{asof}.md"


def _build_report_text(
    *,
    path: str,
    asof: str,
    label_col: str,
    include_provisional: bool,
    summary: dict[str, object],
    rankings: list[dict[str, object]],
) -> str:
    by_regime: dict[str, list[dict[str, object]]] = {regime: [] for regime in REGIMES}
    for row in rankings:
        by_regime[str(row["regime"])].append(row)

    lines: list[str] = []
    title = "섹터-레짐 매핑 실증 검증 보고서"
    if path == "dashboard-parity":
        title = "섹터-레짐 매핑 실증 검증 보고서 (dashboard-parity)"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"> 기준일: {asof}")
    lines.append(f"> path: `{path}`")
    lines.append(f"> label column: `{label_col}`")
    lines.append(f"> include provisional: `{include_provisional}`")
    lines.append(f"> replay window: `{summary['start']}` -> `{summary['end']}` ({summary['points']} points)")
    if path == "dashboard-parity":
        lines.append("> Historical note: `docs/sector-mapping-validation.md`는 2026-02 historical snapshot입니다.")
    lines.append("")
    lines.append("## Regime Counts")
    lines.append(f"- {_render_counts(summary['counts'])}")
    lines.append("")
    lines.append("## Ranking Tables")
    for regime in REGIMES:
        lines.append("")
        lines.append(f"### {regime}")
        rows = by_regime.get(regime, [])
        if not rows:
            lines.append("- 데이터 없음")
            continue
        lines.append("| 순위 | 코드 | 섹터명 | 평균 초과수익(%) | 현재 배정 |")
        lines.append("|---:|---|---|---:|---|")
        for row in rows:
            lines.append(
                f"| {row['rank']} | {row['code']} | {row['sector_name']} | {row['avg_excess_pct']:+.2f} | {row['assigned_regime']} |"
            )
    lines.append("")
    lines.append("## Cross-Regime Notes")
    cross_regime = {}
    for row in rankings:
        if bool(row["is_top_half"]):
            cross_regime.setdefault(str(row["code"]), []).append(str(row["regime"]))
    for code, regimes in sorted(cross_regime.items()):
        if len(regimes) > 1:
            lines.append(f"- `{code}` {', '.join(regimes)} 상위권")
    lines.append("")
    if path == "dashboard-parity":
        lines.append("## Interpretation")
        lines.append("- 이 표의 rank는 `lag0 nowcast empirical reference`입니다.")
        lines.append("- `PIT` 기준 액션이나 매핑 변경 근거로 바로 쓰지 않습니다.")
        lines.append("- static map은 별도 map gate를 통과하기 전까지 informational-only로 유지합니다.")
        lines.append("")
    lines.append("## Reproduction")
    lines.append("```bash")
    if path == "dashboard-parity":
        lines.append(
            f"python scripts/validate_sector_mapping.py --path dashboard-parity --label-column {label_col} --asof {asof}"
        )
    else:
        lines.append(f"python scripts/validate_sector_mapping.py --path legacy --label-column {label_col} --asof {asof}")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _build_sector_fit_risk_report_text(
    *,
    asof: str,
    label_col: str,
    include_provisional: bool,
    summary: dict[str, object],
    risk_summary: dict[str, object],
) -> str:
    lines: list[str] = []
    lines.append(f"# Sector-Fit Risk Closure ({asof})")
    lines.append("")
    lines.append(f"- replay window: `{summary['start']}` -> `{summary['end']}` ({summary['points']} points)")
    lines.append(f"- label column: `{label_col}`")
    lines.append(f"- include provisional: `{include_provisional}`")
    lines.append(
        f"- static map top-half hits: `{risk_summary['top_half_hits']}/{risk_summary['top_half_total']} = {risk_summary['fit_rate_pct']:.1f}%`"
    )
    lines.append(
        f"- overlap_rate: `{risk_summary['shared_top_half_memberships']}/{risk_summary['top_half_memberships']} = {risk_summary['overlap_rate']:.2f}`"
    )
    lines.append("")
    lines.append("## Sample Counts")
    for regime in REGIMES:
        lines.append(f"- {regime}: `{risk_summary['sample_counts'][regime]}`")
    lines.append("")
    lines.append("## Shared Leaders")
    if risk_summary["shared_leaders"]:
        for code in risk_summary["shared_leaders"]:
            lines.append(f"- `{code}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Lag1 Candidate Map")
    for regime in REGIMES:
        codes = list(risk_summary.get("candidate_assignment", {}).get(regime, []))
        lines.append(f"- {regime}: `{', '.join(codes) if codes else 'none'}`")
    lines.append(
        f"- candidate lag1 top-half hits: `{risk_summary.get('candidate_top_half_hits', 0)}/{risk_summary.get('candidate_top_half_total', 0)} = {risk_summary.get('candidate_fit_rate_pct', 0.0):.1f}%`"
    )
    lines.append(
        f"- candidate overlap_rate: `{risk_summary.get('candidate_overlap_rate', 0.0):.2f}`"
    )
    lines.append(
        f"- current mapping matches candidate: `{risk_summary.get('current_matches_candidate', False)}`"
    )
    lines.append("")
    lines.append("## Decision")
    lines.append("- Runtime empirical fit remains `lag0 nowcast empirical reference` only.")
    lines.append("- Static mapping is evaluated on the current canonical confirmed-regime path.")
    lines.append(f"- Map gate: `{risk_summary['map_gate_decision']}`")
    lines.append(f"- Reason: {risk_summary['map_gate_reason']}")
    lines.append("- Runtime empirical fit remains informational-only even when the static map improves materially.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("```bash")
    lines.append(f"python scripts/validate_sector_mapping.py --path dashboard-parity --label-column {label_col} --asof {asof}")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate KR sector-regime mapping.")
    parser.add_argument(
        "--path",
        default="legacy",
        choices=["legacy", "dashboard-parity"],
        help="Validation path to use.",
    )
    parser.add_argument(
        "--label-column",
        default="auto",
        choices=["auto", "regime", "confirmed_regime"],
        help="Regime label column to use for alignment.",
    )
    parser.add_argument(
        "--include-provisional",
        action="store_true",
        help="Keep provisional macro rows in the regime history.",
    )
    parser.add_argument(
        "--window-months",
        type=int,
        default=None,
        help="Optional trailing month window to apply to the regime history.",
    )
    parser.add_argument(
        "--asof",
        default=pd.Timestamp.now().date().isoformat(),
        help="Report label in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write a markdown report under docs/.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  섹터-레짐 매핑 실증 검증")
    print("=" * 65)

    sector_map, macro_series_cfg = load_configs()
    settings = load_settings()
    sector_names = build_sector_name_map(sector_map)
    current_assignment = build_current_assignment(sector_map)
    all_codes = get_all_sector_codes(sector_map)

    print("\n[1] 데이터 로드")
    wide_prices = load_sector_prices_wide(all_codes)
    macro_df = load_macro()
    available = sorted(wide_prices.columns.tolist())
    missing = [code for code in all_codes if code not in wide_prices.columns]
    print(f"  섹터 가격: {wide_prices.index[0].date()} ~ {wide_prices.index[-1].date()}")
    print(f"  사용 가능 코드: {available}")
    if missing:
        print(f"  [경고] 데이터 없는 코드: {missing}")

    print("\n[2] 레짐 이력 계산")
    label_col = resolve_label_column(args.path, args.label_column)
    regime_hist = extract_regime_history(
        macro_df,
        macro_series_cfg,
        settings=settings,
        path=args.path,
        include_provisional=args.include_provisional,
        window_months=args.window_months,
    )
    summary = summarize_regime_history(regime_hist, label_col=label_col)
    print(f"  path: {args.path}")
    print(f"  label_col: {label_col}")
    print(f"  include_provisional: {args.include_provisional}")
    print(f"  replay window: {summary['start']} -> {summary['end']} ({summary['points']} points)")
    for name, count in summary["counts"].items():
        print(f"  {name}: {count}개월")

    print("\n[3] 월별 초과수익 계산")
    excess = compute_monthly_excess_returns(wide_prices)
    combined = align_with_regimes(excess, regime_hist, label_col=label_col)
    if combined.empty:
        print("[오류] 레짐과 수익률의 공통 기간이 없습니다. 데이터를 재수집하세요.")
        sys.exit(1)
    print(f"  분석 월수: {len(combined)}개월")

    print("\n[4] 레짐별 섹터 순위 (평균 대 KOSPI 초과수익 기준)")
    rankings, ranking_rows = print_regime_rankings(combined, sector_names, current_assignment)
    print_summary(rankings, sector_map, sector_names)
    lag1_combined = build_lagged_combined(
        excess=excess,
        regime_hist=regime_hist,
        label_col=label_col,
        lag_months=1,
    )
    lag1_ranking_rows = build_rankings_from_combined(
        lag1_combined,
        current_assignment=current_assignment,
    )
    capacity_by_regime = {
        regime: len(sector_map["regimes"].get(regime, {}).get("sectors", []))
        for regime in REGIMES
    }
    candidate_assignment = build_lag1_candidate_map(
        ranking_rows=lag1_ranking_rows,
        capacity_by_regime=capacity_by_regime,
    )
    current_assignment_by_regime = {
        regime: [
            str(sector["code"])
            for sector in sector_map["regimes"].get(regime, {}).get("sectors", [])
        ]
        for regime in REGIMES
    }
    current_matches_candidate = all(
        sorted(current_assignment_by_regime.get(regime, [])) == sorted(candidate_assignment.get(regime, []))
        for regime in REGIMES
    )
    candidate_summary = summarize_candidate_assignment(
        candidate_assignment=candidate_assignment,
        ranking_rows=lag1_ranking_rows,
    )
    risk_summary = build_sector_fit_risk_summary(
        rankings=ranking_rows,
        sector_map=sector_map,
        regime_summary=summary,
        current_matches_candidate=current_matches_candidate,
        candidate_top_half_hits=candidate_summary["candidate_top_half_hits"],
        candidate_overlap_rate=candidate_summary["candidate_overlap_rate"],
        candidate_generic_overlap=candidate_summary["candidate_generic_overlap"],
    )

    should_write_report = bool(args.write_report or args.path == "dashboard-parity")
    if should_write_report:
        report_path = _report_path(args.path, args.asof)
        report_path.write_text(
            _build_report_text(
                path=args.path,
                asof=args.asof,
                label_col=label_col,
                include_provisional=bool(args.include_provisional),
                summary=summary,
                rankings=ranking_rows,
            ),
            encoding="utf-8",
        )
        print(f"\nreport: {report_path}")
        if args.path == "dashboard-parity":
            risk_report_path = _sector_fit_risk_report_path(args.asof)
            risk_report_path.write_text(
                _build_sector_fit_risk_report_text(
                    asof=args.asof,
                    label_col=label_col,
                    include_provisional=bool(args.include_provisional),
                    summary=summary,
                    risk_summary={
                        **risk_summary,
                        "current_matches_candidate": current_matches_candidate,
                        "candidate_assignment": candidate_assignment,
                        **candidate_summary,
                    },
                ),
                encoding="utf-8",
            )
            print(f"risk_report: {risk_report_path}")

    print("\n완료.")


if __name__ == "__main__":
    main()
