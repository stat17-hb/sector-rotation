"""
섹터-레짐 매핑 실증 검증 스크립트

현재 config/sector_map.yml의 하드코딩된 섹터 배정이 실제 한국 시장 데이터와
얼마나 일치하는지 검증한다.

방법론:
  1. KOSIS 경기선행지수(leading_index) + CPI YoY로 월별 레짐 레이블 계산
  2. 전 KRX 섹터 인덱스의 월별 수익률 계산
  3. KOSPI 대비 초과수익 집계
  4. 레짐별 섹터 순위 vs 현재 배정 비교

사전 조건:
  - data/curated/sector_prices.parquet  (app.py 실행 후 생성)
  - data/curated/macro_monthly.parquet  (app.py 실행 후 생성)
"""
from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.macro.regime import compute_regime_history
from src.macro.series_utils import extract_macro_series
from src.transforms.resample import to_monthly_last

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
CURATED_DIR = PROJECT_ROOT / "data" / "curated"
CONFIG_DIR = PROJECT_ROOT / "config"
BENCHMARK_CODE = "1001"  # KOSPI

REGIMES = ["Recovery", "Expansion", "Slowdown", "Contraction"]


# ---------------------------------------------------------------------------
# 설정 로드
# ---------------------------------------------------------------------------

def load_configs() -> tuple[dict, dict]:
    with open(CONFIG_DIR / "sector_map.yml", encoding="utf-8") as f:
        sector_map = yaml.safe_load(f)
    with open(CONFIG_DIR / "macro_series.yml", encoding="utf-8") as f:
        macro_series_cfg = yaml.safe_load(f)
    return sector_map, macro_series_cfg


def build_sector_name_map(sector_map: dict) -> dict[str, str]:
    """code → 섹터명"""
    names: dict[str, str] = {BENCHMARK_CODE: sector_map["benchmark"]["name"]}
    for regime_data in sector_map["regimes"].values():
        for s in regime_data.get("sectors", []):
            names[str(s["code"])] = s["name"]
    return names


def build_current_assignment(sector_map: dict) -> dict[str, str]:
    """code → 현재 배정 레짐"""
    assignment: dict[str, str] = {}
    for regime, regime_data in sector_map["regimes"].items():
        for s in regime_data.get("sectors", []):
            assignment[str(s["code"])] = regime
    return assignment


def get_all_sector_codes(sector_map: dict) -> list[str]:
    """sector_map에 정의된 모든 섹터 코드 + 벤치마크"""
    codes = [BENCHMARK_CODE]
    for regime_data in sector_map["regimes"].values():
        for s in regime_data.get("sectors", []):
            codes.append(str(s["code"]))
    return codes


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def load_sector_prices_wide(all_codes: list[str]) -> pd.DataFrame:
    """
    sector_prices.parquet (long format: DatetimeIndex, [index_code, index_name, close])
    → wide format: DatetimeIndex × index_code
    """
    path = CURATED_DIR / "sector_prices.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\n"
            "app.py를 먼저 실행하여 데이터를 수집하세요: streamlit run app.py"
        )

    df = pd.read_parquet(path)
    df["index_code"] = df["index_code"].astype(str)

    # 분석 대상 코드만 필터
    df = df[df["index_code"].isin(all_codes)]

    if df.empty:
        raise ValueError("sector_prices.parquet에 분석 가능한 섹터 데이터가 없습니다.")

    # long → wide pivot
    wide = df.pivot_table(index=df.index, columns="index_code", values="close", aggfunc="last")
    wide.columns.name = None
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    return wide


def load_macro() -> pd.DataFrame:
    path = CURATED_DIR / "macro_monthly.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\n"
            "app.py를 먼저 실행하여 데이터를 수집하세요: streamlit run app.py"
        )
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# 분석 로직
# ---------------------------------------------------------------------------

def extract_regime_history(macro_df: pd.DataFrame, macro_series_cfg: dict) -> pd.DataFrame:
    """잠정치 제외 후 레짐 이력 계산."""
    # is_provisional 행 제거
    if "is_provisional" in macro_df.columns:
        macro_df = macro_df[~macro_df["is_provisional"].fillna(False)]

    leading = extract_macro_series(macro_df, macro_series_cfg, "leading_index")
    cpi = extract_macro_series(macro_df, macro_series_cfg, "cpi_yoy")

    if leading.empty:
        raise ValueError("leading_index 데이터를 추출하지 못했습니다.")
    if cpi.empty:
        raise ValueError("cpi_yoy 데이터를 추출하지 못했습니다.")

    # 공통 기간
    common = leading.index.intersection(cpi.index)
    regime_hist = compute_regime_history(leading.loc[common], cpi.loc[common], epsilon=0.0)
    return regime_hist


def compute_monthly_excess_returns(
    wide_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    일별 wide → 월말 종가 → 월수익률 → 대 KOSPI 초과수익
    반환: DatetimeIndex(월말) × sector_code, 값=초과수익(소수)
    """
    monthly = to_monthly_last(wide_prices)
    returns = monthly.pct_change()

    if BENCHMARK_CODE not in returns.columns:
        raise ValueError(f"벤치마크({BENCHMARK_CODE}) 데이터가 없습니다.")

    bench = returns[BENCHMARK_CODE]
    sector_cols = [c for c in returns.columns if c != BENCHMARK_CODE]
    excess = returns[sector_cols].sub(bench, axis=0)
    return excess


def align_with_regimes(
    excess_returns: pd.DataFrame,
    regime_hist: pd.DataFrame,
) -> pd.DataFrame:
    """
    excess_returns (DatetimeIndex 월말) × regime_hist (PeriodIndex or DatetimeIndex)
    → 공통 월 기준 결합, 'regime' 열 추가
    """
    # 두 인덱스를 Period('M') 기준으로 정렬
    excess_copy = excess_returns.copy()
    excess_copy.index = excess_copy.index.to_period("M")

    regime_copy = regime_hist[["regime"]].copy()
    if isinstance(regime_copy.index, pd.DatetimeIndex):
        regime_copy.index = regime_copy.index.to_period("M")

    combined = excess_copy.join(regime_copy, how="inner")
    combined = combined.dropna(subset=["regime"])
    # Indeterminate 제외
    combined = combined[combined["regime"] != "Indeterminate"]
    return combined


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------

def print_regime_rankings(
    combined: pd.DataFrame,
    sector_names: dict[str, str],
    current_assignment: dict[str, str],
) -> dict[str, pd.Series]:
    """레짐별 섹터 평균 초과수익 순위표를 출력하고 결과 dict 반환."""
    rankings: dict[str, pd.Series] = {}

    for regime in REGIMES:
        subset = combined[combined["regime"] == regime]
        n = len(subset)
        print(f"\n{'='*65}")
        print(f"  {regime}  (관측 월수: {n}개월)")
        print(f"{'='*65}")

        if n == 0:
            print("  데이터 없음")
            continue

        sector_cols = [c for c in subset.columns if c != "regime"]
        means = subset[sector_cols].mean().sort_values(ascending=False)
        rankings[regime] = means

        n_total = len(means)
        header = f"{'순위':<4} {'코드':<6} {'섹터명':<26} {'평균 초과수익':>12}  배정"
        print(header)
        print("-" * 65)

        for rank, (code, val) in enumerate(means.items(), 1):
            name = sector_names.get(code, code)
            assigned = current_assignment.get(code, "미배정")
            is_match = assigned == regime
            marker = "✓" if is_match else "✗"
            tier = "상위" if rank <= n_total // 2 else "하위"
            print(
                f"{rank:<4} {code:<6} {name:<26} {val*100:>+10.2f}%"
                f"  {marker} {assigned} ({tier})"
            )

    return rankings


def print_summary(
    rankings: dict[str, pd.Series],
    sector_map: dict,
    sector_names: dict[str, str],
) -> None:
    print(f"\n{'='*65}")
    print("  현재 배정 적합성 요약")
    print(f"{'='*65}")

    total = 0
    hits = 0

    for regime in REGIMES:
        if regime not in rankings:
            continue
        means = rankings[regime]
        n_total = len(means)
        assigned_codes = [
            str(s["code"])
            for s in sector_map["regimes"].get(regime, {}).get("sectors", [])
        ]

        for code in assigned_codes:
            if code not in means.index:
                continue
            rank = means.index.get_loc(code) + 1
            is_top = rank <= n_total // 2
            total += 1
            if is_top:
                hits += 1
            name = sector_names.get(code, code)
            tier = "상위" if is_top else "하위"
            print(f"  [{regime:12s}] {name:<26} → 실증 순위 {rank:2d}/{n_total} ({tier})")

    if total > 0:
        rate = hits / total * 100
        print(f"\n  배정 적합률: {hits}/{total} ({rate:.0f}%)")
        if rate >= 70:
            print("  판정: 현재 매핑이 실증적으로 대체로 지지됨")
        elif rate >= 50:
            print("  판정: 현재 매핑이 부분적으로 지지됨 (일부 재검토 권장)")
        else:
            print("  판정: 현재 매핑이 실증적으로 지지되지 않음 (전면 재검토 필요)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  섹터-레짐 매핑 실증 검증")
    print("=" * 65)

    # 설정
    sector_map, macro_series_cfg = load_configs()
    sector_names = build_sector_name_map(sector_map)
    current_assignment = build_current_assignment(sector_map)
    all_codes = get_all_sector_codes(sector_map)

    # 데이터 로드
    print("\n[1] 데이터 로드")
    try:
        wide_prices = load_sector_prices_wide(all_codes)
    except (FileNotFoundError, ValueError) as e:
        print(f"[오류] {e}")
        sys.exit(1)

    try:
        macro_df = load_macro()
    except FileNotFoundError as e:
        print(f"[오류] {e}")
        sys.exit(1)

    available = sorted(wide_prices.columns.tolist())
    missing = [c for c in all_codes if c not in wide_prices.columns]
    print(f"  섹터 가격: {wide_prices.index[0].date()} ~ {wide_prices.index[-1].date()}")
    print(f"  사용 가능 코드: {available}")
    if missing:
        print(f"  [경고] 데이터 없는 코드: {missing}")

    # 레짐 이력
    print("\n[2] 레짐 이력 계산")
    try:
        regime_hist = extract_regime_history(macro_df, macro_series_cfg)
    except ValueError as e:
        print(f"[오류] {e}")
        sys.exit(1)

    regime_counts = regime_hist["regime"].value_counts()
    for r, cnt in regime_counts.items():
        print(f"  {r}: {cnt}개월")

    # 초과수익 계산
    print("\n[3] 월별 초과수익 계산")
    try:
        excess = compute_monthly_excess_returns(wide_prices)
    except ValueError as e:
        print(f"[오류] {e}")
        sys.exit(1)

    combined = align_with_regimes(excess, regime_hist)
    if combined.empty:
        print("[오류] 레짐과 수익률의 공통 기간이 없습니다. 데이터를 재수집하세요.")
        sys.exit(1)

    print(f"  분석 월수: {len(combined)}개월")

    # 순위표 출력
    print("\n[4] 레짐별 섹터 순위 (평균 대 KOSPI 초과수익 기준)")
    rankings = print_regime_rankings(combined, sector_names, current_assignment)

    # 요약
    print_summary(rankings, sector_map, sector_names)

    print("\n완료.")


if __name__ == "__main__":
    main()
