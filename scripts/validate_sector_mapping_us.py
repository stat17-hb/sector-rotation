"""Validate the US sector-regime mapping against cached market and macro data."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.markets import load_market_configs
from src.macro.regime import compute_regime_history
from src.macro.series_utils import extract_macro_series


CURATED_DIR = PROJECT_ROOT / "data" / "curated"
PRICE_PATH = CURATED_DIR / "sector_prices_us.parquet"
MACRO_PATH = CURATED_DIR / "macro_monthly_us.parquet"
REGIMES = ["Recovery", "Expansion", "Slowdown", "Contraction"]


def _load_prices() -> pd.DataFrame:
    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"Missing {PRICE_PATH}")
    frame = pd.read_parquet(PRICE_PATH)
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def _load_macro() -> pd.DataFrame:
    if not MACRO_PATH.exists():
        raise FileNotFoundError(f"Missing {MACRO_PATH}")
    frame = pd.read_parquet(MACRO_PATH)
    if not isinstance(frame.index, pd.PeriodIndex):
        frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame


def _wide_prices(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.pivot_table(index=frame.index, columns="index_code", values="close", aggfunc="last").sort_index()


def _monthly_excess_returns(wide_prices: pd.DataFrame, benchmark_code: str) -> pd.DataFrame:
    monthly = wide_prices.resample("ME").last()
    returns = monthly.pct_change()
    benchmark = returns[benchmark_code]
    return returns.drop(columns=[benchmark_code]).sub(benchmark, axis=0)


def _regime_history(macro_df: pd.DataFrame, macro_series_cfg: dict) -> pd.DataFrame:
    growth = extract_macro_series(macro_df, macro_series_cfg, "leading_index")
    inflation = extract_macro_series(macro_df, macro_series_cfg, "cpi_mom")
    if inflation.empty:
        inflation = extract_macro_series(macro_df, macro_series_cfg, "cpi_yoy")
    common = growth.index.intersection(inflation.index)
    return compute_regime_history(growth.loc[common], inflation.loc[common], epsilon=0.0)


def main() -> int:
    settings, sector_map, macro_series_cfg, _profile = load_market_configs("US")
    benchmark_code = str(settings.get("benchmark_code", "SPY"))

    prices = _load_prices()
    macro_df = _load_macro()
    excess = _monthly_excess_returns(_wide_prices(prices), benchmark_code)
    regimes = _regime_history(macro_df, macro_series_cfg)
    if isinstance(regimes.index, pd.DatetimeIndex):
        regimes.index = regimes.index.to_period("M")
    excess.index = excess.index.to_period("M")
    combined = excess.join(regimes[["regime"]], how="inner")
    combined = combined[combined["regime"] != "Indeterminate"]

    for regime in REGIMES:
        subset = combined[combined["regime"] == regime]
        print(f"\n=== {regime} ({len(subset)} months) ===")
        if subset.empty:
            print("No observations.")
            continue
        rankings = subset.drop(columns=["regime"]).mean().sort_values(ascending=False)
        for code, value in rankings.items():
            print(f"{code}: {value * 100:+.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
