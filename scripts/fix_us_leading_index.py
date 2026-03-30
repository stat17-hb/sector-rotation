"""
One-time migration: delete stale USALOLITONOSTSAM data stored under the
'leading_index' alias for the US market, so the next app load fetches
fresh USSLIND (Philadelphia Fed Leading Index) data from scratch.

Usage:
    python scripts/fix_us_leading_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_sources.warehouse import warehouse_exists, _connect


def main() -> None:
    if not warehouse_exists():
        print("Warehouse not found — nothing to fix.")
        return

    con = _connect(read_only=False)
    try:
        count = con.execute(
            "SELECT COUNT(*) FROM fact_macro_monthly WHERE market = 'US' AND series_alias = 'leading_index'"
        ).fetchone()[0]
        if count == 0:
            print("No stale US leading_index rows found — already clean.")
            return
        con.execute(
            "DELETE FROM fact_macro_monthly WHERE market = 'US' AND series_alias = 'leading_index'"
        )
        print(f"Deleted {count} stale US leading_index rows (USALOLITONOSTSAM).")
    finally:
        con.close()

    print("Done. Restart the app - it will fetch fresh USSLIND data on next load.")


if __name__ == "__main__":
    main()
