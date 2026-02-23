"""
Contract validators for Korea Sector Rotation Dashboard.

Two-function split (R5):
- validate_only(): read-only structural check, raises ContractError
- normalize_then_validate(): applies fill policy, then validates

Call convention:
- Loaders call normalize_then_validate() before saving to parquet
- Analytics entry points call validate_only() — receive already-normalized data
"""
from __future__ import annotations

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_object_dtype,
)


class ContractError(ValueError):
    """Raised when a DataFrame violates its schema contract."""


# Schema registry — type-family checks (not string literals)
SCHEMAS: dict[str, dict] = {
    "sector_prices": {
        "columns": {
            "index_code": is_object_dtype,
            "index_name": is_object_dtype,
            "close": is_float_dtype,
        },
        "index_type": "DatetimeIndex",
        "non_null": ["index_code", "close"],
        "fill_policy": {"close": ("ffill", 3)},
    },
    "macro_monthly": {
        "columns": {
            "series_id": is_object_dtype,
            "value": is_float_dtype,
            "source": is_object_dtype,
            "fetched_at": is_datetime64_any_dtype,
            "is_provisional": is_bool_dtype,
        },
        "index_type": "PeriodIndex",
        "non_null": ["series_id", "source", "is_provisional"],
        "fill_policy": {},
    },
    "signals": {
        "columns": {
            "index_code": is_object_dtype,
            "macro_regime": is_object_dtype,
            "macro_fit": is_bool_dtype,
            "action": is_object_dtype,
            "is_provisional": is_bool_dtype,
        },
        "index_type": None,
        "non_null": ["index_code", "action", "macro_fit"],
        "fill_policy": {},
    },
}


def validate_only(df: pd.DataFrame, schema_name: str) -> None:
    """Read-only structural check. Raises ContractError if schema violated.

    Does NOT modify df. Use at module entry points to fail fast.
    Called by analytics functions that receive already-normalized data.
    """
    if schema_name not in SCHEMAS:
        raise ContractError(f"Unknown schema: {schema_name!r}")

    schema = SCHEMAS[schema_name]

    # Check required columns and dtypes
    for col, type_checker in schema["columns"].items():
        if col not in df.columns:
            raise ContractError(
                f"Schema '{schema_name}': missing column {col!r}"
            )
        if not type_checker(df[col]):
            raise ContractError(
                f"Schema '{schema_name}': column {col!r} has wrong dtype "
                f"({df[col].dtype})"
            )

    # Check index type
    expected_index = schema["index_type"]
    if expected_index is not None:
        actual_index = type(df.index).__name__
        if actual_index != expected_index:
            raise ContractError(
                f"Schema '{schema_name}': expected index type "
                f"{expected_index!r}, got {actual_index!r}"
            )

    # Check non-null constraints
    for col in schema["non_null"]:
        if col in df.columns and df[col].isna().any():
            raise ContractError(
                f"Schema '{schema_name}': column {col!r} has null values "
                f"(non_null constraint violated)"
            )


def normalize_then_validate(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """Apply fill_policy (ffill + drop), then call validate_only.

    Returns new DataFrame. Use inside data loaders before saving to parquet.
    Does NOT mutate the input DataFrame.
    """
    if schema_name not in SCHEMAS:
        raise ContractError(f"Unknown schema: {schema_name!r}")

    schema = SCHEMAS[schema_name]
    result = df.copy()

    # Apply fill policies
    for col, policy in schema["fill_policy"].items():
        if col not in result.columns:
            continue
        method, limit = policy
        if method == "ffill":
            result[col] = result[col].ffill(limit=limit)
            # Drop rows where fill was insufficient (still NaN)
            result = result.dropna(subset=[col])

    validate_only(result, schema_name)
    return result
