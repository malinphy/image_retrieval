"""test_data_schema.py

Schema validation tests for the joined dataset and split parquets.

We use lightweight hand-rolled validation (no pandera dependency) to keep
the test surface minimal. The schema enforced here mirrors what
prepare_data.py and split_dataset.py are expected to produce.

Required columns and types for joined_dataset.parquet:
  - query_id    : int
  - query       : str (non-empty)
  - product_id  : str (non-empty)
  - esci_label  : str in {"E", "S", "C", "I"}

Required columns for split parquets (train/val/test):
  - All of the above, plus the file must be non-empty.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REQUIRED_COLUMNS = ["query_id", "query", "product_id", "esci_label"]
VALID_ESCI_LABELS = {"E", "S", "C", "I"}


def _validate_schema(df: pd.DataFrame, *, allow_empty: bool = False) -> list[str]:
    """Return a list of schema violations (empty list = valid)."""
    errors: list[str] = []

    # 1. Required columns present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
        return errors  # can't check further

    # 2. Non-empty (unless explicitly allowed)
    if not allow_empty and len(df) == 0:
        errors.append("DataFrame is empty")

    # 3. Column types
    if not pd.api.types.is_integer_dtype(df["query_id"]):
        errors.append(f"query_id must be integer, got {df['query_id'].dtype}")
    if not pd.api.types.is_string_dtype(df["query"]):
        errors.append(f"query must be string, got {df['query'].dtype}")
    if not pd.api.types.is_string_dtype(df["product_id"]):
        errors.append(f"product_id must be string, got {df['product_id'].dtype}")
    if not pd.api.types.is_string_dtype(df["esci_label"]):
        errors.append(f"esci_label must be string, got {df['esci_label'].dtype}")

    # 4. Non-null critical fields
    null_qid = df["query_id"].isna().sum()
    null_pid = df["product_id"].isna().sum()
    null_lbl = df["esci_label"].isna().sum()
    if null_qid:
        errors.append(f"query_id has {null_qid} nulls")
    if null_pid:
        errors.append(f"product_id has {null_pid} nulls")
    if null_lbl:
        errors.append(f"esci_label has {null_lbl} nulls")

    # 5. Non-empty strings
    empty_q = (df["query"].astype(str).str.len() == 0).sum()
    empty_p = (df["product_id"].astype(str).str.len() == 0).sum()
    if empty_q:
        errors.append(f"query has {empty_q} empty strings")
    if empty_p:
        errors.append(f"product_id has {empty_p} empty strings")

    # 6. Valid ESCI labels
    bad_labels = set(df["esci_label"].unique()) - VALID_ESCI_LABELS
    if bad_labels:
        errors.append(f"esci_label has invalid values: {bad_labels}")

    return errors


# ---------------------------------------------------------------------------
# Tests on synthetic data
# ---------------------------------------------------------------------------
def test_synthetic_pairs_passes_schema(synthetic_pairs_df):
    """The synthetic fixture must itself be schema-valid."""
    errors = _validate_schema(synthetic_pairs_df)
    assert errors == [], f"Synthetic fixture has schema errors: {errors}"


def test_synthetic_splits_pass_schema(synthetic_splits):
    """All three split parquets must pass schema validation."""
    for name, path in synthetic_splits.items():
        df = pd.read_parquet(path)
        errors = _validate_schema(df)
        assert errors == [], f"Split '{name}' has schema errors: {errors}"


# ---------------------------------------------------------------------------
# Negative tests (intentional violations)
# ---------------------------------------------------------------------------
def test_missing_columns_detected():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    errors = _validate_schema(df)
    assert any("Missing columns" in e for e in errors)


def test_invalid_esci_label_detected():
    df = pd.DataFrame(
        {
            "query_id": [1, 2],
            "query": ["a", "b"],
            "product_id": ["p1", "p2"],
            "esci_label": ["E", "X"],  # X is invalid
        }
    )
    errors = _validate_schema(df)
    assert any("invalid values" in e for e in errors)


def test_null_query_id_detected():
    df = pd.DataFrame(
        {
            "query_id": [1, None],
            "query": ["a", "b"],
            "product_id": ["p1", "p2"],
            "esci_label": ["E", "S"],
        }
    )
    errors = _validate_schema(df)
    assert any("query_id" in e and "null" in e for e in errors)


def test_empty_query_detected():
    df = pd.DataFrame(
        {
            "query_id": [1, 2],
            "query": ["a", ""],
            "product_id": ["p1", "p2"],
            "esci_label": ["E", "S"],
        }
    )
    errors = _validate_schema(df)
    assert any("query" in e and "empty" in e for e in errors)


def test_empty_dataframe_detected():
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    errors = _validate_schema(df)
    assert any("empty" in e for e in errors)


# ---------------------------------------------------------------------------
# Real data (skipped if not present)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "data" / "processed" / "splits").exists(),
    reason="data/processed/splits not present (run prepare_data.py + split_dataset.py first)",
)
def test_real_splits_pass_schema():
    """If real data exists, validate it. Skipped otherwise."""
    root = Path(__file__).resolve().parent.parent
    splits_dir = root / "data" / "processed" / "splits"
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        errors = _validate_schema(df)
        assert errors == [], f"Real split '{name}' has schema errors: {errors}"
