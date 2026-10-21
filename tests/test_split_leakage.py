"""test_split_leakage.py

Tests that train/val/test splits have NO query_id overlap.

Information leakage between splits is the #1 silent failure mode in
retrieval evaluation: if the same query appears in train and test, the
model "memorizes" the answer and test metrics become meaningless.

We test:
  1. The synthetic fixture has zero overlap (sanity).
  2. A hand-crafted leaky split is detected.
  3. Real splits (if present) have zero overlap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _query_id_sets(splits: dict[str, pd.DataFrame]) -> dict[str, set]:
    return {name: set(df["query_id"].unique()) for name, df in splits.items()}


def _assert_no_overlap(splits: dict[str, pd.DataFrame]) -> list[str]:
    """Return list of overlap violations (empty = clean)."""
    qid_sets = _query_id_sets(splits)
    names = list(qid_sets.keys())
    errors: list[str] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = qid_sets[a] & qid_sets[b]
            if overlap:
                errors.append(
                    f"query_id leakage between '{a}' and '{b}': "
                    f"{len(overlap)} shared query_ids (e.g. {sorted(overlap)[:5]})"
                )
    return errors


# ---------------------------------------------------------------------------
# Synthetic fixture (clean by construction)
# ---------------------------------------------------------------------------
def test_synthetic_splits_have_no_query_overlap(synthetic_splits):
    splits = {name: pd.read_parquet(path) for name, path in synthetic_splits.items()}
    errors = _assert_no_overlap(splits)
    assert errors == [], f"Synthetic splits have leakage: {errors}"


# ---------------------------------------------------------------------------
# Negative test: hand-crafted leaky split must be detected
# ---------------------------------------------------------------------------
def test_leaky_split_is_detected():
    """If we deliberately put query_id=1 in both train and test, the test fails."""
    train = pd.DataFrame(
        {
            "query_id": [1, 2, 3],
            "query": ["a", "b", "c"],
            "product_id": ["p1", "p2", "p3"],
            "esci_label": ["E", "E", "E"],
        }
    )
    val = pd.DataFrame(
        {
            "query_id": [4],
            "query": ["d"],
            "product_id": ["p4"],
            "esci_label": ["E"],
        }
    )
    test = pd.DataFrame(
        {
            "query_id": [1, 5],  # query_id=1 leaks from train!
            "query": ["a", "e"],
            "product_id": ["p1", "p5"],
            "esci_label": ["E", "E"],
        }
    )
    splits = {"train": train, "val": val, "test": test}
    errors = _assert_no_overlap(splits)
    assert len(errors) == 1
    assert "train" in errors[0] and "test" in errors[0]


# ---------------------------------------------------------------------------
# Real data (skipped if not present)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "data" / "processed" / "splits").exists(),
    reason="data/processed/splits not present",
)
def test_real_splits_have_no_query_overlap():
    root = Path(__file__).resolve().parent.parent
    splits_dir = root / "data" / "processed" / "splits"
    splits = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.parquet"
        if path.exists():
            splits[name] = pd.read_parquet(path)
    if len(splits) < 2:
        pytest.skip("Need at least 2 split files to test overlap")
    errors = _assert_no_overlap(splits)
    assert errors == [], f"Real splits have leakage: {errors}"


# ---------------------------------------------------------------------------
# Sanity: split sizes should be reasonable
# ---------------------------------------------------------------------------
def test_synthetic_split_sizes_are_positive(synthetic_splits):
    for name, path in synthetic_splits.items():
        df = pd.read_parquet(path)
        assert len(df) > 0, f"Split '{name}' is empty"
        assert df["query_id"].nunique() > 0, f"Split '{name}' has no unique queries"
