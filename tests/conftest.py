"""conftest.py

Shared pytest fixtures for the IMAGE_RETRIEVAL test suite.

Goals:
  - Keep tests independent of large model downloads (use tiny CLIP / mocks).
  - Provide deterministic, in-memory synthetic data for fast unit tests.
  - Make GPU optional: tests must run on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Make project root importable so `import train` works without install.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _seed_everything():
    """Seed all RNGs before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    yield


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_pairs_df() -> pd.DataFrame:
    """Tiny synthetic (query, product, label) DataFrame for unit tests.

    Layout:
      - 4 unique queries (q1..q4)
      - 5 unique products (p1..p5)
      - 8 rows total, with intentional duplicates to test multi-positive masking
    """
    rows = [
        # q1 -> p1 (E), p2 (E)  -> q1 has 2 positives
        {"query_id": 1, "query": "red shoes", "product_id": "p1", "esci_label": "E"},
        {"query_id": 1, "query": "red shoes", "product_id": "p2", "esci_label": "E"},
        # q2 -> p3 (E)
        {"query_id": 2, "query": "blue hat", "product_id": "p3", "esci_label": "E"},
        # q3 -> p4 (E), p5 (S)  -> only E counts as positive in CLIPDataset
        {"query_id": 3, "query": "green bag", "product_id": "p4", "esci_label": "E"},
        {"query_id": 3, "query": "green bag", "product_id": "p5", "esci_label": "S"},
        # q4 -> p1 (E)  -> duplicate product p1 across queries (multi-positive)
        {"query_id": 4, "query": "scarlet sneakers", "product_id": "p1", "esci_label": "E"},
        # q2 -> p2 (C)  -> non-positive
        {"query_id": 2, "query": "blue hat", "product_id": "p2", "esci_label": "C"},
        # q4 -> p3 (I)
        {"query_id": 4, "query": "scarlet sneakers", "product_id": "p3", "esci_label": "I"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_splits(tmp_path: Path, synthetic_pairs_df: pd.DataFrame) -> dict[str, Path]:
    """Write train/val/test parquet files with NO query_id overlap."""
    rng = np.random.default_rng(0)
    qids = synthetic_pairs_df["query_id"].unique()
    rng.shuffle(qids)
    train_q = set(qids[:2])  # q1, q2
    val_q = set(qids[2:3])  # q3
    test_q = set(qids[3:])  # q4

    paths: dict[str, Path] = {}
    for name, qset in [("train", train_q), ("val", val_q), ("test", test_q)]:
        part = synthetic_pairs_df[synthetic_pairs_df["query_id"].isin(qset)].reset_index(drop=True)
        p = tmp_path / f"{name}.parquet"
        part.to_parquet(p, index=False)
        paths[name] = p
    return paths


@pytest.fixture
def tiny_text_emb() -> np.ndarray:
    """Deterministic 4x4 L2-normalized text embeddings."""
    rng = np.random.default_rng(0)
    e = rng.standard_normal((4, 4)).astype(np.float32)
    return e / np.linalg.norm(e, axis=-1, keepdims=True)


@pytest.fixture
def tiny_image_emb() -> np.ndarray:
    """Deterministic 5x4 L2-normalized image embeddings."""
    rng = np.random.default_rng(1)
    e = rng.standard_normal((5, 4)).astype(np.float32)
    return e / np.linalg.norm(e, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Device fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def cpu_device() -> str:
    """Force CPU for all tests (CI-friendly)."""
    return "cpu"
