"""test_metrics.py

Unit tests for retrieval metrics in eval.py.

We test the metric primitives (dcg, ndcg, hit_rate_at_k, true_recall_at_k,
reciprocal_rank) directly with hand-crafted inputs. These are pure functions
of labels/scores, so no model or data is needed.
"""

from __future__ import annotations

import math

import pytest

from eval import (
    GAINS,
    RELEVANT,
    dcg,
    hit_rate_at_k,
    ndcg,
    reciprocal_rank,
    true_recall_at_k,
)


# ---------------------------------------------------------------------------
# DCG / NDCG
# ---------------------------------------------------------------------------
def test_dcg_empty():
    assert dcg([], 10) == 0.0


def test_dcg_perfect_ranking():
    """Perfect ranking: all relevant items first, gains = [1, 0.1, 0.01, 0]."""
    gains = [1.0, 0.1, 0.01, 0.0]
    # DCG = 1/log2(2) + 0.1/log2(3) + 0.01/log2(4) + 0/log2(5)
    expected = 1.0 + 0.1 / math.log2(3) + 0.01 / math.log2(4)
    assert dcg(gains, 10) == pytest.approx(expected, rel=1e-9)


def test_dcg_truncates_at_k():
    """DCG@k should ignore items beyond position k."""
    gains = [1.0, 0.0, 0.0, 1.0]  # relevant at pos 0 and 3
    full = dcg(gains, 10)
    truncated = dcg(gains, 1)
    # Only first item counts in truncated version.
    assert truncated == pytest.approx(1.0, rel=1e-9)
    assert full > truncated


def test_ndcg_perfect_is_one():
    """If predicted ranking == ideal ranking, NDCG = 1."""
    pred = [1.0, 0.1, 0.01, 0.0]
    ideal = [1.0, 0.1, 0.01, 0.0]
    assert ndcg(pred, ideal, 10) == pytest.approx(1.0, rel=1e-9)


def test_ndcg_worst_is_less_than_perfect():
    """If predicted ranking is reverse of ideal, NDCG < 1.

    Note: NDCG is NOT exactly 0 for a reversed ranking because DCG still
    picks up some gain from items that happen to be in the right relative
    order. We only assert it's strictly less than the perfect NDCG.
    """
    pred = [0.0, 0.01, 0.1, 1.0]
    ideal = [1.0, 0.1, 0.01, 0.0]
    assert ndcg(pred, ideal, 10) < ndcg(ideal, ideal, 10)
    assert ndcg(pred, ideal, 10) >= 0.0


def test_ndcg_empty_ideal_is_zero():
    """If ideal is empty (no relevant items), NDCG is undefined -> 0."""
    assert ndcg([1.0, 0.0], [], 10) == 0.0


def test_ndcg_in_unit_interval():
    """NDCG must always be in [0, 1]."""
    pred = [0.1, 1.0, 0.0, 0.01]
    ideal = [1.0, 0.1, 0.01, 0.0]
    val = ndcg(pred, ideal, 10)
    assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# HitRate@K
# ---------------------------------------------------------------------------
def test_hit_rate_at_k_finds_exact():
    """If an 'E' label is in top-K, HitRate@K = 1."""
    labels = ["I", "C", "E", "S"]
    assert hit_rate_at_k(labels, 5) == 1.0
    assert hit_rate_at_k(labels, 3) == 1.0


def test_hit_rate_at_k_misses_exact():
    """If no 'E' in top-K, HitRate@K = 0."""
    labels = ["I", "C", "S", "E"]
    assert hit_rate_at_k(labels, 3) == 0.0
    assert hit_rate_at_k(labels, 2) == 0.0


def test_hit_rate_at_k_only_counts_E():
    """HitRate@K is specifically about 'E' (Exact) matches, not 'S'."""
    labels = ["S", "C", "I", "E"]
    assert hit_rate_at_k(labels, 3) == 0.0  # only S/C/I in top-3
    assert hit_rate_at_k(labels, 4) == 1.0


# ---------------------------------------------------------------------------
# TrueRecall@K
# ---------------------------------------------------------------------------
def test_true_recall_at_k_zero_total():
    """If total_exact == 0, recall is 0 (avoid div-by-zero)."""
    assert true_recall_at_k(["E", "S"], total_exact=0, k=5) == 0.0


def test_true_recall_at_k_partial():
    """If 2 of 4 exacts are in top-K, recall = 0.5."""
    labels = ["E", "I", "E", "C", "S", "E", "E"]
    # top-3 contains 2 E's
    assert true_recall_at_k(labels, total_exact=4, k=3) == pytest.approx(0.5)


def test_true_recall_at_k_full():
    """If all exacts are in top-K, recall = 1.0."""
    labels = ["E", "E", "E", "C"]
    assert true_recall_at_k(labels, total_exact=3, k=5) == 1.0


def test_true_recall_at_k_monotonic_in_k():
    """Recall@K is non-decreasing in K."""
    labels = ["I", "C", "E", "S", "E", "I", "E"]
    total = 3
    r1 = true_recall_at_k(labels, total, 1)
    r5 = true_recall_at_k(labels, total, 5)
    r10 = true_recall_at_k(labels, total, 10)
    assert r1 <= r5 <= r10


# ---------------------------------------------------------------------------
# ReciprocalRank
# ---------------------------------------------------------------------------
def test_reciprocal_rank_first_relevant():
    """First relevant at position 0 -> RR = 1.0."""
    assert reciprocal_rank(["E", "S", "C"]) == 1.0
    assert reciprocal_rank(["S", "C", "I"]) == 1.0  # S counts as relevant


def test_reciprocal_rank_second_relevant():
    """First relevant at position 1 -> RR = 0.5."""
    assert reciprocal_rank(["I", "E", "C"]) == pytest.approx(0.5)


def test_reciprocal_rank_no_relevant():
    """No relevant items -> RR = 0.0."""
    assert reciprocal_rank(["I", "C", "I"]) == 0.0


def test_reciprocal_rank_uses_RELEVANT_set():
    """Only labels in RELEVANT ({E, S}) count toward RR."""
    # 'C' is NOT in RELEVANT, so first relevant is 'S' at position 1.
    assert reciprocal_rank(["C", "S", "E"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# GAINS / RELEVANT consistency
# ---------------------------------------------------------------------------
def test_gains_keys_match_esci_labels():
    """GAINS must cover all 4 ESCI labels."""
    assert set(GAINS.keys()) == {"E", "S", "C", "I"}


def test_gains_ordering():
    """E > S > C > I in gain value."""
    assert GAINS["E"] > GAINS["S"] > GAINS["C"] > GAINS["I"]


def test_relevant_set_is_subset_of_gains():
    """RELEVANT must be a subset of GAINS keys."""
    assert RELEVANT.issubset(set(GAINS.keys()))
