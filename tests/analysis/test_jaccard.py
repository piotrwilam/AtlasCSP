"""Unit tests for csp_atlas.analysis.jaccard on synthetic sets."""

from __future__ import annotations

import numpy as np
import pytest

from csp_atlas.analysis.jaccard import jaccard_sets, pairwise_jaccard_matrix


def test_jaccard_disjoint() -> None:
    assert jaccard_sets({1, 2, 3}, {4, 5}) == 0.0


def test_jaccard_identical() -> None:
    assert jaccard_sets({1, 2, 3}, {1, 2, 3}) == 1.0


def test_jaccard_partial() -> None:
    # |A ∩ B| = 2, |A ∪ B| = 4 → 0.5
    assert jaccard_sets({1, 2, 3}, {2, 3, 5}) == pytest.approx(0.5)


def test_jaccard_both_empty_returns_zero() -> None:
    assert jaccard_sets(set(), set()) == 0.0


def test_jaccard_one_empty_returns_zero() -> None:
    assert jaccard_sets({1, 2}, set()) == 0.0


def test_pairwise_matrix_shape_and_symmetry() -> None:
    masks = {"a": {1, 2, 3}, "b": {2, 3, 5}, "c": {4, 5}}
    names, m = pairwise_jaccard_matrix(masks)
    assert names == ["a", "b", "c"]
    assert m.shape == (3, 3)
    assert np.allclose(m, m.T)
    assert m[0, 1] == pytest.approx(0.5)
    assert m[0, 2] == 0.0
    assert m[1, 2] == pytest.approx(0.25)
    assert np.allclose(np.diag(m), [1.0, 1.0, 1.0])
