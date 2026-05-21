"""Unit tests for csp_atlas.analysis.decomposition."""

from __future__ import annotations

import pytest

from csp_atlas.analysis.decomposition import concept_fraction, decompose_sets


def test_decompose_disjoint() -> None:
    co, sh, to = decompose_sets({1, 2, 3}, {4, 5, 6})
    assert co == {1, 2, 3}
    assert sh == set()
    assert to == {4, 5, 6}


def test_decompose_partial() -> None:
    co, sh, to = decompose_sets({1, 2, 3, 4}, {3, 4, 5, 6})
    assert co == {1, 2}
    assert sh == {3, 4}
    assert to == {5, 6}


def test_decompose_partitions_are_disjoint() -> None:
    A, B = {1, 2, 3, 4, 5}, {3, 4, 5, 6, 7}
    co, sh, to = decompose_sets(A, B)
    assert co & sh == set() and co & to == set() and sh & to == set()
    assert co | sh | to == A | B


def test_concept_fraction_partial() -> None:
    assert concept_fraction(10, 40) == pytest.approx(0.25)


def test_concept_fraction_empty_universal() -> None:
    assert concept_fraction(0, 0) == 0.0


def test_concept_fraction_rejects_co_larger_than_universal() -> None:
    with pytest.raises(AssertionError):
        concept_fraction(10, 5)
