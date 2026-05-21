"""Three-way decomposition of universal mask A against checker mask B.

Paper §5: given a concept's universal circuit `A` and the keyword
token's checker circuit `B`, three disjoint partitions follow:

    concept_only = A \\ B    fires on the concept but not the bare token
    shared       = A ∩ B    fires on both
    token_only   = B \\ A    fires on the bare token but not the concept

The **concept fraction** = `|A \\ B| / |A|` is the central per-concept
metric — the §6 finding that AST circuits reach up to 62.5% concept-only
at mid-to-late layers, while builtins are near-zero.
"""

from __future__ import annotations


def decompose_sets(
    universal: set[int],
    checker: set[int],
) -> tuple[set[int], set[int], set[int]]:
    """Three-way decomposition `(concept_only, shared, token_only)`.

    Pure; inputs unchanged.
    """
    assert isinstance(universal, set), (
        f"universal must be a set, got {type(universal).__name__}"
    )
    assert isinstance(checker, set), (
        f"checker must be a set, got {type(checker).__name__}"
    )
    return universal - checker, universal & checker, checker - universal


def concept_fraction(
    concept_only_size: int,
    universal_size: int,
) -> float:
    """Concept fraction: `|A \\ B| / |A|`.

    Returns 0.0 when the universal mask is empty.
    """
    assert concept_only_size >= 0, (
        f"concept_only_size must be non-negative, got {concept_only_size}"
    )
    assert universal_size >= 0, (
        f"universal_size must be non-negative, got {universal_size}"
    )
    assert concept_only_size <= universal_size, (
        f"concept_only ({concept_only_size}) cannot exceed universal "
        f"({universal_size})"
    )
    if universal_size == 0:
        return 0.0
    return concept_only_size / universal_size
