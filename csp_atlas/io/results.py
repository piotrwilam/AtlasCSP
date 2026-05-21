"""Read the pre-computed analysis-result CSVs.

Three families of CSV are produced by the analysis notebooks:
    14_summary.csv                          — overall (ε, cons) sweep summary
    14_target_checker_eps{ε}_cons{c}.csv    — per-(concept, layer, ε, cons) detail
    relaxed_modularity_{scores,detail}.csv  — §6.1 modularity finding
    token_independence_{summary,detail,no_keyword}.csv — §6 token-independence
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from csp_atlas.paths import DATA_ROOT


def load_summary(*, data_root: Path | None = None) -> pd.DataFrame:
    """The §7 overall parameter-sweep summary table (14_summary.csv)."""
    root = Path(data_root) if data_root is not None else DATA_ROOT
    path = root / "14_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected summary file not found: {path}")
    return pd.read_csv(path)


def load_target_checker(
    *,
    eps: float = 0.5,
    cons: float = 0.8,
    data_root: Path | None = None,
) -> pd.DataFrame:
    """Per-(concept, layer) target / checker statistics at one (ε, cons) cell."""
    root = Path(data_root) if data_root is not None else DATA_ROOT
    path = root / f"14_target_checker_eps{eps}_cons{cons}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected target-checker file not found: {path}")
    return pd.read_csv(path)


def load_modularity_scores(
    *,
    data_root: Path | None = None,
    detail: bool = False,
) -> pd.DataFrame:
    """§6.1 relaxed-modularity scores per concept.

    If `detail=True`, returns the long-form per-(object, layer, p_trim) table.
    Otherwise returns the summary table (one row per concept, columns =
    significant-layer count at each p ∈ {0.0, 0.05, 0.10, 0.25}).
    """
    root = Path(data_root) if data_root is not None else DATA_ROOT
    fname = "relaxed_modularity_detail.csv" if detail else "relaxed_modularity_scores.csv"
    path = root / fname
    if not path.exists():
        raise FileNotFoundError(f"Expected modularity file not found: {path}")
    return pd.read_csv(path, index_col=0 if not detail else None)


def load_token_independence(
    *,
    data_root: Path | None = None,
    kind: str = "summary",
) -> pd.DataFrame:
    """§6 token-independence finding.

    `kind`:
        "summary"     — aggregate result per concept
        "detail"      — per-(concept, layer) values (may not be on HF in v1)
        "no_keyword"  — concept-only ablation with bare keyword token zeroed
    """
    assert kind in {"summary", "detail", "no_keyword"}, (
        f"kind must be summary|detail|no_keyword, got {kind!r}"
    )
    root = Path(data_root) if data_root is not None else DATA_ROOT
    path = root / f"token_independence_{kind}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected token-independence file not found: {path}")
    return pd.read_csv(path)
