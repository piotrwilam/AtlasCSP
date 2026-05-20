"""CSP-Atlas — analysis & plotting library for the v2 paper.

Three sub-packages:
    csp_atlas.io        — loaders for the frozen artifacts produced by `circuits/`
    csp_atlas.analysis  — pure-compute statistical primitives
    csp_atlas.plotting  — matplotlib rendering primitives

Zero PyTorch dependency. Reproducing the paper figures from frozen artifacts
needs only `uv pip install -e ".[dev]"`.
"""

from csp_atlas.paths import DATA_ROOT, PAPER_FIGURES_DIR, RESULTS_DIR, REPO_ROOT

__all__ = ["DATA_ROOT", "PAPER_FIGURES_DIR", "RESULTS_DIR", "REPO_ROOT"]
