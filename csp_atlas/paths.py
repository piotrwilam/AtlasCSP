"""Repository-anchored paths.

Structural paths only (package root, results, paper figures). Per-experiment
paths come from the Hydra config, not from here.

`DATA_ROOT` is the source of truth for experimental artifacts. It reads
the `CSP_ATLAS_DATA_ROOT` env var if set, otherwise defaults to the user's
local mirror at `~/Data/CSP-Atlas`. On Colab, set the env var to the
mounted Drive path; on CI, to the downloaded HF dataset path.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root: this file lives at <repo>/csp_atlas/paths.py.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Default location of the artifact mirror. Override via $CSP_ATLAS_DATA_ROOT.
_DEFAULT_DATA_ROOT = Path.home() / "Data" / "CSP-Atlas"
DATA_ROOT: Path = Path(os.environ.get("CSP_ATLAS_DATA_ROOT", _DEFAULT_DATA_ROOT))

# Where experiment runs write their outputs (per-run subdir, gitignored).
RESULTS_DIR: Path = REPO_ROOT / "results"

# Paper directory (separate repo `Code/Papers/AtlasCSP` — see promote_figures.sh).
PAPER_DIR: Path = REPO_ROOT.parent / "Papers" / "AtlasCSP"
PAPER_FIGURES_DIR: Path = PAPER_DIR / "figures"
