"""Hugging Face Hub fallback for the CSP-Atlas dataset.

When `CSP_ATLAS_DATA_ROOT/<file>` is absent locally, fetch it from
`huggingface.co/datasets/CSP-Atlas` via `huggingface_hub.hf_hub_download`.
Cached under `~/.cache/huggingface/`; once downloaded, subsequent calls
are free.

Importing this module pulls in `huggingface_hub`, which is in the
`extraction` extras — install via `uv pip install -e ".[extraction]"`.
"""

from __future__ import annotations

import os
from pathlib import Path

from csp_atlas.paths import DATA_ROOT

HF_DATASET_REPO_ID = "CSP-Atlas"


def fetch_from_hf(filename: str) -> Path:
    """Download `filename` from the CSP-Atlas HF dataset if not local.

    Returns the local path to the file (either the existing one under
    DATA_ROOT, or the HF cache path).
    """
    local = DATA_ROOT / filename
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        filename=filename,
    ))


def ensure_data_root(*filenames: str) -> None:
    """Pre-fetch a list of files into DATA_ROOT (or HF cache).

    Convenience for setting up before a batch run. Each file is
    materialised at its DATA_ROOT location via symlink to the HF cache
    if it was downloaded from the Hub.
    """
    os.makedirs(DATA_ROOT, exist_ok=True)
    for name in filenames:
        path = fetch_from_hf(name)
        target = DATA_ROOT / name
        if target.exists() or target.is_symlink():
            continue
        # Materialise as a symlink to the HF cache path.
        target.symlink_to(path)
