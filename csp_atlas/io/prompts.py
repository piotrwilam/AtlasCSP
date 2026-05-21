"""Read the Parquet prompt artifacts.

Three files cover the §4 prompt set:
    11A_object_prompts.parquet      63,800 prompts (1,276 pairs × 50 variations)
    11B_checker_prompts.parquet     ~2,800 checker prompts (keyword without role)
    token_checker_prompts.parquet   ~2,800 token-only prompts

Schemas:
    object  : ast_node, builtin_obj, variation_id, prompt_text, sequence_loss,
              token_length, ast_verified
    checker : object, keyword, variation_id, prompt_text
    token   : object, keyword, variation_id, prompt_text
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from csp_atlas.paths import DATA_ROOT

PromptKind = Literal["object", "checker", "token_checker"]

_FILENAMES: dict[PromptKind, str] = {
    "object": "11A_object_prompts.parquet",
    "checker": "11B_checker_prompts.parquet",
    "token_checker": "token_checker_prompts.parquet",
}


def load_prompts(
    kind: PromptKind = "object",
    *,
    data_root: Path | None = None,
) -> pd.DataFrame:
    """Load one of the three prompt parquet files. Returns the raw DataFrame.

    Columns vary by kind — see module docstring.
    """
    assert kind in _FILENAMES, f"kind must be one of {list(_FILENAMES)}, got {kind!r}"
    root = Path(data_root) if data_root is not None else DATA_ROOT
    path = root / _FILENAMES[kind]
    if not path.exists():
        raise FileNotFoundError(f"Expected prompts file not found: {path}")
    return pd.read_parquet(path)
