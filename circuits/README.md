# `circuits/` — Layer 1: artifact-generation pipeline

Two pipeline stages, each as a subpackage. This code produced every frozen
artifact the rest of the codebase consumes. Re-running it requires `openai/circuit-sparsity`
loaded (HuggingFace Hub) and is **frozen** for v0.1.0 — under normal use you
read its outputs via the loaders in [`../csp_atlas/io/`](../csp_atlas/io/), not
re-extract.

Read this alongside paper sections **§3–§5**.

## `circuits/prompts/` — Module 1: prompt generation

| File | Purpose |
|---|---|
| [`concept_matrix.py`](prompts/concept_matrix.py)   | Build the 43 AST × 63 builtin pair queue (the concept space). |
| [`variance_schema.py`](prompts/variance_schema.py) | Lexical-domain, wrapper, and padding variance constants. |
| [`generators.py`](prompts/generators.py)           | `ASTPromptGenerator` — synthesises one prompt per (pair, variation) by `ast.unparse` of a programmatic AST. |
| [`filters.py`](prompts/filters.py)                 | `PerplexityFilter` — scores by `openai/circuit-sparsity` sequence loss, keeps top-N, drops catastrophic cells. |
| [`pipeline.py`](prompts/pipeline.py)               | `run_pipeline` — A → B → C → D orchestrator, writes validated Parquet. |

## `circuits/extraction/` — Module 2: extraction → universal

| File | Purpose |
|---|---|
| [`extraction.py`](extraction/extraction.py)         | `ActivationExtractor` — forward-hook the MLP output at every layer; last-token-position activations. |
| [`binarization.py`](extraction/binarization.py)     | `PairRepresentationBuilder`, `RawActivationCollector` — apply (ε, consistency) thresholds and build per-pair masks. |
| [`marginalization.py`](extraction/marginalization.py) | `UniversalModuleComputer` — intersect across the complementary set to produce the universal mask per concept. |
| [`metrics.py`](extraction/metrics.py)               | Array-based Jaccard / entanglement-index utilities used by the universal-stage marginalisation. (Distinct from set-based Jaccard in `csp_atlas/analysis/`.) |
| [`io_utils.py`](extraction/io_utils.py)             | HDF5 readers / writers for the intermediate artifacts. |
| [`pipeline.py`](extraction/pipeline.py)             | `Module2Pipeline` — end-to-end orchestrator. |

## Runtime requirements

- Python 3.12 (uv-managed; pinned via `.python-version`)
- A GPU with ≥ 8 GB VRAM for the forward passes on `openai/circuit-sparsity`
- Optional deps declared as the `extraction` extras group in `pyproject.toml`:

  ```bash
  uv pip install -e ".[extraction]"
  ```

## How notebooks call this

See [`../notebooks/1_artifact_generation/`](../notebooks/1_artifact_generation/) —
each notebook imports a handful of classes from here and runs one pipeline stage.

```python
from circuits.prompts.generators import ASTPromptGenerator
from circuits.prompts.pipeline import run_pipeline
from circuits.extraction.extraction import ActivationExtractor
from circuits.extraction.io_utils import save_activations_hdf5
```

## Why this is separate from `csp_atlas/`

`csp_atlas/` is the **analysis & plotting** library — it depends only on the
frozen artifacts on disk, never on PyTorch or HuggingFace. That separation
means: a reader who just wants to reproduce a paper figure installs
`uv pip install -e ".[dev]"` (no torch / transformers), points
`CSP_ATLAS_DATA_ROOT` at the data mirror, and runs `experiments/fig*.py`. The
expensive extraction machinery in this package stays optional.
