# AtlasCSP

Code and analysis for the paper *CSP-Atlas: Concept-Specific Neural Circuits in a Sparse Python Transformer*.

A sparse 8-layer code transformer (`openai/circuit-sparsity`) develops dedicated neural circuitry for every Python construct tested, organised by a clean computational principle rather than by semantic category. This repository extracts neural circuits for 106 concepts (43 AST node types + 63 builtin objects) by marginalising across 63,800 controlled prompts and decomposes each circuit into concept-specific, shared, and token-driven components.

- **Data:** [huggingface.co/datasets/piotrwilam/AtlasCSP](https://huggingface.co/datasets/piotrwilam/AtlasCSP)
- **Model:** [openai/circuit-sparsity](https://huggingface.co/openai/circuit-sparsity)
- **License:** Apache-2.0

## Quickstart

```bash
git clone https://github.com/piotrwilam/AtlasCSP.git
cd AtlasCSP
uv sync                      # or: python -m venv .venv && pip install -e .
```

Verify the locked paper claims against the released artifacts:

```bash
export CSP_ATLAS_DATA_ROOT=/path/to/AtlasCSP-data
pytest tests/test_paper_numbers.py -v
```

Regenerate the atomicity dendrogram:

```bash
python experiments/fig1_atomicity_dendrogram.py
```

## Repository structure

The codebase is organised in three layers that mirror the paper's pipeline.

```
circuits/        # Layer 1 — artifact generation (GPU)
    prompts/         (11) prompt synthesis: 1,276 (AST × builtin) × 50 variations
    extraction/      (12) MLP-output activation extraction
    binarisation/    (13) ε-threshold + consistency filter; marginalisation
    decomposition/   (14) A \ B / A ∩ B / B \ A against checker masks

csp_atlas/       # Layer 2 — analysis library (CPU, fast, importable)
    io/              HDF5 / parquet / CSV loaders, HF Hub fallback
    analysis/        jaccard, decomposition, hierarchy (Ward), modularity
    plotting/        paper / poster / slides style; dendrogram renderer
    paths.py         DATA_ROOT resolution

experiments/     # Layer 3 — one script per paper figure
    fig1_atomicity_dendrogram.py

configs/         # Hydra configs; one per figure
tests/           # pytest, mirrored layout
    test_paper_numbers.py    # golden-numbers lock — every paper claim
    analysis/                # unit tests for analysis primitives
scripts/         # ops: promote figures to paper repo, regenerate scores
```

## Data

The frozen experimental artifacts — 63,800 object prompts, checker prompts, activation masks at all 9 (ε, C) settings, 106 universal circuit masks, per-object per-layer decomposition tables — are released as a HuggingFace dataset. The Python loaders auto-fetch missing files from the Hub:

```python
from csp_atlas.io import load_universal_masks, load_concept_inventory

ast, builtin = load_concept_inventory(eps=0.5, cons=0.8)        # 43, 63
masks = load_universal_masks(eps=0.5, cons=0.8)                 # 106 concepts × 8 layers
```

Direct download:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="piotrwilam/AtlasCSP",
    repo_type="dataset",
    filename="13_object_masks_eps0.5_cons0.8.h5",
)
```

See the [dataset README](https://huggingface.co/datasets/piotrwilam/AtlasCSP) for the full file schema (HDF5 groups, parquet columns, CSV layouts).

## Reproducing the paper

Every numerical claim in the paper is locked in `tests/test_paper_numbers.py`. With the dataset materialised at `$CSP_ATLAS_DATA_ROOT`:

```bash
pytest tests/test_paper_numbers.py -v
```

The locked claims (Appendix R of the paper):

| Section | Claim |
|---|---|
| §3.2 | 43 AST + 63 builtin = 106 testable concepts |
| §7.1 | All 106 universal circuits non-empty at all 9 (ε, C) settings |
| §7.2 | AST/builtin concept-fraction ratio 4–9× across all settings |
| §7.3 | Top of relaxed-modularity ranking dominated by atomicity members; `Break` at top with 3 significant layers at p=0 |
| §7.3 | Atomicity 6-set forms a single cluster at L3 under k=4 Ward cut on 1 − Jaccard |

## Citation

```bibtex
@article{Wilam2026CSPAtlas,
  title  = {CSP-Atlas: Concept-Specific Neural Circuits in a Sparse Python Transformer},
  author = {Wilam, Piotr},
  year   = {2026}
}
```
