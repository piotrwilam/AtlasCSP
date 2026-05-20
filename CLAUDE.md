# CLAUDE.md — CSP-Atlas project notes

> Read this first in every session.

## What this is

The codebase for the **CSP-Atlas** paper: *Concept-Specific Neural Circuits in a Sparse Python Transformer* (Wilam, 2026a). A single-model, single-language mechanistic-interpretability study of syntactic concept circuits in the `openai/circuit-sparsity` 8-layer transformer over Python.

- **Repo (this one, personal/working):** `piotrwilam/CSP-Atlas`
- **Public release repo (planned):** `piotrwilam/AtlasCSP` — to be created at publication
- **Data on Hugging Face Hub:** `CSP-Atlas` (kept at this name even after the public-release rename)
- **Model:** `openai/circuit-sparsity` (HuggingFace Hub) — 8-layer sparse transformer, code-only training, released 2025
- **License (eventual):** Apache-2.0

## Pre-refactor archive

The pre-refactor state is frozen at the git tag **`archive-pre-refactor`**. If anything goes wrong during the refactor, recover from there.

## Coding standards

Follow `coding_guidelines.md` at the repo root. **Read it at the start of every session.** Highlights:
- Three-layer structure (`circuits/`, `csp_atlas/`, `experiments/`) — substantive functions go into the `csp_atlas/` package once a stage is "done".
- Hydra configs, structured paths in `csp_atlas/paths.py`, no hardcoded numbers or paths.
- pytest mirrored layout (`csp_atlas/x/y.py` → `tests/x/test_y.py`).
- One script per figure in `experiments/`, plotting style centralised via `apply_style()`.
- Refactor-on-branch: any change > 50 lines / > 2 files lives on a feature branch and merges squashed.

## Project-specific quirks

- **Model name:** `openai/circuit-sparsity`. Loaded dynamically from the HF Hub via `hf_hub_download` of `gpt.py` and `hook_utils.py`. The package name in Python is `circuit_sparsity`.
- **Tokenizer:** The CSP tokenizer (`AutoTokenizer.from_pretrained("openai/circuit-sparsity")`) is **not** a generic GPT-2 tokenizer — token boundaries differ. Always load the bundled one.
- **8 layers, 2048-dim MLP output.** "Neuron" throughout refers to one dimension of the MLP output vector entering the residual stream, not an internal neuron in the expanded MLP hidden layer.
- **Default extraction parameters: ε = 0.5, consistency = 0.8.** All paper-shipped results use this setting unless stated otherwise. The 3×3 parameter sweep (ε ∈ {0.001, 0.1, 0.5} × cons ∈ {0.2, 0.5, 0.8}) is the §7.1 stability finding.
- **Concept-only / shared / token-only decomposition.** Universal mask `A` vs checker mask `B` → three partitions: `concept_only` (A\B), `shared` (A∩B), `token_only` (B\A). The **concept fraction** = `|A\B|/|A|`.
- **106 concepts:** 43 AST node types + 63 builtin objects. The full 43 × 63 = 1,276 (ast, builtin) pair matrix defines the prompt-generation surface.
- **63,800 prompts:** 1,276 pairs × 50 prompts per pair (after the perplexity filter). Source: §4.

## Data locations

The experimental artifacts (prompts, activations, masks, neuron lists) live in three mirrors:

| Tier | Location | Use |
|---|---|---|
| **Canonical for public release** | Hugging Face Hub — `CSP-Atlas` (kept at this name) | What the README points reviewers/readers at |
| **Working storage** | Google Drive `gdrive_innest:DATA/CSP-Atlas/` | Where Colab experiments write |
| **Local working mirror** | `/Users/piotrwilam/Data/CSP-Atlas/` | What `csp_atlas/paths.py` points `DATA_ROOT` at on this Mac |

`csp_atlas/paths.py` reads `CSP_ATLAS_DATA_ROOT` from env / config, defaulting to the local mirror. Same code → works on Mac, Colab, and CI.

## Refactor status (Phase 1)

Phase 1 (structural three-layer pass) is in progress on branch `refactor/three-layer`:
- ✅ `src/module1` → `circuits/prompts`, `src/module2` → `circuits/extraction`
- ✅ Notebook imports rewritten (`from module1.X` → `from circuits.prompts.X`)
- ✅ Notebooks reorganised into `1_artifact_generation/` + `2_analysis/`
- ✅ `csp_atlas/` package skeleton (`io/`, `analysis/`, `plotting/`, `paths.py`)
- ✅ `pyproject.toml`, `.python-version` (3.12), uv venv
- ⏳ **Phase 2** (Data not yet mirrored locally — needs `rclone sync gdrive_innest:DATA/CSP-Atlas/`): write `csp_atlas/io/` loaders, `csp_atlas/analysis/` primitives, `csp_atlas/plotting/`, then one `experiments/figN_*.py` per paper figure.
- ⏳ **Phase 3**: write `tests/test_paper_numbers.py` — lock every numeric claim in the paper (§7.1 parameter-stability, §7.2 concept-fraction values, §7.3 atomicity super-cluster, four-tier hierarchy).

## Known issues to fix during refactor

1. **§3.1 of `Papers/AtlasCSP/csp_atlas_v2.md`** has a draft paragraph on the CSP model provenance — verify the exact sparsity mechanism against the `openai/circuit-sparsity` HF model card and replace the soft "activation function in the MLP module" wording with the precise term.
2. **`csp_atlas_v2.md` references `[repository]`** in §Data and Code Availability — fill with `https://github.com/piotrwilam/AtlasCSP` once the public release exists.
3. **Figures and Appendices A–D** in `csp_atlas_v2.md` are placeholders ("To be generated; reuses the clustering function from the shared analysis code"). Populate once Phase 2 produces the figure scripts and Phase 3 locks the numbers.

## Reproducibility guarantees (target — Phase 3)

- **Frozen-numbers test** (`tests/test_paper_numbers.py`) will lock every numeric claim in the paper to the digits reported. Any future drift > 0.001 fails loudly.
- **Pin model revision SHA** (not just `openai/circuit-sparsity`) in the Hydra config for any re-extraction.
- **All permutation tests seed `random.seed(42)`** at the function level — not in the script — so p-values stay identical across reruns.

## Don't

- Don't regenerate frozen artifacts during the refactor unless explicitly asked.
- Don't commit large data files to git. Drive / HF Hub for data; git for code only.
- Don't add `utils/` or `helpers/` directories. Functions go in named pipeline-stage modules.
- Don't put logic in notebooks once a stage is done (see `coding_guidelines.md` notebook contract).

## See also

- `coding_guidelines.md` — full style and protocol doc.
- `circuits/README.md` — per-file map of the artifact-generation pipeline.
- `csp_atlas/` — analysis & plotting library (in progress).
- `../Papers/AtlasCSP/csp_atlas_v2.md` — paper draft.
