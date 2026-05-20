# Layer 1 — Artifact generation

These notebooks produce the frozen data the rest of the codebase reads from.
They are **expensive to run** (the extraction notebooks need GPU access to
`openai/circuit-sparsity`) and their outputs are checked into the
`~/Data/CSP-Atlas/` mirror, not regenerated under normal use.

Read the corresponding paper sections **§3–§5** alongside.

## Pipeline order

1. **`1_prompt_gen.ipynb`** — Smallest end-to-end prompt-generation run; the
   v1 small-scale version of `11A_object_prompts`. One-off / smoke test.
2. **`11A_object_prompts.ipynb`** — Full object-prompt generation: 1,276 AST ×
   builtin pairs × 100 variations → 63,800-prompt Parquet. The §4.1 dataset.
3. **`11B_checker_prompts.ipynb`** — Matched checker prompts where the keyword
   token appears outside its structural role. The §4.2 contrast set.
4. **`5A_token_checker_generation.ipynb`** — Token-only / token-checker prompts
   for the §6 token-independence diagnostic.
5. **`2A_extraction.ipynb` / `12_extraction.ipynb`** — Forward-pass the model,
   record per-layer MLP last-token activations. Two variants: 2A reads from the
   small Parquet (modularity-score pipeline), 12 from the full 63,800-prompt
   set (threshold-sweep pipeline).
6. **`2B_universals.ipynb` / `13_universals.ipynb`** — Apply (ε, consistency)
   thresholds, marginalise across the complementary set, write the per-pair +
   universal HDF5 atlas.

## Code these notebooks import from

The `circuits/` top-level package — `circuits.prompts.*` (generators, filters,
pipeline) and `circuits.extraction.*` (ActivationExtractor, UniversalModuleComputer,
HDF5 I/O). See [`../../circuits/README.md`](../../circuits/README.md) for the
per-file map.

## What gets produced

Outputs land in the data mirror (`$CSP_ATLAS_DATA_ROOT`, default
`~/Data/CSP-Atlas/`) as:

- `11A_object_prompts.parquet` — 63,800 object prompts (`§4.1`)
- `11B_checker_prompts.parquet` — matched checker prompts (`§4.2`)
- `12_object_activations.h5`, `12_checker_activations.h5` — raw per-pair MLP activations (`§5.1`)
- `13_*_masks_eps{ε}_cons{c}.h5` — binarised + marginalised masks (`§5.2`–`§5.3`)
- `14_*_neuron_list_*.xlsx` — per-pair, per-layer concept-only / shared / token-only neuron lists (`§6`)

All paper-shipped results use **ε = 0.5, consistency = 0.8**. The 3 × 3 sweep
(`ε` × `cons`) is the §7.1 parameter-stability finding.
