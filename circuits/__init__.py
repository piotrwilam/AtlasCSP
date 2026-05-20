"""Layer 1: artifact-generation pipeline (frozen, GPU-required to rerun).

Two pipeline stages live here as subpackages:

    circuits.prompts/      Prompt-generation pipeline (Module 1 in the v2 paper).
                           Builds the 1,276-pair (AST, builtin) concept matrix,
                           injects lexical/structural/padding variance, scores
                           prompts by `openai/circuit-sparsity` sequence loss,
                           and exports the validated 63,800-prompt Parquet.

    circuits.extraction/   Activation-extraction + binarisation + marginalisation
                           pipeline (Module 2 in the v2 paper). Hooks MLP outputs
                           at every layer, applies (ε, consistency) thresholds,
                           and intersects across complementary objects to produce
                           the universal circuit per concept.

Module-level re-exports are intentionally omitted because each submodule pulls
in heavy optional dependencies (torch, transformers, huggingface_hub). Always
import the specific submodule you need:

    from circuits.prompts.generators import ASTPromptGenerator
    from circuits.extraction.extraction import ActivationExtractor
    from circuits.extraction.marginalization import UniversalModuleComputer

See `circuits/README.md` for the per-file map.
"""
