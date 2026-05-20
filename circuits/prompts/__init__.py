"""Module 1 — Programmatic Prompt Generation & Comprehension Filtering.

Pipeline: Concept Matrix → Variance Engine → Perplexity Filter → Parquet Export.

Submodules:
    concept_matrix  — AST × builtin concept matrix construction
    variance_schema — lexical/structural/padding variance constants
    generators      — `ASTPromptGenerator`, builds prompts via `ast.unparse`
    filters         — `PerplexityFilter`, scores via `openai/circuit-sparsity` (needs torch)
    pipeline        — `run_pipeline`, A→B→C→D orchestrator

Module-level re-exports are omitted because `filters` and `pipeline` pull in
torch + transformers. Import submodules directly:

    from circuits.prompts.concept_matrix import generate_concept_matrix
    from circuits.prompts.generators import ASTPromptGenerator
    from circuits.prompts.filters import PerplexityFilter  # needs torch
"""
