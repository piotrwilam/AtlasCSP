# Multicircuit Analysis

## Definitions

- **Object**: an AST node (e.g., `For`, `ClassDef`) or a builtin (e.g., `len`, `ValueError`)
- **Universal object**: the universal representation of an object — the set of neurons that consistently activate for that object across all prompt variations and pairing contexts
- **Multicircuit**: a union of universal representations of several objects that share neural substrate
- **Strong multicircuit**: the constituent circuits clearly coincide — high pairwise similarity across multiple layers
- **Weak multicircuit**: the constituent circuits hardly coincide — low pairwise similarity, or similarity limited to one or two layers

Multicircuits may be weak on some layers and strong on others. A multicircuit that is strong at layers 5–7 but weak at layers 0–2 suggests that the shared processing emerges in the later stages of the network.

## Objective

Prove or disprove that the following four groups, identified from hierarchical clustering at Layer 5 (EPSILON=0.5, CONSISTENCY=0.80), are strong multicircuits:

1. **Generic AST** (19 circuits): AsyncWith, Attribute, ClassDef, Delete, Dict, GeneratorExp, Lambda, Return, SetComp, Slice, Subscript, YieldFrom, classmethod, isinstance, property, repr, staticmethod, super, zip

2. **Exceptions** (20 circuits): Continue, DictComp, ListComp, With, Yield + AttributeError, Exception, FileNotFoundError, IndexError, KeyError, MemoryError, NameError, OSError, OverflowError, RecursionError, RuntimeError, StopIteration, TypeError, ValueError, ZeroDivisionError

3. **Raise + Error hierarchy** (6 circuits): While, Raise, ArithmeticError, ImportError, LookupError, NotImplementedError

4. **Conditionals + Loops** (13 circuits): For, AsyncFor, Global, ImportFrom, Nonlocal, Starred, ExceptHandler, Try, If, IfExp, FunctionDef, AsyncFunctionDef, Break

## General Setup

**Model**: openai/circuit-sparsity — 8-layer sparse transformer (circuitgpt), 2048 MLP neurons per layer.

**Data**: 1370 (AST node, builtin) pairs, 100 prompt variations each. Raw activations extracted via Module 2A. Universal objects constructed via Module 2B with configurable thresholds.

**Current parameters**: EPSILON=0.5 (signal strength), CONSISTENCY=0.80 (firing share). These produce 115 universal objects (44 AST + 71 builtins) with mean circuit size of ~16 neurons per layer.

**Similarity measure**: Jaccard distance on binary masks of selective neurons (neurons that are neither dark nor always-on). Hierarchical clustering with average linkage.

**What "strong" means** (to be refined by the experiment):
- The group's mean intra-cluster Jaccard is significantly higher than random (permutation test, p < 0.05)
- This holds across multiple layers, not just one
- This holds across a range of EPSILON values, not just one threshold
- The group is more cohesive internally than it is similar to outsiders

## Planned Stages

### Stage 1 — Layer stability

For each of the four candidate groups, compute intra-group mean Jaccard at every layer (0–7). Plot the similarity profile across layers.

**Question**: At which layers does each group cohere? Is the grouping stable across the network, or confined to one layer?

**Expected outcome**: Exceptions should cohere broadly. Generic AST may fall apart at layers where the model differentiates fine-grained concepts. Conditionals+Loops should show coherence at layers involved in control flow processing.

**Pass criterion**: The group shows significantly elevated intra-group similarity at 3 or more layers.

### Stage 2 — Genuineness (vs. random baseline)

For each group that survives Stage 1, run a permutation test at every layer where the group coheres: randomly sample groups of the same size 1000 times, compare intra-group Jaccard.

**Question**: Is the group more self-similar than a random collection of circuits of the same size?

**Expected outcome**: Semantically coherent groups (Exceptions) should pass easily. Incoherent groups (Generic AST) may fail — their apparent cohesion might be indistinguishable from random at finer resolution.

**Pass criterion**: Permutation p-value < 0.05 at the layers identified in Stage 1.

### Stage 3 — Epsilon robustness

For groups that survive Stages 1 and 2, rebuild universal objects at multiple EPSILON values (0.01, 0.05, 0.1, 0.5, 1.0) and repeat the analysis.

**Question**: Does the group structure persist when the activation threshold changes? A strong multicircuit should exist across a range of thresholds, not just at one lucky value.

**Expected outcome**: Groups defined by genuinely shared neural substrate should remain cohesive as EPSILON varies. Groups that only appear at extreme thresholds are artifacts.

**Pass criterion**: The group passes the permutation test (Stage 2) at 3+ EPSILON values.

## Verdict Criteria

| Result | Interpretation |
|--------|---------------|
| Passes all 3 stages | **Strong multicircuit** — stable, genuine, robust |
| Passes stages 1+2, fails 3 | **Threshold-dependent multicircuit** — real but only at specific resolution |
| Passes stage 1, fails 2 | **Apparent multicircuit** — cohesive but not more than random, likely artifact |
| Fails stage 1 | **Not a multicircuit** — grouping is a single-layer artifact |
