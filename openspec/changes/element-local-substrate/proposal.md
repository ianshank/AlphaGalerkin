# Proposal: `element-local-substrate`

## Why

The project's central claim — MCTS multi-step look-ahead beats classical adaptive refinement —
**cannot currently be measured**, and the charter says so in its own words.

`DorflerAMRSolver._dorfler_mark_2d` (`src/research/baselines.py:969`) projects element-wise
marks onto the x and y axes, and `_refine_grid` runs per axis, so marking one element inserts
full grid *lines* across the whole domain. The refinement budget is spent away from the
singularity.

That is no longer an argument; it is a committed artifact. `results/lshape_adaptive_vs_uniform.csv`
records adaptive Dörfler marking as **worse than plain uniform refinement** at matched DOF —
1.5× at 56 DOF rising to 10.5× at 2847, converging at `N^-0.14` against uniform's `N^-0.63`.
Comparing two *marking policies* on that substrate measures the substrate.

Two further facts make this the right next change rather than one of several:

1. **The fix is already 80% built and unused.** `src/research/fem_baseline.py` contains a
   genuine element-local FEM solver — conforming refinement, a Zienkiewicz-Zhu estimator,
   Dörfler marking — behind the existing `[fem]` extra. It self-registers under two
   `SOLVER_REGISTRY` keys that **nothing in the repository ever resolves**, and
   `src/research/__init__.py` never imports it, so those entries are unreachable in any real
   process. `specs/lshape_amr_compare.spec.md:150` names "skfem" as the prerequisite and never
   connects it to this class.
2. **A task-zero spike shows it inverts the result.**
   `evidence/spikes/2026-08-23-skfem-substrate.md`: on an element-local substrate, adaptive
   beats uniform by 4–10× at matched DOF, recovering the optimal P1 rate (`N^-1.256`) against
   uniform's singularity-limited `N^-0.710`. All four scikit-fem assumptions the design rests
   on were verified, and `tests/research/test_fem_baseline.py` — which had **never executed**
   in this environment, because a module-level `pytest.importorskip` skipped it silently —
   passes 23/23 with no API drift.

## What Changes

**A substrate abstraction, not a rewrite.** `RefinementSubstrate` is a stepwise interface
(`initial_mesh` / `solve` / `mark` / `refine` / `n_units`) so that any two arms of a refinement
comparison provably share one discretisation and differ only in *how they choose what to
refine*. Two implementations ship:

- `TensorGridSubstrate` wraps today's solver and refinement primitives **verbatim**. It is
  deliberately defective-by-design: it is the control, it reproduces the committed numbers
  bitwise, and it turns the tensor-product defect from prose in the charter into a *measured,
  guarded* fact.
- `SkfemTriSubstrate` decomposes `ScikitFEMPoissonSolver`'s monolithic adaptive loop into the
  four primitives it already contains. `solve()` keeps its exact behaviour, proven by its
  existing test file staying green rather than by inspection.

**An adequacy gate.** Adaptive must beat uniform at matched DOF on the substrate, asserted as a
log-log rate separation over a pinned DOF range — and the same assertion must **fail** on
`TensorGridSubstrate`. A gate that passes on both substrates is not a gate.

**One marking function.** The repository currently has two divergent Dörfler implementations
(squared versus linear bulk quantity; different zero-total behaviour). Both call sites delegate
to one function with two frozen variants, asserted byte-identical to their previous behaviour.

**`src/refinement/` gets its first runtime registrant**, retiring the charter deviation that
reads *"`RefinementGameRegistry` has zero runtime registrants … a forward-looking abstraction."*

## What This Change Does NOT Do

Stated explicitly, because the temptation to bundle is the failure mode this cycle exists to
correct:

- **No experiment.** Arms, budget matching, statistics and the pre-registration are a separate
  change. This one delivers the substrate and the gate that says whether a comparison on it
  would mean anything.
- **No trained evaluator.** Excluded so the eventual comparison isolates planning depth.
- **No deletion of the existing harness.** `LShapeAMRGame` and `lshape_amr_compare.py` are
  frozen as the back-compat golden reference. The resulting two-path period is recorded as a
  **time-boxed charter deviation with its retirement condition stated**, so it is disclosed
  rather than accidental.

## Risks

| Risk | Mitigation |
|---|---|
| The skfem substrate might *also* fail the adequacy gate on some problem. | That is the gate working. It fires in task zero, loudly and early, rather than after an experiment has been built on it. Already measured passing on the L-shape. |
| Mesh immutability is an API property, not an array one — `mesh.p.flags.writeable` is `True`. | Write flags cleared behind a config field, and pinned by a property test. |
| The estimator, not mesh cloning, is the dominant cost (~2.5× the solve, and the in-repo version is a Python double loop). | Measured in task zero. Vectorisation is sequenced **after** golden capture, because it perturbs floating-point results. |
| A geometry transform silently removes the singularity from the domain. | Cost one wrong result during the spike. Now an acceptance criterion, plus a uniform-rate band that treats *too good* convergence as a defect. |
| Bitwise back-compat may not hold across a refactor. | Asserted bitwise against the live legacy arm; if it genuinely cannot hold, the deviation is recorded rather than the assertion loosened. |

## Impact

- **Specs:** adds `specs/refinement_substrate.spec.md`.
- **Charter:** modifies *Scope Integrity* (new modules) and retires one deviation; adds a
  time-boxed one for the two-path period.
- **Backwards compatibility:** fully additive. No existing public symbol, config, YAML,
  committed artifact or test changes behaviour.
