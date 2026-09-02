# Tasks: `element-local-substrate`

Ordered. The critical path is 0 → 3 → 6 → 8; everything else parallelises.
Task 0 is already complete and is what justifies the rest.

## 0. Task-zero spike — DONE (PR #134)

- [x] 0.1 Install `[fem]`; run `tests/research/test_fem_baseline.py` for real (23/23 on
      scikit-fem 12.0.2, no API drift). Pin the verified range `>=9.0,<13`.
- [x] 0.2 Verify mesh immutability under `refined(elements)` — holds for the API; `mesh.p` is
      writable, so the substrate must clear write flags.
- [x] 0.3 Verify refinement is local and conforming — 32→49 local vs 32→128 uniform; zero
      hanging edges after four successive local refinements.
- [x] 0.4 Verify `skfem.Functional` gives a mesh-independent quadrature L2, and that it differs
      from the nodal RMS on a graded mesh (0.34→0.53 uniform vs 0.34→0.76 adaptive).
- [x] 0.5 Verify `basis.get_dofs()` shape on the installed version.
- [x] 0.6 Profile the estimator against the solve (6.2 ms vs 2.5 ms — the estimator dominates,
      not mesh cloning).
- [x] 0.7 Measure adaptive vs uniform on the element-local substrate (4–10× better at matched
      DOF; `N^-1.256` vs `N^-0.710`).
- [x] 0.8 Record the geometry trap that produced a wrong result mid-spike, and the tripwire
      that caught it.

## 1. Shared marking

- [x] 1.1 **[CORRECTED — lives at `src/research/marking.py`, not `src/refinement/marking.py`.]**
      CI's `reference-baselines-do-not-import-the-candidate` contract
      (`tests/regression/test_import_contracts.py`) forbids `src/research/baselines.py` and
      `fem_baseline.py` from importing anything under `src.refinement` — discovered only when
      the originally-planned `src/refinement/marking.py` failed that check in CI (both baselines
      delegate to `dorfler_mark`, so the import is real marking *behaviour*, not the kind of
      inert protocol/type import the contract's one existing exemption, `src/mcts/gumbel.py`,
      tolerates). `dorfler_mark(indicators, theta, variant)` with frozen `squared`/`linear`
      presets now lives under `src.research`, alongside its only two callers plus
      `src.research.substrates.tensor_grid`.
- [x] 1.2 **[Amended]** `DorflerAMRSolver._dorfler_mark` delegates directly. `._dorfler_mark_2d`
      does **not** delegate — it stays exactly as it was (it *is* the legacy behaviour the golden
      pins) — but `TensorGridSubstrate.mark()`/`.refine()` (Slice B) reproduce its
      selection-then-axis-projection as a provably equivalent two-step composition (flat
      selection via the shared `dorfler_mark`, then axis projection + `_refine_grid`), verified
      not by inspection but by `tests/research/test_tensor_grid_substrate.py`'s end-to-end
      trajectory golden test against a live `run_dorfler_arm` call.
- [x] 1.3 Delegate `ScikitFEMPoissonSolver._dorfler_mark`.
- [x] 1.4 Hypothesis byte-parity test for both variants, including the all-zeros case where
      the two implementations genuinely differ (AC4).

## 2. Protocol

- [x] 2.1 `src/refinement/substrate.py`: `RefinementSubstrate` + `SubstrateSolveResult` with
      `__post_init__` invariants (`n_dof_free <= n_dof`, both non-negative — AC5).
      numpy-only imports.
- [x] 2.2 Registry via `src/templates/registry.py::create_registry` — the canonical pattern;
      `src/refinement/substrate_registry.py`.
- [x] 2.3 Confirmed `python -m scripts.audit_abstractions src/refinement --fail-on-missing`
      stays clean — but only after fixing a real bug the audit tool had: it silently treated any
      generic `Protocol[T]` base (an `ast.Subscript`, not `ast.Name`/`ast.Attribute`) as a
      non-Protocol class, so `RefinementSubstrate`'s 8 members were never checked at all. The
      8 members have zero real callers today (this Protocol ships ahead of its first concrete
      consumer, Slice E's task 7.1) — exempted via a new, explicitly time-boxed
      `_STAGED_FOR_UPCOMING_TASK` allowlist in `scripts/audit_abstractions.py`, distinct from
      `_KNOWN_LIVE` (a real caller the heuristic can't see).
- [x] 2.4 **[Added — not in the original checklist]** `src/research/substrates/config.py::SubstrateConfig`:
      the Pydantic Data Contract, found missing by an independent adversarial review of the
      implementation plan.

## 3. `TensorGridSubstrate` — the back-compat proof

- [x] 3.1 Wrap `DorflerAMRSolver._solve_on_grid_2d`/`_compute_indicators_2d`/`_refine_grid`
      (all `@staticmethod`, called verbatim) plus `lshape_amr_compare._area_weighted_l2`.
      `_dorfler_mark_2d` itself is **not** called directly (see 1.2's amendment) — its
      selection+projection is reproduced as an equivalent `mark()`/`refine()` composition,
      proven by 3.2's golden test rather than by literal wrapping.
- [x] 3.2 **Golden**: bitwise against a live `run_dorfler_arm` (`tests/research/test_tensor_grid_substrate.py`),
      float-tolerance against `results/lshape_mcts_vs_dorfler.csv`'s `dorfler` rows (AC1). Both
      passed on the first real run — no deviation to record. Mutation-checked: forcing `mark()`
      to the wrong (`"linear"`) variant diverges the trajectory at level 2, confirming the golden
      test discriminates rather than passing vacuously.
- [ ] 3.3 Assert the adequacy gate **fails** here (AC7's mirror image) — deferred to Slice D
      (task 6.2 states the identical requirement from the gate's own side; doing it once there
      avoids a forward dependency on a gate that does not exist yet).

## 4. `fem_baseline` decomposition

- [ ] 4.1 **Capture goldens first**, before any refactor: `solve()` outputs for
      `uniform/P1`, `h_adaptive/P1`, `hp_adaptive/P2`.
- [ ] 4.2 Extract `build_initial_mesh`, `assemble_and_solve`, `zz_indicator`,
      `quadrature_l2_error` as module-level primitives.
- [ ] 4.3 Re-express `solve()` over them; the existing 23-test file must stay green untouched.

## 5. `SkfemTriSubstrate`

- [ ] 5.1 Implement over the task-4 primitives; report quadrature L2, carry nodal RMS in
      `extra` (AC6).
- [ ] 5.2 Clear mesh write flags behind `enforce_immutable_meshes`; property-test it (AC3).
- [ ] 5.3 Assert the reentrant corner is a mesh node at the origin (AC8).
- [ ] 5.4 `fem_required` marker + root-conftest hook + `ALPHAGALERKIN_REQUIRE_EXTRAS=1`;
      add `skfem_tri.py` to the coverage `omit`.

## 6. The adequacy gate

- [ ] 6.1 `tests/research/test_amr_arena_interpretability.py`: log-log rate separation over
      `RATE_FIT_DOF_RANGE`, uniform rate inside `UNIFORM_RATE_BAND` (too *good* is a defect),
      adaptive below `ADAPTIVE_RATE_MIN` (AC7).
- [ ] 6.2 Mutation-test it against `TensorGridSubstrate` — it must fail there.

## 7. First registrant

- [ ] 7.1 Minimal `RefinementGame` over the substrate, `@register_refinement_game`.
- [ ] 7.2 Registration only via an explicit `register_games` module — **never** from
      `__init__.py` (the documented SIGSEGV class).
- [ ] 7.3 Import-graph test pinning that.

## 8. Governance

- [ ] 8.1 Charter: scope register gains the new modules; the
      `RefinementGameRegistry`-has-no-registrants deviation is retired.
- [ ] 8.2 Charter: add the **time-boxed** two-path deviation, with its retirement condition
      (the golden test is the only remaining consumer of the legacy harness).
- [ ] 8.3 `specs/lshape_amr_compare.spec.md`: mark superseded, pointing here.
- [ ] 8.4 `CLAUDE.md` Regression Surface rows; `CHANGELOG.md`.
- [ ] 8.5 Run manifest for any artifact this change commits.

## Deferred — deliberately not in this change

- Estimator vectorisation (after 4.1's goldens, separate tolerance'd change).
- The arms, budget matching, statistics, pre-registration — the arena change.
- Any trained evaluator.
- Deleting `LShapeAMRGame` / `lshape_amr_compare.py`.
