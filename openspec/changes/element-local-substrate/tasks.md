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

- [ ] 1.1 `src/refinement/marking.py`: one `dorfler_mark(indicators, theta, variant)` with
      frozen `squared`/`linear` presets.
- [ ] 1.2 Delegate `DorflerAMRSolver._dorfler_mark` and `._dorfler_mark_2d`; the 2-D axis
      projection stays exactly where it is — it *is* the legacy behaviour the golden pins.
- [ ] 1.3 Delegate `ScikitFEMPoissonSolver._dorfler_mark`.
- [ ] 1.4 Hypothesis byte-parity test for both variants, including the all-zeros case where
      the two implementations genuinely differ (AC4).

## 2. Protocol

- [ ] 2.1 `src/refinement/substrate.py`: `RefinementSubstrate` + `SubstrateSolveResult` with
      `__post_init__` invariants. numpy-only imports.
- [ ] 2.2 Registry via `src/templates/registry.py::create_registry` — the canonical pattern; do
      not add a new one.
- [ ] 2.3 Confirm `python -m scripts.audit_abstractions src/refinement --fail-on-missing` stays
      clean: every protocol member needs a reader.

## 3. `TensorGridSubstrate` — the back-compat proof

- [ ] 3.1 Wrap `make_solve_fn` + `DorflerAMRSolver._refine_grid` verbatim.
- [ ] 3.2 **Golden**: bitwise against a live `run_dorfler_arm`, float-tolerance against the
      committed CSV (AC1). If bitwise cannot hold, record the deviation — do not loosen.
- [ ] 3.3 Assert the adequacy gate **fails** here (AC7's mirror image).

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
