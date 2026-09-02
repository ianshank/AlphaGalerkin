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
- [x] 3.3 Assert the adequacy gate **fails** here (AC7's mirror image) — done in Slice D as
      `TestAdequacyGateFailsOnTensorGridSubstrate` (task 6.2 states the identical requirement
      from the gate's own side; doing it once there avoided a forward dependency on a gate that
      did not exist yet). Measured at θ=0.5 over (200, 4000): adaptive `-0.2325` vs uniform
      `-0.6489`, ratio `13.35` at
      matched DOF. Deliberately carries **no** `fem_required` marker, so the discriminating
      half of the gate runs on every CPU CI job.

## 4. `fem_baseline` decomposition

- [x] 4.1 **Capture goldens first**, before any refactor: `solve()` outputs for
      `uniform/P1`, `h_adaptive/P1`, `hp_adaptive/P2`. Byte-identical before/after the
      extraction (`values`, `l2_error`, `metadata` all match).
- [x] 4.2 Extracted `build_initial_mesh`, `build_lshaped_initial_mesh` (the
      `ScikitFEMLShapedSolver`-specific L-shape builder, needed by `SkfemTriSubstrate`),
      `assemble_and_solve` (takes an injected `l2_error_fn` so it stays pure while
      `SolverResult.l2_error`'s nodal-RMS meaning is preserved byte-for-byte), `zz_indicator`,
      `element_gradients`, `triangle_area`, `estimate_smoothness`, and the new
      `quadrature_l2_error` as module-level primitives in `src/research/fem_baseline.py`.
- [x] 4.3 Re-expressed `solve()`/class methods as thin delegates; the existing 23-test file
      stays green untouched, plus 2 new tests for `quadrature_l2_error` (differs from nodal
      RMS; decreases under refinement).
- [x] 4.4 **[Added — the risk an independent adversarial review of the implementation plan
      surfaced]** Re-measured the adaptive-vs-uniform rate separation through the actual
      production primitives above (not the spike's own stronger ZZ reimplementation), on the
      real L-shaped Poisson benchmark via `SkfemTriSubstrate`: adaptive rate **-1.3109**
      (spike: -1.256), uniform rate **-0.6710** (spike: -0.710), at θ=0.5 over (200, 4000) —
      both comfortably inside the planned Slice D thresholds (`ADAPTIVE_RATE_MIN=-1.10`,
      `UNIFORM_RATE_BAND=(-0.85,-0.55)`). The production estimator's weaker (Python-loop,
      P1-downsampled) implementation does **not** measurably change the rate separation; no
      threshold recalibration needed.

      **Two figure corrections, both made in Slice D and both the same root cause** — a number
      transcribed from an exploratory script rather than read out of committed code, which is
      exactly what the gap-analysis review flagged. This line first read `-1.322`/`-0.671`,
      mixing two *fitting windows*; its replacement `-1.2515` was measured at **θ=0.3** while
      the committed gate passes `ComparisonParams.marking_fraction = 0.5`. Same substrate,
      same window, different θ, a 5% different rate. Convention going forward: **a convergence
      rate is quoted with its θ and its DOF window, or it is not quoted.**

## 5. `SkfemTriSubstrate`

- [x] 5.1 Implemented over the task-4 primitives (`src/research/substrates/skfem_tri.py`);
      reports quadrature L2 as the primary metric (`error_metric="quadrature"`, default), nodal
      RMS in `extra["l2_error_nodal_rms"]` (AC6). Verified end-to-end: 384 -> 391 elements from
      marking a single element (vs 384 -> 1536 for `mesh.refined()` with no args) — genuinely
      element-local, not the tensor-grid substrate's full-grid-line defect.
- [x] 5.2 Clears mesh write flags behind `enforce_immutable_meshes` (default `True`); verified
      both the enforced (`ValueError` on mutation attempt) and opt-out (`False`) paths.
- [x] 5.3 Verified the reentrant corner is a mesh node at the origin (AC8) on the real
      `build_lshaped_initial_mesh` output.
- [x] 5.4 `fem_required` marker registered (`pyproject.toml`); root `conftest.py` hook mirrors
      `gpu_required`'s shape but additionally reports a skip count via `pytest_terminal_summary`,
      and `ALPHAGALERKIN_REQUIRE_EXTRAS=1` raises `pytest.UsageError` (exit code 4) instead of
      skipping when scikit-fem is absent. `tests/research/test_fem_baseline.py` and
      `test_skfem_substrate.py` switched from `pytest.importorskip` to `pytestmark =
      pytest.mark.fem_required` so the hook can intercept before any test body runs (both
      modules already used lazy `_require_skfem()` calls, not module-level skfem imports, so
      this is a safe swap). Manually verified end-to-end both ways: uninstalling scikit-fem
      shows `fem_required: skipped 45 test(s) -- ...`; the same run under
      `ALPHAGALERKIN_REQUIRE_EXTRAS=1` exits 4 with the install-hint message. Wired into the
      `test-extras` CI job (`.github/workflows/ci.yml`). `skfem_tri.py` added to
      `pyproject.toml`'s coverage `omit`, alongside `fem_baseline.py`. **Not done**: no
      automated (`pytester`-based) regression test for the conftest hook itself — no precedent
      for this in the codebase, and out of proportion given the task's own scope; the hook's
      correctness is exercised every CI run via the two fem_required-marked suites' successful
      collection (a broken marker registration would fail collection under `--strict-markers`).

## 6. The adequacy gate

- [x] 6.1 `tests/research/test_amr_arena_interpretability.py`: log-log rate separation over
      `RATE_FIT_DOF_RANGE`, uniform rate inside `UNIFORM_RATE_BAND` (too *good* is a defect),
      adaptive below `ADAPTIVE_RATE_MIN` (AC7). Measured on `SkfemTriSubstrate`: adaptive
      **-1.3109**, uniform **-0.6710**, ratio **0.0946** at matched DOF (all at θ=0.5, the value
      `ComparisonParams.marking_fraction` gives and the gate passes; a rate quoted without its
      θ and window is not a fact — an earlier draft of this line said `-1.2515`, measured at
      θ=0.3) — all three of the spec's
      originally pinned thresholds hold against the *production* primitives, so task 4.4's
      recalibration contingency was not triggered.

      **One spec constant corrected**: `RATE_FIT_DOF_RANGE` `(200, 2600)` → `(200, 4000)`. The
      original window is physically incapable of holding `RATE_FIT_MIN_POINTS` (3) *uniform*
      points — a 2D uniform arm quadruples DOF per level, so a 13x window spans at most two.
      Not a judgement call: `fit_log_log_rate` raised `InsufficientSweepPointsError` rather than
      fitting a two-point slope, which is how it surfaced. Recorded with its reason in
      `specs/refinement_substrate.spec.md`. A *window-width* correction, not a threshold
      loosening — the other three constants were untouched.

      The measurement machinery itself (`run_refinement_sweep`, `fit_log_log_rate`,
      `measure_rate_separation`) lives in the reusable, substrate-agnostic
      `src/research/substrates/sweep.py`, **not** in the test file, so the eventual arena change
      consumes it instead of growing a second, subtly-different copy — the exact failure mode
      that made `dorfler_mark` necessary. It is also where `RATIO_FLOOR`, `AREA_FLOOR` and
      `RATE_FIT_MIN_POINTS` finally get real consumers (they had none, and
      `test_named_constants_match_spec` was asserting `RATIO_FLOOR == 1e-15` against a constant
      literally defined as `1e-15` — a tautology). 100% branch coverage; 22 unit tests drive it
      through a synthetic substrate, which is what makes "substrate-agnostic" a tested claim.
- [x] 6.2 Mutation-test it against `TensorGridSubstrate` — it must fail there.
      `TestAdequacyGateFailsOnTensorGridSubstrate` asserts the *identical* predicate
      (`gate_violations`, shared by both halves, so "the same assertion" is literally rather
      than approximately true) rejects the control substrate, **and** that the reason is the
      adaptive arm specifically — the tensor grid's uniform arm converges normally at `-0.6489`,
      inside the band, so a substrate too broken for either arm to converge would not pass this
      for the wrong reason.

      5/5 mutations killed (loosened `ADAPTIVE_RATE_MIN`; widened `UNIFORM_RATE_BAND`; loosened
      `ADAPTIVE_VS_UNIFORM_MAX_RATIO`; adaptive policy swapped for uniform; the DOF window
      reverted to the spec's `(200, 2600)`). The band mutation initially **survived** — both
      substrates' uniform rates sit comfortably inside it, so loosening it removed a constraint
      nothing exercised, leaving AC8's "too good is a defect" tripwire documented but unasserted.
      Closed by `TestGatePredicate`, which unit-tests `gate_violations` on synthetic
      `RateSeparation` values (including the `-1.05` both-arms rate that was the spike's actual
      wrong result), with no PDE solve.

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
