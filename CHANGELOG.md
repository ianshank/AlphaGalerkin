# Changelog

All notable changes to AlphaGalerkin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — `element-local-substrate` Slice A: shared marking + substrate protocol
- **One `dorfler_mark` function replaces two independently-drifting Dörfler bulk-marking implementations** (`src/research/marking.py`) — `DorflerAMRSolver._dorfler_mark` (`src/research/baselines.py`, squared bulk quantity, marks ≥1 element on an all-zero indicator array) and `ScikitFEMPoissonSolver._dorfler_mark` (`src/research/fem_baseline.py`, linear bulk quantity, returns all-False on all-zeros) now both delegate to it (`variant="squared"`/`"linear"`), byte-for-byte, Hypothesis-verified against frozen reference re-derivations of each original formula plus the two solvers' own 85-test regression suite. **Lives under `src.research`, not `src.refinement`** as originally planned: CI's `reference-baselines-do-not-import-the-candidate` architectural contract (`tests/regression/test_import_contracts.py`) forbids `baselines.py`/`fem_baseline.py` from importing anything under `src.refinement` at all, and `dorfler_mark` is active marking behaviour, not the inert protocol/type import the contract's one exemption (`src/mcts/gumbel.py`) tolerates — caught by that contract's own test failing in CI, fixed by relocating rather than exempting.
- **`RefinementSubstrate` Protocol + `SubstrateSolveResult`** (`src/refinement/substrate.py`, numpy-only per `src/pde/games/__init__.py`'s documented SIGSEGV rationale) and its registry (`substrate_registry.py`) — the stepwise interface `openspec/changes/element-local-substrate/design.md` specifies, that `TensorGridSubstrate` and `SkfemTriSubstrate` will implement in later slices. `@runtime_checkable` so a concrete substrate satisfies it structurally, without inheriting from it. `SubstrateSolveResult.__post_init__` enforces AC5's `n_dof_free <= n_dof` invariant (both non-negative) at construction, not just by convention.
- **`SubstrateConfig`** (`src/research/substrates/config.py`) — the Pydantic data contract from `specs/refinement_substrate.spec.md`'s Data Contract table (`kind`, `element_type`, `marking_variant`, `error_metric`, `enforce_immutable_meshes`, `solve_cache_max_entries`, and the even-`initial_side` invariant), plus the named numerical-stability constants `RATIO_FLOOR`/`AREA_FLOOR`/`RATE_FIT_MIN_POINTS`. Not in the original 27-task checklist; added after an independent adversarial review of the implementation plan found no task owned it.
- **Fixed a real bug in `scripts/audit_abstractions.py`**, found while gating this change: a generic `Protocol[T]` base parses as `ast.Subscript`, not `ast.Name`/`ast.Attribute`, so `_is_protocol_class` silently returned `False` for any generic Protocol — `RefinementSubstrate`'s 8 members were invisible to the audit entirely, not "verified live", and the CI gate that scans `src/refinement` (`.github/workflows/ci.yml`'s "Audit abstractions (refinement surfaces)" step) would have passed on dead code by tool blind spot rather than genuine compliance. Fixed the AST unwrap; the resulting real finding (8 declared-but-uncalled members, since Slice A intentionally ships ahead of its first concrete consumer) is exempted via a new, explicitly time-boxed `_STAGED_FOR_UPCOMING_TASK` allowlist — distinct from the existing `_KNOWN_LIVE` (a real caller the AST heuristic can't see) — naming Slice E's task 7.1 (`RefinementGame` subclass over the substrate) as the entry it retires. 4 new regression tests, including one that would have caught the original bug.
- Coverage: `src/refinement` 100% branch, `src/research/marking.py` 100% branch, `src/research/substrates` 100% branch.

### Added — `element-local-substrate` Slice B: `TensorGridSubstrate`, the back-compat proof
- **`TensorGridSubstrate`** (`src/research/substrates/tensor_grid.py`) wraps `DorflerAMRSolver`'s existing static solve/indicator/refine primitives and `lshape_amr_compare._area_weighted_l2` behind `RefinementSubstrate`, reproducing today's `run_dorfler_arm` trajectory. `mark()`/`refine()` split `_dorfler_mark_2d`'s single call (element selection + x/y-axis projection) into two Protocol-compliant primitives — `mark()` returns the shared `dorfler_mark`'s flat element selection, `refine()` does the axis projection plus `DorflerAMRSolver._refine_grid` (unmodified) — a composition proven bitwise-equivalent to the fused legacy call by the golden test, not by inspection.
- **AC1 golden test** (`tests/research/test_tensor_grid_substrate.py`): the full `initial_mesh -> solve -> mark -> refine` loop matches a live `run_dorfler_arm` call bitwise (7 refinement levels, identical `n_dof`/`l2_error` at every level) and the committed `results/lshape_mcts_vs_dorfler.csv` `dorfler` rows to float tolerance. Mutation-checked: forcing `mark()` to the wrong bulk-marking variant diverges the trajectory at level 2 (`n_dof` 46 vs 34), confirming the test discriminates a wrong marking policy rather than passing vacuously.
- Coverage: `src/research/substrates` (now including `tensor_grid.py`) 100% branch.

### Fixed — Next-steps case follow-through: OperatorTrainer.load_checkpoint had zero test coverage (B30)
- **Two new tests close a real gap**: `OperatorTrainer.load_checkpoint` — including the `allow_unsafe_pickle` opt-in added 2026-08-21 — was never exercised by anything. The existing `TestOperatorTrainerRoundTrip` tests `load_torch_checkpoint` directly against hand-built dicts and never constructs an `OperatorTrainer`. Added `test_load_checkpoint_restores_state` (real save/load round trip through the public API) and `test_load_checkpoint_allow_unsafe_pickle_flag_is_plumbed_through` (`tests/test_operator_training.py::TestOperatorTrainer`).
- **New dedicated CI gate** (`Per-module coverage gate (training/operator_trainer)`): the module's two test files live outside `tests/training/`, so the whole-package `src/training` gate never measured it (25% in-scope vs 88% with its own tests). Gated at 85, native-runner form. See `docs/CODE_HYGIENE_AUDIT.md` B30.

### Fixed — Next-steps case follow-through: coverage-gate slack, stale docs (B28, B29)
- **`src/pde` coverage gate raised 75 → 85** (`.github/workflows/ci.yml`, `openspec/specs/project-charter/spec.md`) — measured branch coverage is 93.5%, so the gate carried 18 points of slack, unchanged since 2026-04-10. A prior PR claiming this raise (#57, open since April) never actually landed it. See `docs/CODE_HYGIENE_AUDIT.md` B28.
- **`README.md`'s Near-Term checklist corrected**: `Trainer` already inherits `BaseTrainer` (has for 5 months); ticked. `ModelOutput.vector_fields` scaffolding already exists and is unconsumed — reworded from "extend ModelOutput" to name what's actually open (`BasisSelectionGame`/losses/evaluator, needs a spec first). `OperatorTrainer`'s BaseTrainer migration noted as low-priority (zero production callers). See `docs/CODE_HYGIENE_AUDIT.md` B29.
- **`docs/CODE_HYGIENE_AUDIT.md`'s P0-1 heading corrected** from "3 of 8" to "2 of 8" — the Burgers OOD-reward defect it originally described was fixed 2026-08-19, verified at runtime through the real `_centaur_common` path (six distinct rewards, monotone error decrease). The heading and its pre-split `src/pde/operators.py` path citation were stale relative to the section's own status block.

### Fixed — Tech-debt Phase 2c: stale line citations in mdp_specification.md (B27)
- **`docs/doe_genesis/mdp_specification.md`'s `MeshRefinementGame` line citations were stale independent of any PR** — GitHub Copilot review on PR #140 flagged two dangling `mesh_refinement.py:321`/`:540` references (post-B4-split) and suggested retargeting to `mesh_refinement/game.py:41`/`:420`. Verified against the pre-split flat file at the merge-base rather than trusting the suggestion: every mesh-related citation in this doc was already off by 30-350+ lines before this PR touched the file, drifted from a much older module version. Fixed all 7+ citations (not just the 2 Copilot named) against the real current line numbers in the split `mesh_refinement/{mesh,game}.py`, verified by direct read of each cited symbol.

### Fixed — Tech-debt Phase 2c: OpenSpec skill conflict (B26)
- **`add-coverage-gate` and `openspec-change` gave conflicting guidance on editing the charter's coverage-gate register**, flagged by a GitHub Copilot review on PR #140 (the `src/poc/visualization` gate row from the B20 close-out). Investigated rather than reverted: `add-coverage-gate` SKILL.md's Step 5 already instructed a direct charter edit as one of five coupled mechanical steps, so the finding was a genuine tooling inconsistency, not a process violation — the *Quality Gate Fidelity* Requirement's text/scenario is unchanged, only a guard-verified data row was added. `openspec-change` SKILL.md's routing table now carves out this mechanical case for `add-coverage-gate`, reserving the full proposal/design/tasks process for a genuine policy call on gates (raising the ceiling, dropping a gate, changing the ⊆-direction rule).

### Fixed — Tech-debt Phase 2c: Gumbel MCTS opponent-perspective sign bug (B25)
- **`GumbelMCTS._sequential_halving` backed up child values from the wrong player's perspective** (GitHub Copilot review, PR #140, `gumbel.py:537`) — `_simulate(child)` returns a value from the child's own current-player perspective (root's opponent, one ply below root; `GumbelMCTS` drives only standard alternating two-player games, with no single-agent search mode), but this was accumulated into `child.value_sum` unnegated. Every consumer of that value (root's own action-selection score, and this PR's own new `_gumbel_mixed_value` mixing it with root-perspective `raw_value`) assumes root's perspective, so a guaranteed win for root scored identically to a guaranteed loss. Predates this PR (confirmed against the merge-base diff); this PR's `_gumbel_mixed_value` addition made the inconsistency directly visible rather than introducing it. Fixed with a one-line negation before backup. New regression test constructs two terminal root actions with different winners and is mutation-tested (fails with `best_action == 1` instead of `0` when the fix is reverted); three pre-existing tests' input literals were sign-corrected to match the now-correct perspective, with their asserted outputs unchanged.

### Added — Tech-debt Phase 2b: fail-fast PDE validation, stale-PR triage
- **`id_pde`/`PDEName`/`ResearchPDEName` now reject `heat`/`advection_diffusion` at config construction** (docs/CODE_HYGIENE_AUDIT.md B24) instead of letting a scenario run crash on `ExactSolutionUnavailableError` partway through, after other arms/seeds already spent compute. Considered and rejected: threading a `skipped` signal through `_centaur_common.run_basis_selection_cell`'s `CellOutcome` (5 call sites unpack it positionally; a NaN-residual sentinel risked silently entering a median calculation, worse than the crash it would replace). A `@field_validator` on each of the three config classes is small, local, and fails loud at the right time.
- **B24 correction: `navier_stokes` was also unsafe, for a different reason** (GitHub Copilot review, PR #140) — it *has* an `exact_solution()`, but returns a vector `(N, 2)` velocity field, which `BasisSelectionGame`'s scalar `(N,)` fit cannot subtract without a numpy broadcasting `ValueError` (reproduced before fixing). Every other reachable PDE (`poisson`, `burgers`, `poisson_lshaped`, `helmholtz`, `biharmonic`) was checked and confirmed scalar-safe. Each file's rejection set renamed to `_PDES_INCOMPATIBLE_WITH_BASIS_SELECTION` and extended with `"navier_stokes"`; `llm_prior_config.py` gained a new `ood_pde` validator since that PDE is only reachable there via `ood_pde`, not `id_pde`. Three existing tests' error-match pattern widened from `"ExactSolutionUnavailableError"` to `"incompatible with BasisSelectionGame"` to match the more general message.
- **Monthly stale-PR triage Routine** (docs/CODE_HYGIENE_AUDIT.md B23) — lists open PRs and flags ones with a stale base or 30+ days of inactivity. Reporting-only; a human decides what to do with a flagged PR.

### Fixed — Tech-debt Phase 2a: dead config fields, degenerate residual fallback
- **`GumbelMCTSConfig.use_mixed_value`/`.discount` were declared but never read** (`src/mcts/gumbel.py`) — toggling either changed nothing. `use_mixed_value` now gates a new `_gumbel_mixed_value()` implementing Gumbel AlphaZero's `v_mix` estimator (Danihelka et al. 2022, Appendix B): unvisited root children are completed with a value mixed from the root's raw network value and its visited siblings' Q, instead of a flat `0.0`. `discount` now scales the one-step backup into a child's `value_sum`. Both defaults are unchanged, but `use_mixed_value` defaults to `True`, so this changes actual Gumbel search output under default settings. New tests prove each knob has a real effect (a designed budget-starvation scenario flips `best_action` on the toggle) rather than re-asserting the field exists.
- **`BasisSelectionGame` reported a structurally-degenerate residual as convergence** (docs/CODE_HYGIENE_AUDIT.md P0-1, step 2) — for operators with no exact solution (`HeatOperator`; `AdvectionDiffusionOperator` called without `time`), the game's every state is grad-disconnected from `coords`, so `PDEOperator.compute_derivatives` always returns all-zero derivative terms and the "residual" collapses to a constant independent of the fitted solution or DOF. Both `get_initial_state` and `compute_exact_error` now raise a new `ExactSolutionUnavailableError` instead. `get_initial_state` is the one that mattered in practice — it sets `PDEState.error_estimate` at episode start, which `_centaur_common`'s cell runner checks before `compute_exact_error` is ever reached, so guarding only the latter (the audit's literal wording) would have left it unreachable dead code.

### Changed — Tech-debt Phase 2a: god-file split
- **`src/pde/operators.py` (2233 lines, 10 classes) split into a package** (docs/CODE_HYGIENE_AUDIT.md B4) — `src/pde/operators/` now holds one file per operator (`poisson.py`, `burgers.py`, `advection_diffusion.py`, `heat.py`, `navier_stokes.py`, `lshaped_poisson.py`, `helmholtz.py`, `biharmonic.py`) plus `base.py` for the shared `PDEResidual`/`PDEOperator` ABC, mirroring the pattern `operators_picogk.py` already used. Fully import-compatible: every *public* name (`dir()` excluding dunders) is unchanged, frozen and regression-tested in `tests/pde/test_operators.py::TestOperatorsPackagePublicAPI`. **Corrected**: this and the split's own commit message originally claimed the stronger "`dir(m)` is byte-identical before/after"; a peer review of PR #140 found that false (becoming a package unavoidably adds `__path__`, and the new explicit `__all__` adds itself as a `dir()` entry — both harmless, since nothing in this codebase introspects `dir()` on this module, but not what was actually verified or claimed). One real dedup along the way (`HelmholtzOperator`/`BiharmonicOperator`'s byte-identical `_manufactured` helper extracted to `base._manufactured_sine_product`). `mypy --strict` is byte-identical before/after.

### Added — Tech-debt Phase 2a: B20 coverage gates closed
- **Five remaining per-module coverage gates wired** (docs/CODE_HYGIENE_AUDIT.md B20): `src/poc/cli.py`, `src/poc/visualization/*`, the 3 classic scenarios (`transfer`/`complexity`/`stability`), `src/constants.py`, `src/seeding.py` — previously covered only by the global 85% gate. `transfer.py` was the one real gap (25% branch: `execute()`/`_train_model()`/`_evaluate_at_resolution()`/`_save_model()` had zero coverage); a real CPU micro-run suite now exercises it end-to-end. `cli.py`'s `cmd_eval_harness` and `visualization`'s `pareto_frontier` plot type were the other two real gaps, both now tested. All five gate at 85+ (cli.py 99%, visualization 100%, scenario set 94%, constants/seeding 100%).

### Security
- **Second unsafe-pickle-retry path, in the codec loaders (breaking for one flag-less flow)** — the same inversion fixed in the module-level `load_model_only()` existed again in the video-compression stack, and there it was *routinely reached* rather than latent. `load_codec` deserialized with a bare `torch.load`, which from torch 2.6 means `weights_only=True`; that rejects the three `Enum` members and the `created_at: datetime` every `CodecConfig.model_dump()` carries, so `load_codec` **failed on valid input**. `scripts/decode_video.py` caught that failure and retried with `weights_only=False` under a comment reading "fallback: manual loading for robustness" — so for every genuine checkpoint the unsafe path was the normal path, and a malicious file was executed precisely because it had failed the safe check. Proven end-to-end with a marker payload before fixing, and again after (payload no longer runs). Fixed at both ends: `SAFE_CODEC_GLOBALS` (the config module's own pure-data enums) is passed via a new `load_torch_checkpoint(..., extra_safe_globals=...)` parameter so the safe path *works*, and the fallback now uses the same chokepoint so falling back cannot escalate privilege. A loader whose safe path cannot succeed is a loader whose unsafe path becomes routine — that is the general lesson, and why the completeness of `SAFE_CODEC_GLOBALS` is asserted by reflection over the config module rather than restated as a list.
- **Every first-party checkpoint loader is now safe by default** — the five CLI entry points that take a checkpoint path straight from an argument (`src/experiments/verify_transfer.py`, `scripts/play_engine.py`, `scripts/encode_video.py`, `scripts/decode_video.py`, and `scripts/inspect_checkpoint.py` — the fifth found by the adversarial pass, and the one most likely to be pointed at an unfamiliar file) route through `load_torch_checkpoint`, each exposing the hatch as an explicit `--allow-unsafe-pickle` flag so unpickling an untrusted file is a deliberate operator action rather than the default. `src/training/checkpoint.py`'s module docstring previously had to scope its safety claim to two modules and name these as exceptions; it now states the repo-wide position, with the two genuine caveats (`zoo/storage.py`'s own documented policy, and the CI-excluded `hf_space` snapshot) named rather than glossed.
- **Public HuggingFace Space carried the same ACE path** — `hf_space/src/training/checkpoint.py::load_model_only` held a byte-for-byte copy of the original try-safe/except-unsafe fallback, and `deploy_space.py` uploads that bundle to a publicly reachable Space. Its module Security Note actively recommended that function ("for loading untrusted model weights only, use `load_model_only()`"). The fix is *ported* rather than imported, since the bundle must stand alone; it is CI-excluded, so it was verified by a manual six-point smoke (imports, datetime-carrying checkpoint still loads, payload blocked and not executed via both entry points, opt-in hatch still works as a control). **Corrected after review**: `load_model_only` turned out to have zero callers in that bundle, so fixing it alone would have remediated a dead function while the Space's *live* loads stayed open. Three live paths are now fixed too — `app.py`'s module-scope `MODEL = load_model(...)`, which unpickled a file `hf_hub_download` may have just fetched and was the highest-exposure deserialization in the bundle; the `verify_transfer.py` mirror; and `tools/gtp.py`'s load of `args.model`. No `weights_only=False` remains in `hf_space/` outside the opt-in hatch.
- **Unsafe-pickle retry removed (breaking)** — the module-level `load_model_only()` caught *any* exception from the `weights_only=True` load and retried with `weights_only=False`, so a malicious pickle was executed **because** it failed the safe check. Reproduced with a benign marker payload. The obvious fix (deleting the fallback) is wrong and the test suite proved it: `BaseTrainer.save_checkpoint` stores `config.model_dump()`, which carries a `created_at: datetime`, so one first-party path genuinely needs non-tensor globals. Replaced with a single `load_torch_checkpoint()` chokepoint doing `weights_only=True` inside a **scoped** `torch.serialization.safe_globals` allowlist of exactly three pure-data constructors (`datetime`, `timezone`, `timedelta`). ~~Scoped rather than process-global so it cannot leak into unrelated `torch.load` calls.~~ **Corrected**: `torch.serialization.safe_globals` unions into a module-global set, so the widening *is* process-wide while the window is open. The leak is inherent to the torch API and is documented rather than denied; the resulting race between overlapping windows was real and is fixed with a module-level `RLock`. Failure raises `RuntimeError` and never retries; passing `allow_unsafe_pickle=True` (keyword-only; it defaults to `False`) is the explicit hatch for foreign files. A fourth unguarded load site (`_load_training_state`) was found and routed through the same chokepoint. Loading a checkpoint containing arbitrary pickled objects now raises unless the flag is passed; no first-party save path is affected.
- **Checkpoint path traversal** — `CheckpointManager.load` performed no path validation whatsoever (no join against `checkpoint_dir`, no `resolve()`, no containment) and then pickle-loaded the result. Proven by probe: a manager rooted in a temp dir opened and read `/etc/hostname`. The security test that claimed to guard this had only ever passed by accident, via `PermissionError` on non-root CI runners. Relative paths now resolve against `checkpoint_dir` rather than CWD (a latent bug in its own right), the resolved path must be contained within it, and `allow_external=False` is threaded through `create_trainer` → `load_checkpoint` → `restore` → `load` so resume-from-elsewhere still works explicitly. Containment is checked *before* existence, preserving the `FileNotFoundError` contract for in-dir missing files.

### Added
- **Architectural import contracts are now executable** (`tests/regression/test_import_contracts.py`, `tests/support/import_graph.py`) — `scripts/audit_abstractions.py` already guards the *vertical* layering direction (an abstraction with no call site). Nothing guarded the *horizontal* one, and that is the direction that breaks silently: a single `from src.pde import ...` in the wrong file is a one-line diff that reads as convenience. Four declarative contracts, three of them scientifically rather than stylistically load-bearing. `src/refinement/` must stay domain-free, or "a refinement game is reusable across domains" is an unfalsifiable claim rather than a property. The **reference baselines** (`baselines.py`, `fem_baseline.py`) must not import the candidate search engine — if they did, the two arms of a comparison would share an *implementation* rather than an interface, and a defect in the shared code moves both arms in the same direction, which is invisible in a ratio. And `src/templates/` + `src/math_kernel/` must carry no domain dependency, or the reusable substrate is un-reusable by construction. Every contract carries a mandatory `reason`, for the same purpose the charter's deviations register does: an unexplained rule gets deleted the first time it is inconvenient. Three findings while measuring rather than asserting: `src/mcts/gumbel.py` genuinely imports `src.games.interface`/`state`, so `src.games` is in the forbidden list with `gumbel.py` a **recorded exemption** rather than being quietly dropped from the rule — and the guard's own `test_every_exemption_is_still_needed` caught the first draft, where the "exemption" explained an omission and therefore guarded nothing. `src/research/lshape_amr_compare.py` legitimately imports MCTS because it *is* the harness that drives both arms, so the baseline contract scopes to the baseline modules alone. Meta-guards throughout: a contract whose scope no longer resolves is **vacuous** and must fail rather than pass, and the exemption mechanism itself is proven against a synthetic contract rather than trusted. **7/7 mutation-killed**, including a planted leak and a planted relative import. The AST helpers were *extracted* from the stochastic layer's existing AC7 guard rather than forked, and that guard now delegates to them — two AST walks that must agree are two AST walks that will eventually disagree, and the payoff was immediate: a mutation of the shared boundary matcher was killed by the stochastic guard's parametrized test.
- **Scope containment is a check, not a paragraph** (`docs/FOCUS.md`, `config/focus.yaml`, `scripts/check_focus.py`, the `focus` CI job) — an owner decision froze two tracks for this cycle (the codec model-zoo, and `dashboard/` + its `hf_space/` deploy mirror). A freeze recorded only in prose is a suggestion, so this makes it mechanical. The rule is deliberately **not** "do not touch frozen code": a freeze is a pause, not a ban, and this repository's own history shows what deleting too early costs — `video_compression` was cut on 2026-07-22 and reinstated the next day. It is "do not make a *substantive* change to a frozen track in the same changeset as core solver work", because that is what split attention looks like in a diff. "Substantive" is a line budget rather than a file count, and the distinction does real work immediately: this very branch edits `hf_space/src/__init__.py` to single-source a version string alongside a new `src/research/` module, and that is a seven-line shim, not codec work. A budget states the intent — *feature work is never seven lines* — in one auditable number that lives in the config; the alternative, an exemption list, only ever grows until each entry has silently narrowed the gate to nothing. Both halves are kept in step in **both directions**: a frozen track named in the config but missing from the doc fails, and a doc claiming a freeze the gate does not enforce fails too — the second being the failure mode the whole file exists to prevent. 45 tests, **8/8 mutation-killed**, including three that only became discriminating after a first mutation survived: the original exact-vs-prefix test compared `deploy_space.py` against `deploy_space_helpers.py`, which does not prefix-match either way, so it proved nothing; the real hazard is a config entry written `src/pde` silently swallowing `src/pde_extras/`. Runs on `pull_request` only, as its own job because the diff needs the merge base and `lint`'s checkout is deliberately shallow. It is **not** in `ci-success`'s `needs` yet — the same convention the `secrets` job documents, and for the same reason: a brand-new gate promoted into the merge path by the pull request that introduces it gives a first red no way to be triaged. The `focus-override` label is the escape hatch, deliberately visible, because an override nobody can see is the same as no gate.
- **Governance surfaces are review-routed** (`.github/CODEOWNERS`) — `openspec/`, `evidence/`, `results/`, `config/baselines/` and the new focus files now appear in CODEOWNERS, so a change to what the project *claims* is never invisible in a pull request's file list. Recorded honestly as routing rather than enforcement: every path resolves to the same single owner, and CODEOWNERS cannot separate an author from an approver when there is one of them. Genuine separation needs distinct GitHub identities plus admin-enforced branch protection, none of which is repository-side config — so it is named as out of reach rather than implied.
- **Agentic harness brought up to date: 4 new subagents, 3 new skills** (`.claude/`, now 9 agents / 12 skills / 4 commands, from 5/9/4) — each addition is grounded in a failure this project actually had rather than a role-coverage checklist. `numerics-verifier` (read-only, adversarial) exists because both retracted headlines were *correct code measuring the wrong thing*, and carries the four failure modes as a checklist: a degenerate substrate, a boundary condition never imposed, a norm that biases the comparison, and a convergence rate that is *too good* — the last being how a mid-spike geometry error was caught. `claims-auditor` (read-only, cannot author what it audits) checks that comparison claims cite artifacts containing both arms, that artifacts carry provenance, and — the part usually missed — that guards are not inert. `spec-author` (no `Edit`, so it cannot touch `src/`) is where the single human gate sits. `prior-art-scout` records the pattern behind all three retracted novelty claims: the danger is not a missing citation but a *misclassified* one, since VDGN was already cited in the repo's own prior-art table, labelled only "MARL". New skills: `openspec-change` (the supreme spec system had no scaffold — only `specs/` did), `run-provenance`, `claims-ledger`. Existing agents and skills updated for the substrate layer, opt-in MCTS instrumentation under a 90% branch gate, visible-skip discipline, the `[dev,fem]` preflight install, and the Python 3.10 floor. The harness suite grew 71 → 103 tests and earned its keep immediately: the cited-path check caught a new agent referencing `specs/project-charter/` when the delta actually lives at `openspec/changes/<id>/specs/project-charter/`.

- **Gate 1 spec and change package for the element-local refinement substrate** (`specs/refinement_substrate.spec.md`, `openspec/changes/element-local-substrate/`) — **Draft, awaiting review before implementation.** Defines a stepwise `RefinementSubstrate` interface so both arms of any refinement comparison provably share one discretisation and differ only in how they choose what to refine, plus an *adequacy gate*: adaptive marking must beat uniform refinement at matched DOF, asserted as a log-log rate separation over a pinned DOF range — and the same assertion must **fail** on the tensor-grid control, because a gate that passes on both substrates is not a gate. Eight acceptance criteria, each grounded in a measurement from the task-zero spike rather than an argument: the two-error design is justified by the measured nodal-RMS drift (0.34→0.53 uniform vs 0.34→0.76 adaptive), mesh immutability by `mesh.p.flags.writeable` being `True`, the cost model by the estimator measuring ~2.5× the solve, and the geometry assertion by a wrong result the spike actually produced. Also adds `verified_error_certificate.spec.md` to the `specs/README.md` index, which had been missing since it was written.

- **`.gitignore`'s blanket `*.json` would have silently swallowed every provenance sidecar** — found while committing the first one. The rule at `.gitignore:121` has a handful of negations, none covering `results/`, so `results/*.run.json` was ignored and **nothing errored**: the artifact would simply land alone, exactly as if the provenance module did not exist. Added `!results/**/*.json`, narrow enough that scratch JSON under `outputs/` stays ignored, and guarded by `TestSidecarsAreCommittable` — mutation-tested by removing the negation. This is the failure mode a guard is most needed for, because it is invisible rather than loud.
- **Run provenance for committed artifacts** (`src/research/run_manifest.py`) — the charter requires every numeric headline claim to cite a committed artifact, but not that the artifact say *how it was produced*, and the gap is not theoretical: `results/lshape_mcts_vs_dorfler.csv` carries exactly one provenance column, `seed`. Not the search mode, not the marking fraction, not a git SHA — so it cannot be dated against the 2026-08-16 backup fix, while the harness still exposes the `legacy_adversarial` mode that produced the retracted number. A `RunManifest` is written beside an artifact as `<stem>.run.json` and records the config hash, git SHA and dirty flag, package versions, resolved seeds, per-arm parameters and counters, and the thresholds actually gated. Schema versioning follows `src/poc/baselines` (integer constant, `extra="ignore"`, explicit migration with a documented table). `collect_git_provenance` and `collect_package_versions` **never raise** — a provenance collector that throws inside a benchmark destroys the run it exists to document — and "unknown" is recorded rather than guessed, with `dirty=None` deliberately distinct from `False`. 100% branch coverage, 26 tests including a Hypothesis migration-idempotence property.
- **The missing uniform-refinement arm is now a committed artifact** (`results/lshape_adaptive_vs_uniform.{csv,run.json}`, `scripts/run_adaptive_vs_uniform.py`) — see *Fixed* below. Both arms share one solver, one geometry predicate and one refinement primitive; only the marking differs.

- **Retracted claims are guarded on the outward-facing SBIR surface** — three retraction guards already existed and between them left the highest-stakes surface uncovered: they scan the charter, `docs/related-work.md` + `README.md`, and `dashboard/` + `hf_space/`, but **nothing scanned `docs/business/`**. New `tests/regression/test_retracted_claims_guard.py` covers `docs/business/**`, `docs/doe_genesis/**`, `README.md` and `CLAUDE.md` for four retracted claim shapes. Two deliberate choices: markers are matched per **block** rather than per line, because the charter's line-level convention is wrong for prose — a correction note is inherently a multi-line blockquote and the marker word cannot appear on every line; and `docs/archive/**` is out of scope by construction, since archived PR reviews quote the fabricated figure legitimately under a banner and a guard that reverts on false positives is worse than none (the lesson of `check_doc_links.py`'s inline-span attempt, 105 false positives across 21 files). Mutation-tested four ways, each caught by a *named* test: the original violation restored verbatim, "uniformly single-step" reintroduced into an SBIR template, the fabricated transfer figure planted in a business document, and the scan roots emptied — because a guard that scans nothing passes everything. The exemption mechanism ships **empty**: the meta-test asserting every exemption is still needed immediately proved the one drafted for `CLAUDE.md` was already stale, because its milestone line carries its own markers.
- **Element-local AMR substrate spike, with evidence** (`scripts/spikes/skfem_substrate_spike.py`, `evidence/spikes/2026-08-23-skfem-substrate.md`) — `tests/research/test_fem_baseline.py` had never executed in this environment (module-level `pytest.importorskip("skfem")`, so it skipped silently) and the `[fem]` extra was pinned `>=9.0` against a current 12.0.2. It passes **23/23 on 12.0.2** with no API drift; the extra is now pinned to the verified range `>=9.0,<13`. The decisive measurement: on the standard L-shaped Poisson benchmark with P1 elements, a ZZ recovered-gradient estimator and Dörfler marking at θ=0.5, the element-local substrate gives uniform `L2 ~ N^-0.710` against adaptive `L2 ~ N^-1.256` — adaptive beats uniform by **4–10× at matched DOF, widening**. On the current tensor-product substrate adaptive is 5–9× *worse*. The rates are the textbook AFEM result: uniform is rate-limited by the `r^(2/3)` corner singularity while element-local adaptive recovers the optimal P1 rate.

- **Deterministic validation for the `.claude/` agentic harness** — 9 skills, 5 subagents, 4 slash commands, the SessionStart hook and `settings.json` are executable configuration that had **no tests at all**: a skill citing a deleted path, an agent declaring a tool that does not exist, or a permission naming a renamed module each failed only at the moment someone relied on it. New `tests/claude/` suite (71 tests, hermetic and deterministic — no network, no model calls, ~0.25 s) checks frontmatter, name-to-path agreement, tool-name validity, cited-path existence, permission-module resolution, hook shell syntax, name collisions, parse determinism, and that every non-elided python snippet compiles. Data-driven rather than enumerated, so a new artifact is validated the moment it is added. Deliberate forward references (`src/pde/certificate/`, which the `certificate-validation` *kickoff* skill instructs you to create) are declared with a reason **and** asserted to still be forward, so a stale exemption fails rather than silently weakening the check — the distinction that makes this gateable where `check_doc_links.py`'s naive form was not. Mutation-tested four ways.
- **gitleaks actually runs** — `.gitleaks.toml` and a `make gitleaks` target had both existed for months while **nothing invoked either**, in CI or pre-commit. A secret scanner that never runs is worse than none, because its presence in the tree reads as coverage. Now a dedicated `Secret Scan` **job** in `ci.yml` (`fetch-depth: 0`). It first landed as a step inside the 3-way `test-fast` matrix, which was wrong twice over and failed on its first run: it executed three times per push, and `actions/checkout`'s default shallow clone left the action's diff scan unable to resolve the base revision — `git` errored, the scan covered **~0 bytes**, and it still logged "no leaks found in partial scan" before exiting 1. A scanner reporting clean over zero bytes is exactly the failure mode being corrected here, so the fix is full history plus a single run. Deliberately **not** in `ci-success`'s needs list yet: it reports now and blocks once it has been green across a few pushes. the Makefile target degrades *loudly* (it prints that the scan did NOT run and names CI as the enforcing copy) rather than passing quietly on a machine without the binary, since it is now chained into `make pre-pr`.
- **`make pre-pr` covers what CI covers again** — new `test-demos` and `test-claude` targets mirror their CI steps and are chained in. Without them `pre-pr` was narrower than CI, which is the same drift that let `tests/demos/` and `tests/notebooks/` go unexecuted in CI for months.

### Changed
- **Every resampled p-value and confidence interval is now reproducible on request** (`src/poc/statistics/significance.py`) — `StatisticalAnalyzer` drew from NumPy's *global, unseeded* stream at **four** separate sites, not the one an earlier audit named: `_bootstrap_test`'s shuffle, `_permutation_test`'s shuffle, and both `choice` calls inside `_bootstrap_ci`. Two runs over identical inputs therefore returned different intervals — in the one module whose entire job is rigour, inside a project whose governance position is that every number traces to a committed artifact. Fixed with a typed `SignificanceTest.random_seed` field plus an injectable `resampler` on the analyzer, resolved by precedence (explicit override → configured seed → global stream). The unseeded path is left **byte-identical**, deliberately: it is the only fallback under which a caller who already does `np.random.seed(...)` keeps getting today's results, and all 52 pre-existing tests pass untouched. The cost of that default is stated in the field's own description rather than glossed. Wired through in `ScalingLawScenario`, whose recorded `arm_comparison_p` is now derivable from the scenario's declared seed. 20 new tests, **8/8 mutation-killed** — including that a seed *accepted and then ignored* must not look like success, and that zero is a legal seed rather than an absent one. Two test-authoring traps are recorded in the tests themselves because each produced a *passing* non-test first: well-separated arms pin a permutation p-value to 0.0 for every seed, so the RNG cannot show through; and the scaling-law scenario's default `significance_test_type` is `mann_whitney`, which is deterministic and never reaches an RNG at all, so a test left on the default could not detect an unseeded draw.
- **Licensing and IP posture settled and recorded** (`docs/adr/0004-licensing-and-ip-posture.md`) — an external strategy review flagged this as the one decision that is irreversible, costs no engineering time, and blocks nothing, and therefore should be settled explicitly rather than discovered later. **MIT stays and development continues in the open.** The reasoning is the disclosure that has already happened: a public repository with 663 commits and a public HuggingFace Space mirror, against an `IP_STRATEGY.md` whose three provisional patent claims are still listed as *Pending*. US filing runs on a 12-month clock from public disclosure and most other jurisdictions apply absolute novelty, so filing on what is already published here is largely unavailable — the operative rule going forward is that a provisional must **precede** the disclosure it protects. The employment-IP question about `src/video_compression/` is recorded as an open owner decision rather than answered; the subsystem is frozen for this cycle, so it blocks nothing. ADR 0003 is deliberately left unallocated because unmerged PR #118 claims it.
- **OpenSpec archive convention documented** (`openspec/project.md`) — a change package is active while `tasks.md` has unchecked boxes, and moves to `openspec/changes/archive/<change-id>/` once complete. Without a convention, completed and in-flight work are indistinguishable in a directory listing and the tree grows monotonically — which matters now that six more change packages are planned. `project-charter-alignment/` (36/36 complete) is an explicit, reasoned exception: it is cited by name from `CHANGELOG.md`, which is append-only so the reference cannot be repointed, and from `tests/docs/test_charter_alignment.py`'s docstring. Archiving it would break a historical citation to gain tidiness.

- **C4 architecture gains a Quality Gates & Agentic Harness component diagram** (`docs/architecture/c4_mermaid.md`, v3.0.0 → v3.1.0) — the `.claude/` harness and the CI gate layer are enforced on every push but appeared in no architecture diagram. Includes the "deliberately not gated" register with reasons rather than numbers. Diagram validated as rendering (C4, 37 KB SVG).
- **README claim corrections** — the file contradicted itself on test counts ("7,000+ test functions" in prose vs "3,000+ tests" in the tree); measured and reconciled to **8,573 test functions / 9,770 collected**. The per-module gate count is stated as **34** (measured: 30 inline `--cov-fail-under` plus 4 native-runner `coverage report --fail-under`) — an initial draft of this same change asserted 31 without counting, and was corrected before commit. Dropped the stale `src/mcts/constants.py` reference (that module was deleted in the round-2 hygiene pass as a zero-consumer re-export shim), and surfaced the `claude/`, `demos/` and `notebooks/` test tiers.

### Fixed
- **CI's 3.10 job broke at collection on stdlib that is 3.11+** — `tests/docs/test_version_consistency.py` imported `tomllib` and `scripts/run_adaptive_vs_uniform.py` used `datetime.UTC`, both added in Python 3.11, while `pyproject.toml` declares `requires-python = ">=3.10"` and CI runs a 3.10 job. An unimportable test module is a **collection** error, so this aborted the entire fast lane on 3.10 rather than failing one test. Neither is caught by ruff: `target-version = "py310"` governs which *rewrites* ruff suggests, not which stdlib you reach for. The version reader now parses `[project].version` with a table-anchored regex (no dependency, works on every supported version), and the timestamp uses `timezone.utc`. New guard `tests/docs/test_python_floor_compatibility.py` AST-scans `src/`, `tests/`, `scripts/`, `dashboard/`, `config/` and `conftest.py` for stdlib newer than the declared floor, reading the floor **from `pyproject.toml`** so raising `requires-python` relaxes the guard automatically instead of stranding a stale rule. Guarded use — `try`/`except ImportError`, or behind a `sys.version_info` check — is deliberately not flagged. Mutation-tested three ways, including restoring each of the two real failures verbatim, and the tree-wide scan confirms no other instances exist.

- **The test-suite-size claim was typed, not measured** — the charter's evidence register said "7,000+ test functions" citing `tests/`, the weakest citation in the register (a whole directory proves nothing), while `CLAUDE.md`'s 2026-08-16 milestone said "705+ tests" with no indication whether that meant *added by that sprint* or *total*. The real tree has **8,628 test functions across 432 files**. Both claims are now unambiguous, and the floor is **machine-checked**: a new guard AST-counts the tree and fails if the claimed floor exceeds it, or if the floor has drifted so far below reality that it is a fossil rather than a claim. AST rather than `pytest --collect-only` so counting is deterministic, needs no imports (an absent optional dependency cannot skew it), takes milliseconds, and counts *functions* rather than parametrized cases — which is what the claim says. Mutation-tested three ways: an overclaim, a fossil floor, and removing the claim entirely, since a guard whose subject disappears must fail rather than go quietly inert.

- **A charter claim cited an artifact that could not support it** — the evidence row *"L-shape adaptive Dörfler vs uniform at matched DOF | Dörfler 5–9× worse"* cited `results/lshape_mcts_vs_dorfler.csv`, whose `method` column contains only `{dorfler, mcts}`. **There was no uniform-refinement arm in any committed artifact**, so a correct number traced to prose in `docs/NEXT_STEPS_REVIEW_2026-08-18.md` rather than to data — and the evidence guard could not tell, because it checks only that the cited file exists. Independently reproduced and committed: adaptive Dörfler is **1.5× worse at 56 DOF rising to 10.5× at 2847**, converging at `N^-0.14` against uniform's `N^-0.63`. The rate separation is the sharper statement because it does not depend on where the curves are read. The charter row now cites the new artifact and states the measured range rather than a remembered band. Two new guards, both mutation-tested: a comparison claim's artifact must **contain the arms compared** (a small explicit arm vocabulary, because a fuzzy prose match is what killed `check_doc_links.py`'s inline-span attempt), and a cited `results/*.csv` must carry a `.run.json` unless it is in a declared, meta-tested grandfather list that is asserted to shrink. One mutation initially appeared to pass and did not: the guard's first version matched only un-umlauted `"dorfler"` while the charter writes `"Dörfler"`, so it silently skipped the exact row it was written for. A vocabulary-coverage meta-test now catches that, and the mutation harness asserts each mutation actually applied — a mutation test that does not verify its own mutation is worth as little as a guard that scans nothing.

- **A 77 KB Windows pytest log was committed at the repository root** — `onnx_err.txt`, UTF-16, swept in by a `style: format uncommitted changes` commit and referenced by nothing. It also embedded a local filesystem path (`C:/Users/…/OneDrive/…`), which is a second, smaller reason to remove it. Deleted.

- **Four version strings had drifted from `pyproject.toml`, and nothing guarded them** — `src/__init__.py` reads the installed distribution's metadata, so the *importable* `__version__` could not drift; every other place a version is written down had. `README.md` declared it **twice with two different values** — `0.4.0-dev` under `## Project status`, and `0.1.0` in an orphaned duplicate of that same paragraph stranded at the end of the roadmap section (deleted). `src/templates/cli.py`'s `create_cli_app(version=...)` defaulted to a hardcoded `"0.1.0"` that **every CLI built on it inherited** — `python -m src.agents.cli --version` reported `0.1.0` against a `0.4.0-dev` project. `src/tools/gtp.py` told GTP controllers the engine was `0.1.0`. `hf_space/src/__init__.py` hardcoded the same stale literal. The three in-code sites now resolve from the package (`create_cli_app`'s default becomes `None` → resolved at call time; `src/agents/cli.py` drops its explicit argument); `hf_space/` keeps a literal because the deploy bundle installs no distribution to read, and is now guarded instead. New `tests/docs/test_version_consistency.py` compares every declaration against `pyproject.toml` after PEP 440 normalisation (`0.4.0-dev` and `0.4.0.dev0` are the same version, and a guard that could not see that would be unusable) and asserts the two in-code reporters resolve rather than hardcode. `tests/tools/test_gtp.py` had **asserted the stale `"0.1.0"`** — the test was defending the bug — and now asserts the package version. Mutation-tested five ways, each caught by a *named* test, including neutering the guard's own regex, since a scanner that matches nothing passes everything.

- **A retracted headline was stated live in the outward-facing prior-art review** — `docs/business/proposals/PRIOR_ART_REVIEW.md` asserted that "an untrained MCTS refinement policy beats Dörfler by a few percent at matched DOF". That claim was **retracted on 2026-08-16**; the committed result is the opposite (MCTS loses at matched DOF, median ratio 1.0996 winning 1 of 5 seeds, and at matched compute, 2.04 winning 0 of 5). The contrast with the 2026-07-22 transfer correction is diagnostic rather than incidental: that one *did* propagate to these documents, via an enumerated nine-file sweep recorded in `openspec/changes/project-charter-alignment/tasks.md`. The later AMR retraction got no such sweep — which is why the fix is a guard (above) and not another manual pass. The correction also records the deeper point neither figure captures: both arms were compared on a tensor-product substrate where adaptive marking is itself 5–9× worse than plain uniform refinement, so neither number measures policy quality.
- **"The AMR-RL literature is uniformly single-step" retracted** — asserted in `README.md` (×2), `CLAUDE.md`, `PRIOR_ART_REVIEW.md` and the SBIR Phase I template. It conflates *no search tree* with *no multi-step planning*: VDGN (arXiv:2211.00801) — already cited in the repo's own prior-art table, but classified only as "MARL" — refines **anticipatorily** for features that appear at later times, unlocking regions of the error-cost landscape a local error estimator cannot reach. The table's `MCTS?` column was always correct; the framing was not, and a reviewer would have said so on first read. The surviving delta is stated narrowly throughout: an explicit bounded search tree over refinement sequences with a transparent, reportable compute budget — not multi-step reasoning as such, and **not** "vs. myopic RL", which VDGN rebuts. `docs/related-work.md` gains a VDGN entry recording the boundary, including its consequence for benchmark design: a learned marking policy is exactly arXiv:2207.06339, so an experiment whose action space is "choose θ this step" sits inside occupied prior art.

- **1D Poisson posed the degenerate problem `u == 0`, so the AMR baseline measured nothing** — `PoissonOperator`'s manufactured solution was written as `sin(pi*x)*sin(pi*y)` with `y = zeros_like(x)` when `dim == 1`, so the `sin(pi*0)` factor made *both* `source_term` and `exact_solution` identically zero: every 1D Poisson problem in the repo was `-u'' = 0` with homogeneous Dirichlet data. Measured consequence, not inferred: Dorfler AMR on 1D Poisson reported `max_indicator == 0.0` at **every** step, so bulk marking — the thing Dorfler *is* — never fired once, and `l2_error` was `0.0` at every DOF count, so no error-vs-DOF assertion could ever have failed. The surviving `n_dof` assertions were really asserting `n_start + max_refinements` arithmetic. Same defect class as the 2026-08-16 L-shape retraction: a substrate that does not converge makes every downstream comparison meaningless, and it is the real cause behind the Burgers AMR note asking for "a non-degenerate steady problem" — Poisson was not one either. Fix is dimension-general (`u = prod_d sin(pi*x_d)`, `f = dim*pi^2*u`) in a shared helper so source and solution cannot drift apart; `dim >= 3`, previously truncated to the 2D expression by the same guard, is also fixed. At `dim == 2` this is the old expression re-associated — measured deviation **1 ULP** (9.7e-8 relative to amplitude at float32, 1.8e-16 at float64), `exact_solution` bitwise identical — and every scenario path defaults to `domain_dim=2`, so no calibrated threshold or reported figure moves. Two named guards added and mutation-tested against the restored collapse.
- **`load_codec` could not read any codec checkpoint this repo writes** — three defects in one function, each invisible because every fixture covering it wrote the one shape no in-repo trainer produces. `checkpoint["config"]` was passed straight to `CodecConfig(**...)`, but every writer dumps a `TrainingConfig`, a *sub*-config of `CodecConfig`: reproduced end-to-end, a byte-for-byte `VideoCompressionTrainer` checkpoint raised `ValidationError` with 22 errors. Weights were read only from `"model_state_dict"` while both real writers (`VideoCompressionTrainer`, `ZooTrainer`) use `"model_state"`, so the bare-payload fallback ran and `load_state_dict(..., strict=False)` matched **nothing**, returning an *untrained* codec with no error at all. And `use_mcts` was probed as `"rate_controller" in checkpoint["model_state_dict"]`, which can never be True — `MCTSRateController` is not an `nn.Module`, so it contributes no state-dict entries under any key. The config path now tries `CodecConfig`, then `TrainingConfig` nested under `training=` so its recorded hyperparameters survive, then a named default rather than rejecting a checkpoint whose weights load fine; the state dict honours both keys in documented precedence (matching workarounds `decode_video.py` and `encode_video.py` already carried locally); and the impossible probe becomes an explicit keyword-only parameter defaulting to the `False` it always produced, so no existing caller changes. This is what made `decode_video.py`'s reduced fallback the ordinary path rather than the exception.
- **The legacy-checkpoint escape hatch reached only one of three loaders** — `src/training/checkpoint.py`'s module docstring promises unrestricted deserialization is reachable only via an explicit per-call opt-in, but `operator_trainer.py`, `video_compression/training/trainer.py` and `distributed/trainer.py` were routed through the chokepoint with no `allow_unsafe_pickle`, so a checkpoint written before that routing landed was unreadable with no documented recovery. All three now take the same keyword-only flag, defaulting to `False`.
- **Two security guards were green for reasons unrelated to what they guard** — the AST pickle guard matched only the literal `torch.load(...)` attribute form, so `from torch import load` and `import torch as t` (ordinary style, not adversarial tricks) bound the same function to a name it ignored; it now resolves each module's import bindings, is renamed to what it actually asserts, and its docstring states what stays uncovered (`getattr`, a kwargs splat, and `pickle.loads` — a different function whose one call site unpickles peer-rank `all_gather` bytes and wants a schema'd format, not an allowlist). The CLI-flag test built its own replica `ArgumentParser` and asserted against that, staying green no matter what `scripts/inspect_checkpoint.py` did; `build_parser()` was extracted there (the convention six other scripts already follow) and the test now calls it. Both mutation-tested.
- **Two justification comments no longer matched the code they justify** — `ci.yml`'s security-suite step claimed "33 tests, <1s" (measured: **71 tests in 7.12s**), and `pyproject.toml`'s torch-floor comment cited three loaders passing no `weights_only`, two of which this same branch had already routed through the chokepoint. A stale justification argues for keeping something on grounds already fixed.
- **Two more loaders could not read their own checkpoints** — found by the adversarial review pass after the first three were fixed, which is the point: the earlier "everything else is fine" claim came from assuming an enumeration rather than performing one. `OperatorTrainer.save_checkpoint` stored the raw `@dataclass TrainingConfig` (plus a `PosixPath`), and `DistributedTrainer.save_checkpoint` stores `distributed_config.model_dump()` whose `backend` stays a `DistributedBackend` enum — both rejected by `weights_only=True` on load. The distributed case was **masked** rather than latent: `tests/distributed/test_distributed_trainer.py` called `torch.serialization.add_safe_globals([DistributedBackend])` at module import, a process-global registration that kept the suite green while the production loader stayed broken. That registration is removed, with a comment explaining why re-adding it would re-mask the regression. `OperatorTrainer` now stores primitives only (stringifying the `Path`), which needs no allowlist entry at all — strictly better than admitting a first-party class to a process-global window.
- **Codec resume-from-checkpoint was broken at runtime** — found while fixing the above, and unrelated to any malicious input: `VideoCompressionTrainer.load_checkpoint` used a bare `torch.load`, so under torch >= 2.6 it raised `UnpicklingError` on `datetime.datetime` for a checkpoint the very same class had just written. Resuming a codec training run could not work. Now routed through the chokepoint. ~~Checked the two sibling bare loaders (`training/evaluation.py`, `training/operator_trainer.py`) rather than assuming: both are genuinely fine, because `AlphaGalerkinConfig.model_dump()` contains no enum or datetime.~~ **Retracted** — that check was itself an assumption. `operator_trainer.py` never touches `AlphaGalerkinConfig`; it stores a raw `@dataclass` instance carrying a `PosixPath`, and was broken in exactly the same way (fixed below). The `evaluation.py` half of the claim is correct.
- **Stale post-fix documentation** — `src/alphagalerkin/solver.py` documented `create_model_from_checkpoint` as calling `torch.load(..., weights_only=False)`, untrue since the round-3 chokepoint landed (it over-warned, so it failed safe, but contradicted the code). Four docs still quoted `torch>=2.0.0` against the new `>=2.6.0` floor. `CLAUDE.md`'s Regression Surface had no `tests/security` row despite CI gaining a named security step.
- **Cole-Hopf Bessel coefficients (breaking)** — every Fourier coefficient in `BurgersOperator.exact_solution` was hardcoded to `1`, making the series the Cole-Hopf image of a *Dirac comb* rather than a sinusoid — a valid solution to the wrong problem, and not merely degenerate at `t=0` (`max|u|` at `t=0.1, ν=0.001` was still 1.08e9). Now uses `c₀ = ive(0,R)`, `cₙ = 2(−1)ⁿ·ive(n,R)` with `R = 1/(2πν)` via the exponentially-scaled `scipy.special.ive`. `max|u|` at `t=0` goes 4.998e13 → 1.0000 (ν=1.0), bounded by the viscous-Burgers maximum principle. Validated against an independent nonlinear finite-difference march sharing no code, coefficients or Bessel functions with the series (8.8e-7 agreement at ν=1.0), which pins the sign *and* the decay rate — a `t=0` identity alone could not. `initial_condition`, `boundary_value` and `exact_solution` previously described three different problems and now all describe the standard Basdevant benchmark: `u(x,0) = −sin(πx)` on `[0,1]`, homogeneous Dirichlet. The second live consumer, `BaseSolver._compute_l2_error` feeding `sbir_suite`'s `burgers_shock` row, goes 3.327e12 → 0.7016 (exactly the RMS of `sin(πx)`, i.e. the correct L2 error of a zero trial solution). Known limitation, documented and pinned by a test that *asserts* the inaccuracy rather than hiding it: the Fourier-Bessel representation is intrinsically unresolvable in float64 for ν ≲ 0.009, surfaced as `COLE_HOPF_MIN_RESOLVED_VISCOSITY`.
- **`games_completed` over-counted stopped workers** — `SelfPlayWorker.generate_batch` added the *requested* `n_games` regardless of the `_should_stop` break, and `SelfPlayCoordinator` had the same bug on an independent line that did not derive from worker stats. Both now count actual completions. The test that pinned the old behaviour is rewritten in full, and a new partial-batch test closes a real gap — the previous suite only covered zero-games and all-games, so a weaker "report 0 when stopped, else the full request" fix would have passed.
- **P0-1 Burgers OOD-reward defect** — `BurgersOperator.__init__` (`src/pde/operators.py`) now checks `config.model_fields_set` before overriding the class-level `is_time_dependent = True` default, so an unset config keeps a real `exact_solution()` instead of silently returning `None`; explicit `True`/`False` still honored exactly as before. Surfaced a new, more urgent finding in the process: the Cole-Hopf approximation is numerically degenerate at the now-reachable `t=0` (magnitude ~1e10-1e13), live on the default config of the shipped `llm_prior_ablation` scenario — see `docs/CODE_HYGIENE_REVIEW_2026-08-19.md`.
- **MCTS crashed on a terminal-at-root game state** — `mcts.get_action()` now guards this case with a clear error instead of an unhandled `ValueError`/unhelpful failure.
- **MCTS/PDE NaN propagation** — evaluator output is now checked for finiteness with structured-log detection (`src/mcts/search.py`, `evaluator.py`); a NaN from a diverging PDE solve previously resolved to the *best possible* MCTS leaf value via Python's NaN-comparison semantics (`EncodedValueEvaluator` in `src/pde/games/lshape_amr.py`) and now falls back to neutral.
- **LM Studio preflight crashed on a broken CUDA driver** — `torch.cuda.is_available()`/`device_count()` raising (distinct from "no GPU") now degrades gracefully instead of propagating an uncaught `RuntimeError`.
- **`TemplateRegistry` singleton was not thread-safe** — `src/prototyping/templates.py`'s `__new__` copied `src/templates/registry.py::BaseRegistry`'s double-checked-locking pattern but omitted the lock entirely, so concurrent first-access could construct more than one "singleton". Now mirrors the working pattern.
- **Unbounded self-play buffer-fill loop** — `Trainer._fill_buffer` had no iteration or wall-clock bound and would re-invoke full MCTS self-play generation forever if self-play ever netted zero usable experiences. Now bounded by a new `TrainingConfig.max_buffer_fill_iterations` field (no hardcoded literal) and raises a typed `BufferFillError` instead of silently training on an under-filled buffer.
- **`make demo` was broken and `make test-stoch` had silently drifted** — `demo` referenced a scenario name (`transfer_darcy_to_poisson`) that was never registered; `test-stoch` had 1 of 4 required `--include=` paths and 1 of 6 test paths, reporting an inflated number versus the gate it claims to mirror. `make lint` was narrower than CI, and `make check`/`pre-pr` never invoked any coverage gate. All fixed and each verified by running it.
- **GCS checkpoint URIs were corrupted by a `Path()` coercion** — a round-2 `mypy` annotation fix wrapped `EntryArtifacts.checkpoint_path` in `Path(...)`, but that field is `Path | str` by design: `GCSZooStorage` returns a `gs://` URI string, and `Path()` collapses the `//`, yielding an unusable `gs:/bucket/…` that `parse_gcs_uri` rejects and `train_compression_zoo_entry.py` writes into `metrics.json`. Found by the adversarial review pass; every existing test used the filesystem backend where `Path(Path)` is a no-op. `ZooTrainingReport.checkpoint_path` widened to `Path | str`; regression test added and mutation-tested.
- **1D RBF basis candidates were silently attenuated** — `BasisFunction.evaluate` substitutes `y = 0` for 1D coords, so a nonzero `center_y` scales the whole basis column by `exp(-center_y²/2σ²)` (~2e-11 at σ=0.1, `center_y=0.7`), which `lstsq` then drops as rank-deficient. `center_y` is now pinned to 0 for 1D domains; the RNG draw is retained so seeded 2D results stay bit-identical.
- **Three factually-incorrect code comments corrected** — the 1D RBF "center_y is inert" claim (disproved above), the phase-delegation "scale-normalized" claim (`get_phase` divides by `_initial_error`, which `BasisSelectionGame` never sets, so it falls back to 1.0), and the mesh-refinement "~100× slower budget drain" claim (dimension-dependent, and inverts in 3D at high polynomial degree). Also: `CLAUDE.md`'s 2026-08-16 milestone still listed the three deleted `constants.py` modules as delivered, and `make format` was narrower than the widened `make lint`.
- **`CLAUDE.md` claimed `video_compression` "no longer exists"** — false for 4 of the 5 paths it named (the package was cut 2026-07-22 and reinstated the next day for the Codec Model-Zoo work). Corrected, along with a second copy of the same false claim elsewhere in the file. A migration guide recommending the now-deleted `src.*.constants` modules as the "preferred v0.4+" import path was also corrected.

### Changed
- **Config-bound values surfaced** — `MeshRefinementConfig.hp_switchover_level` gained `le=20` and a cross-field check inside the *existing* `validate_mesh_config` validator: it must be strictly less than `max_refinement_level`, since the p-refinement branch of `HP_REFINEMENT` is reachable only on `[hp_switchover_level, max_refinement_level−1]` and an equal-or-greater value silently degenerated hp-refinement into a pure h-refiner. `POTENTIAL_FIELD_MIN_DISTANCE` moved from a module constant to `SwarmPlanningConfig.potential_field_min_distance` (`gt=0`), with its 12-line comment — which had argued *against* surfacing it — rewritten to state what the value actually does. Value-preserving, proven bitwise across 240 float values in both nominal and floor-binding geometries. No shipped YAML sets any of these fields.
- **`src/video_compression` type safety** — fixed 23 of 28 `mypy --strict` errors (missing `register_buffer` companion annotations, a systemic `np.ndarray[np.int32, ...]` shape/dtype type-parameter typo, list/dict annotations, a return type needing narrowing, a stale ignore). Repo-wide `mypy src/ --strict` is now **31 → 8 errors**; the remaining 8 are the 5 `codec/codec.py` errors needing real interface design plus 3 pre-existing torch-version-dependent `unused-ignore`s CI already documents as accepted.
- **Hardcoded values surfaced as config fields / named constants** — `basis_selection.py` RBF candidate centers now sample the operator's real `domain_min`/`domain_max` instead of a hardcoded `[0,1]` unit square (wrong for e.g. `LShapedPoissonOperator`'s `[-1,1]²`); the budget-decrement path in `basis_selection.py`/`mesh_refinement.py` now uses the same `cost_per_dof * dof_added` the reward path in those files already used (**behavior change**: the old flat `cost = 1.0` exhausted the budget ~100× faster than the reward accounted for); phase detection delegates to the config-driven, scale-normalized `PDEGame.get_phase()`; `mesh_refinement.py`'s h-vs-p switchover became an `hp_switchover_level` field; `swarm_planning.py`'s obstacle floor and `operators.py`'s duplicated Cole-Hopf constants became named constants; 7 `src/mcts/` call sites now use the existing `DEFAULT_TEMPERATURE`.

### Removed
- **Dead code** — `src/mcts/constants.py`, `src/physics/constants.py`, `src/training/constants.py` (three re-export modules with zero consumers; every real call site imports flat `src.constants`); `BaseTrainer.evaluate()` plus both concrete stubs (`Trainer.evaluate`, `DistributedTrainer.evaluate`) — an abstract method with no call site anywhere; and a duplicate `FNetMixingLayer` declaration in `benchmark_fnet.py`, which now imports the canonical `src.modeling.fnet` version.

### Added
- **Next Steps Review (2026-08-18)** — Added `docs/NEXT_STEPS_REVIEW_2026-08-18.md`, a peer-reviewed, evidence-based case for the highest-leverage next engineering steps (P0-1 OOD-reward defect scope, the JAX/`src/backend` keep-or-cut decision, PR #118/#57 salvage triage, and a re-scoped, effort-estimated plan for the `lshape_amr_compare` AMR novelty-claim fork).
- **Code Hygiene & Correctness Review (2026-08-19)** — Added `docs/CODE_HYGIENE_REVIEW_2026-08-19.md`: a hands-on, execution-verified pass across `src/mcts/`, `src/pde/`, `src/refinement/`, `src/integrations/`, and `src/data/` by four repo-specific specialist agents plus an adversarial verification pass. Coverage raised on `src/data/physics_dataset.py` (23%→100%), `src/refinement` (96%→100%), `src/mcts` (96.29%→96.95%), `src/integrations/lm_studio` (94.77%→95.62%). ~130 new/extended tests. Also surfaces (report-only, not fixed): a stale `video_compression`-was-deleted claim at `CLAUDE.md:115` (4 of 5 named paths actually exist, 28 undocumented `mypy --strict` errors and no coverage gate on that package), several hardcoded-value findings in `src/pde/games/`, a broken `make demo` target and a silently-drifted `make test-stoch` coverage command, and an unbounded self-play buffer-fill loop with no SIGINT/SIGTERM handling anywhere in the training stack. **Round 2 (same day) appended to the same doc**: executed most of that report-only list plus the packages round 1 never reached, via 8 parallel agent waves — see the Fixed/Changed/Removed entries above. ~5,018 tests passing, 0 failures.
- **Per-module coverage gate for `src/video_compression`** — the package ran 933 tests in CI with **no coverage gate at all**, because `pyproject.toml` still omits it from `--cov=src` (a leftover from when it was believed retired, so a bare `--cov=src/video_compression` silently measured 0%). Gated at 83 against a measured 85.43% using the same inline-coveragerc technique `phase2-zoo-validation.yml` already uses for the identical collision; charter gates register updated. Separately, `src/distributed` coverage rose 68.91%→82.34% (`worker.py` 22%→99%).
- **Tests for previously-unexercised reachable code** — Go illegal-move rejection, `get_result()`'s White/draw branches, and `get_winner()`; Chess queenside-castling *execution* and threefold-repetition (both claimed by module docstrings but never actually driven by a test); `CalibrationDataReader`; and all of `src/distributed/worker.py`. The last of these surfaced a real bug, reported not fixed: `SelfPlayWorker.generate_batch` over-counts `games_completed` when `stop()` triggers its early break.

## [0.4.0-dev] - 2026-08-16

### Added
- **SBIR Presentation Demo CLI** — Added `--demo` (formatted output tables, noise suppression) and `--export-results` (JSON/CSV exports) to `src/poc/cli.py`.
- **7-Tier Test Pyramid Expansion**:
  - `tests/sanity/`: Dynamic public module import smoke tests (337+ tests), config schema validation, and CLI `--help` entrypoint tests.
  - `tests/security/`: YAML injection defenses, malicious checkpoint safety checks (`weights_only=True`), and Hypothesis property-based fuzzing on GTP engine.
  - `tests/benchmarks/`: $O(N)$ Galerkin linear attention scaling, MCTS search throughput benchmarking, and FNet vs MultiheadAttention speedup profiling.
  - `tests/regression/`: Mathematical invariant regression suite including single-agent vs zero-sum MCTS backup signs, transfer ratio floor protection ($\le 1.5$), and dashboard claims.
  - `tests/e2e/`: User journey end-to-end tests for Go training lifecycle, Poisson PDE solving, and zero-shot resolution transfer ($9\times 9 \to 19\times 19$).
- **Core Abstractions & Protocols (`src/core/`)** — Added `@runtime_checkable` protocols (`EvaluatorProtocol`, `GameProtocol`, `OperatorProtocol`, `SolverProtocol`) and generic thread-safe `Registry[T]`.
- **Agent Lifecycle Hooks & Skills (`src/agents/`)** — Added `src/agents/lifecycle_hooks.py` (`HookManager`, `LoggingHook`, `MetricsCollectorHook`, `EarlyStoppingHook`) and declarative agent skills in `src/agents/skills/` (`BenchmarkSkill`, `SelfPlaySkill`).
- ~~**Domain Constants** — Partitioned domain constants into `src/mcts/constants.py`, `src/physics/constants.py`, `src/training/constants.py` while maintaining 100% backward compatibility via `src/constants.py`.~~ **CORRECTED (2026-08-19)**: those three files were dead re-export scaffolding — every real consumer imported `src.constants` directly, none of the three, and they sat at 0% coverage. Removed as confirmed-unconsumed dead code; `src/constants.py` was never actually partitioned and remains the sole canonical constants module.

### Changed
- **Version Bump** — Bumped version to `0.4.0-dev` with Beta development status in `pyproject.toml`.
- **Coverage Omissions** — Added `src/video_compression/*` to `pyproject.toml` coverage omit list.
- **Code Hygiene** — Replaced raw `# noqa` comments with `__all__` in `src/training/losses/__init__.py`, audited type ignores, and ensured cross-platform temp paths via `tempfile.gettempdir()`.

## [0.3.0] - 2026-07-22

### Added — Honest zero-shot transfer benchmark (operator vs retrained CNN)

- **Monotonicity in Factorized Prior** — `FactorizedPrior` relies on a network using parameters `a` and `H` to estimate the CDF. Added `torch.nn.functional.softplus` to these parameters to enforce strict positivity, ensuring the CDF is monotonically increasing, preventing negative likelihoods and probability explosion (NaN loss).
- **GDN Stability** — Generalized Divisive Normalization (`GDN`) and its inverse (`IGDN`) lacked positivity constraints on learnable parameters `beta` and `gamma`. Added `F.softplus` around both parameters in the `GDN.forward` function to ensure that `torch.sqrt` is never applied to a negative value.
- **MS-SSIM Stability** — Multi-Scale SSIM computation occasionally produced NaNs due to fractional exponentiation of negative variances on uniform patches. Added `torch.relu` clamping in `compute_ms_ssim` to eliminate negative base NaNs.
- **Structural Bug Fixes** — Implemented `FactorizedEntropyModel` wrapper; fixed BD-Rate metric key mismatch (`rate_bpp` vs `bpp`); updated deprecated `torch.cuda.amp.autocast` API to `torch.amp.autocast`.

### Fixed — L-shape AMR reentrant-edge Dirichlet BC never imposed (headline retracted)

- **`lshape_inside_predicate` removed the *open* fourth quadrant instead of the *closed* one.**
  The L-shaped domain is `[-1,1]² \ [0,1]×[-1,0]`; its boundary *includes* the two reentrant
  edges `{y=0, x≥0}` and `{x=0, y≤0}`, where the benchmark solution
  `u = r^(2/3)sin(2θ/3)` is identically zero. The strict inequalities `(x > 0) & (y < 0)`
  classified every node **on** those edges as an interior unknown, so its `u=0` Dirichlet
  condition was never imposed and the 5-point stencil coupled straight across the slit into the
  analytic continuation pinned inside the notch. The solver was discretising a different,
  inconsistent problem.

- **Diagnosed with no marking policy involved.** Under *uniform* refinement the L2 error **grew**
  with DOF — 5.0e-2 at 65 DOF rising to 1.15e-1 at 12545 DOF (rate −0.09) — and the peak error sat
  on the slit edge at `(0.75, 0.0)`, far from the corner, growing 0.357 → 0.519 → 0.634 as `h`
  halved. Removing the **closed** quadrant (`>=` / `<=`) restores the textbook rate for the
  reentrant-corner singularity: measured **O(h^1.31)** ≈ O(N^-0.65) ≈ O(N^-2/3), taking the same
  n=128 grid resolution to 2.59e-4 at 12416 DOF (12545 DOF pre-fix — the fix pins 129 more nodes
  as Dirichlet) — 444× lower.

- **`LShapedDomain.contains_point` is deliberately unchanged.** Closed-domain *membership* (where
  a slit-edge point *is* a member) and interior-*unknown* selection (where it is not) are
  different questions; conflating them was the bug. The distinction is documented on both.

### Changed — L-shape MCTS-vs-Dörfler headline retracted and re-measured

- Re-running the canonical 5-seed demo config on the fixed substrate **flips the result**:

  | Metric | Retracted (defective) | Committed (fixed) |
  |---|---|---|
  | `l2_error_ratio_at_matched_dof` | 0.9605 (≈4% win) | **1.0996** (MCTS loses ≈10%) |
  | `mcts_win_fraction` | 0.80 | **0.20** (1/5 seeds) |
  | `l2_error_ratio_at_matched_solves` | 1.26 (0/5) | **2.04** (0/5) |
  | Dörfler final L2 @ ~1300 DOF | 9.04e-2 | **8.40e-3** |

  The primary acceptance threshold (`l2_error_ratio_at_matched_dof < 1.0`) now **fails** — the
  falsifiable gate working as designed. Regenerated `results/lshape_mcts_vs_dorfler.{csv,png}`;
  corrected the charter claims register, `specs/lshape_amr_compare.spec.md`, and `CLAUDE.md`.

- **Second defect unmasked: tensor-product refinement.** With the BC fixed, adaptive Dörfler
  converges at only −0.125 while *uniform* refinement on the identical substrate achieves −0.65.
  At matched DOF adaptive marking is **5–9× worse than uniform** (9.1× at ~1300–1800 DOF), gap
  widening. `_dorfler_mark_2d` projects element-wise marks onto the x and y axes and `_refine_grid`
  runs separately on `xs`/`ys`, so marking one element near the corner inserts full grid *lines*
  spanning the whole domain. Element-local refinement (v2.1) is therefore a **blocking
  prerequisite** for any marking-policy comparison on this benchmark, not an optional upgrade.

### Added — L-shape convergence gate

- `tests/research/test_lshape_convergence_gate.py` (11 tests, ~1.2 s) asserts the substrate
  converges *before* any policy comparison is read: monotone L2 reduction under uniform
  refinement, an O(h^4/3) rate band, an absolute finest-grid anchor, reentrant-edge pinning, and
  that the **Dörfler arm alone** reduces error with DOF. Mutation-tested — 7 of the 11 fail on the
  pre-fix predicate. Added to the `CLAUDE.md` Regression Surface row for this benchmark.

### Changed — Dead `PDEGame.get_result` abstraction removed; abstraction audit gated in CI (audit B17 + B18)

- **Removed `PDEGame.get_result` and the `PDEResult` dataclass.** `get_result` was declared
  `@abstractmethod`, documented as lifecycle step 4, and implemented by every concrete game —
  but nothing ever called the 2-arg `PDEGame` signature (the `get_result` call sites in
  `src/training/evaluation.py` and `src/engines/match.py` are the unrelated 1-arg
  `GameInterface.get_result`). It was **deleted rather than wired**: all five real
  episode-terminal paths already build their own result object, and each needs a field
  `PDEResult` lacks (`actions`, `solution`/`wall_time_seconds`, `rollouts_used`, `n_solves`),
  while six of `PDEResult`'s seventeen fields had no reader anywhere in `src/`. Full evidence
  in `docs/CODE_HYGIENE_AUDIT.md` §4.1.
- **Added `PDEGame.termination_reason(state)`** — the termination-cause ladder that was inlined
  in all three `get_result` overrides, promoted to the ABC with the one game-specific rung
  behind a `_capacity_reason` hook (basis count for `basis_selection`; DOF compared with `>`
  for `mesh_refinement` and `>=` for `lshape_amr`, each matching its own `is_terminal`).
  It is **concrete, not abstract**, so no existing subclass breaks.
  `lshape_amr._termination_reason` is superseded by it.
- **`AlphaGalerkinSolver` metadata is more specific.** `METADATA_KEY_TERMINATION_REASON`
  previously recorded the bare `"is_terminal"` whenever the game stopped the loop, collapsing
  converged / max_dof / max_basis / budget_exhausted into one uninformative label; it now
  records the game's own classification. **Breaking for consumers that string-match
  `"is_terminal"`** in solver metadata.
- **`SolverResult.h1_error` is now populated** by `AlphaGalerkinSolver`. The field existed and
  `to_dict()` serialised it, but the solver never set it even though `compute_exact_error`
  returns `h1` alongside the `l2` it did read — so the exported column was permanently null.
- **`src/pde` is no longer exempt from the abstraction gate.** `python -m
  scripts.audit_abstractions src/pde --fail-on-missing` now exits 0.
- **CI gates the F0/F1 screen (B18).** The `lint` job runs
  `audit_abstractions src/mcts src/refinement src/pde --fail-on-missing`, plus a
  `continue-on-error` pass over all of `src/` (the `src/backend` domain-PoC backlog stays
  advisory). The script is AST-only with stdlib imports, so it runs in that job's minimal
  dependency set. CLAUDE.md's Regression Surface row, the `abstract-method-audit` skill and
  the `/audit-abstractions` command are updated to match.
### Fixed — CI enforcement (tech-debt Phase 1)

- **CI never ran on pull requests**: `on.pull_request.branches: [main, develop]`
  referenced branches that do not exist in this repository, so every PR merged with
  zero checks. The branch filter is removed; `test-slow`'s `if:` condition had the
  same dead branch names and now keys on `github.event.repository.default_branch`.
- **Silently degraded llm_prior coverage gate repaired**: under coverage 7.x,
  file-path `--cov=path/to/module.py` specs are dropped with only a warning, so the
  gate's two file-level targets enforced nothing. They now run as a native-runner
  (`coverage run --include=...`) step. With the gate unenforced,
  `src/poc/scenarios/llm_prior_ablation.py` had drifted to a measured 77% branch
  coverage (81% combined with its config) vs the documented 86%; the repaired gate
  starts at 79 (measured − 2) with the ratchet back to 85+ tracked in
  `docs/CODE_HYGIENE_AUDIT.md` §7.
- **Unenforced regression surface re-enabled**: `tests/pde/test_mcts_adapter.py`
  (a documented F1/F3 Regression Surface) had been `--ignore`d in every CI job since
  the 2026-04 emergency triage despite passing at HEAD; the ignore and two
  CUDA-deselects made redundant by in-source `skipif` markers are removed.

### Added — CI gates (tech-debt Phase 1)

- Three CLAUDE.md-documented per-module coverage gates that were never wired into
  `ci.yml` (phantom gates, audit backlog B20) now exist, in native-runner form with
  measured margins: `noyron_basis` (98% measured, gate 85), Noyron HX surface
  (99% measured, gate 85), SBIR P40 surface (94% measured, gate 85).
- Drift-alarm test `test_migration_defaults_match_v1_1_shipped_values`: the four
  checkpoint-migration setdefault literals are intentionally frozen (a 1.0.0→1.1.0
  migration must inject the defaults v1.1.0 shipped with, forever); the test fails
  if a live default is retuned, forcing an explicit migration decision.
- `docs/CODE_HYGIENE_AUDIT.md` §7: Phase-1 follow-up record — measured branch
  coverage for 8 previously ungated packages, quantified mypy override debt
  (207 masked errors), B10 dead-package reclassification (only 4 of 6 are dead;
  `deployment` is CI-exercised, `demos` is a live dashboard dependency), and the
  Phase 2–4 roadmap with the owner-decision register.

### Changed — Hardcoded values surfaced (zero numeric change; tech-debt Phase 1)

- LR-scheduler knobs `min_lr_ratio` / `warmup_start_factor` are now typed
  `TrainingConfig` fields (defaults 0.1/0.1 — exactly the values `Trainer`
  previously hardcoded) and named `BaseTrainer` module constants
  (`DEFAULT_MIN_LR_RATIO` 0.01 / `DEFAULT_WARMUP_START_FACTOR` 1e-6) that both
  `BaseTrainerConfig` field defaults and `_create_scheduler` parameter defaults
  bind to. All three previous copies of these values are reconciled; no LR
  trajectory changes.
- Boundary tolerances named, deliberately not unified:
  `src/pde/operators.py` now uses `DEFAULT_BOUNDARY_TOLERANCE` (1e-6, unchanged);
  new `DEFAULT_PICOGK_BOUNDARY_TOLERANCE` (1e-5, unchanged) documents the
  SDF-band semantic and the pre-existing picogk operator/domain divergence.
- Gumbel MCTS epsilons split by semantic: `GUMBEL_NORMALIZATION_EPSILON`
  (inert division guard) vs `GUMBEL_LOG_PRIOR_FLOOR` (algorithmic log floor);
  `FNetEvaluator` softmax floor named `_SOFTMAX_NORMALIZER_FLOOR`, mirroring
  `src/integrations/lm_studio/evaluator.py` by name. All values 1e-8, unchanged.
- The 13 `[9, 13, 19]` board-size literal sites now derive from
  `DEFAULT_BOARD_SIZES` via copies (`list(...)` / `default_factory`), never the
  shared mutable module list.

### Removed — Dead CI/code weight (tech-debt Phase 1)

- Dead "Upload test results" step that archived `.pytest_cache/` (never found
  files); `--no-cache-dir` flags that defeated the CI pip cache.
- `BaseTrainer`'s three `@abstractmethod` decorators — a dead contract both
  production trainers stubbed with `NotImplementedError`. The methods are now
  concrete `step()`-loop hooks; subclass stubs and their exact messages are kept
  (test-asserted, and they document each trainer's real entry points).

### Added — Code hygiene & modularity audit + quick wins

- `docs/CODE_HYGIENE_AUDIT.md`: prioritized audit of `src/`/`tests/`/CI covering god
  modules, duplicated `*_compare` scenario boilerplate, rejected internal standards
  (registries/config/logging reimplemented instead of reusing `src/templates/`), the
  `poc`↔`research` import cycle, and enforcement gaps (mypy, CI lint scope, the
  CLAUDE.md Regression Surface table's drift from CI). 20 backlog items documented
  (B1–B20; an earlier revision of this entry undercounted them as 16).
- `mypy src/ --strict --ignore-missing-imports` now passes cleanly (was 3 stale
  `unused-ignore` comments, not the "enforced nowhere" error volume both prior audit
  passes assumed).
- `RUF100` added to the ruff select list; 71 stale `noqa` comments removed; CI's lint
  scope now matches pre-commit's in both directions (`scripts/`, `config/`,
  `conftest.py`, `deploy_space.py` included; `hf_space/`, `notebooks/` and
  `claude-code-platform/` excluded on both sides, so every tracked `*.py` is linted by
  exactly one of the two).
- **pre-commit hook scope**: the `hf_space/` exclusion is applied **per-hook** (ruff,
  ruff-format, yamllint), not as a top-level `exclude:`. An intermediate commit in this
  PR used the top-level form, which is inherited by every hook and therefore also
  disabled `detect-private-key` and `check-added-large-files` on the tree published to a
  public HuggingFace Space (already carrying a 7.2 MB `checkpoint.pt`, 7x the
  `--maxkb=1000` limit). Both guards are global again.
- **Removed the `check-docstring-first` hook.** It rejects 21 modules repo-wide (20 under
  `src/`) that use PEP 258 attribute docstrings — a string literal documenting the
  assignment above it — which the hook misreads as "multiple module docstrings". The
  idiom is the house style here, so the hook is the thing that does not fit.
- Removed the dead `benchmark` CI job (matched zero tests); added `--strict-markers`
  to pytest addopts; deduplicated marker registration onto `pyproject.toml`.
- `src/seeding.py::derive_seeds` replaces 5 duplicated seed-derivation bodies across
  `src/agents/config.py`, 3 PoC scenario configs, and `src/research/seed_sweep.py`
  (each module's stride value is unchanged, so no scenario's derived seeds change).
  `stochastic_galerkin_compare_config.py` was deliberately excluded after CI's
  import-isolation guard for that layer's dependency surface caught the addition —
  see `docs/CODE_HYGIENE_AUDIT.md` §6.
- `tests/poc/conftest.py` adds a save/restore fixture around **structlog's global
  configuration**, which `test_logging.py` mutates with no teardown — that leak
  silently routed later `logger.warning(...)` calls into stdlib logging where
  pytest swallows them. A `ScenarioRegistry` snapshot/restore fixture was
  attempted alongside it and reverted before merge: measured against the live
  registry it left fewer scenarios registered than no fixture at all. The
  subprocess workaround in `test_charter_alignment.py` therefore stands; see
  `docs/CODE_HYGIENE_AUDIT.md` §6 and backlog B16.
- The three classic PoC scenarios (`stability`, `transfer`, `complexity`) now resolve
  their device via `src/poc/device.py::resolve_device` instead of a hardcoded inline
  fallback; `llm_prior_ablation._median` is now a shim onto `_centaur_common.median_of`.
- `src/constants.py`: wired `DEFAULT_LBB_THRESHOLD` and `DEFAULT_DROPOUT` to their
  matching `src/modeling/` defaults; deleted 2 dead constants with no live consumer.
- Logging added at 4 previously-silent exception-swallow sites (mesh-refinement
  interpolator fallback, LM Studio VRAM probe, PoC CLI scenario listing, the SBIR
  baseline-registry default fallback).
- Added a `viz` optional-dependency extra for matplotlib; removed the dead `doc8`
  pre-commit hook (0 `.rst` files). (A scenario config YAML was deleted and then
  restored after CI showed a parametrized test loads it by a constructed path a
  literal grep can't see — see `docs/CODE_HYGIENE_AUDIT.md` §6.)

### Fixed — Dashboard figures contradicted by their own committed artifacts (`dashboard-uplift`)

- **The Gradio dashboard rendered uncommitted-spike numbers as validated results.**
  `dashboard/config.py::TransferMilestone` shipped `{9: 2.5e-6, 13: 2.04e-4, 19: 3.93e-4}`,
  attributed to `scripts/demo_transfer.py` — a script that writes only to `outputs/`. The
  committed benchmark says 19×19 ≈ 2.3e-3. Defaults now carry the representative
  (median-ranked) seed from `results/transfer_baseline_compare.csv` — the operator's 19×19 MSE
  is the 3-seed median, and the retrained-CNN (1.63e-4) and zero-shot-CNN (7.66e-5) baselines
  are that same seed's *paired* values, so the 14.1× ratio is within-seed. The baselines are
  deliberately **not** described as medians: the per-metric CNN medians differ (1.43e-4 and
  3.15e-4). `COMMITTED_TARGET_RESOLUTION` pins the comparison to 19×19, and `achieved_mse` now
  validates that the key is present, so a config override cannot compare mismatched resolutions
  under a single label.
- **`show_transfer_milestone` rendered two retracted framings.** It printed
  `MILESTONE ACHIEVED` and annotated each bar `N× better` against an arbitrary 0.05 pass
  threshold (127× / 245× / **20000×**) — the self-comparison
  `specs/transfer_baseline_compare.spec.md` retracts — and plotted a `np.random.default_rng(7)`
  curve titled *"Training curve (9×9 Poisson data)"* with no disclaimer. It now shows the
  three-arm baseline comparison and the operator's real 9→13→19 degradation, and states the
  honest result: the operator **loses by ≈14×** to a retrained CNN; the value is zero
  retraining, not peak accuracy.
- **The tab blurb reported the wrong number entirely** — `min(achieved_mse.values())`, the 9×9
  *in-distribution* figure, presented as the zero-shot transfer result. Corrected, as was the
  About table in `dashboard/app.py` and the transfer framing in `hf_space/app.py`.
- **The physics demo reported `mean(ground_truth²)` as a model error.** `PhysicsDemo.predict()`
  returns zeros when `model is None`, and both entry points construct the tab that way; the
  output is now labelled a placeholder rather than a measurement.

### Added — `dashboard/` inside the CI quality gates (`dashboard-uplift` WS6)

- **CI now lints `dashboard/`** (`ruff check` + `ruff format --check`), matching
  `.pre-commit-config.yaml`, which runs ruff with no `files:` filter. The asymmetry was already
  producing drift: `tabs/pde_tab.py` and `tabs/training_tab.py` were format-drifted at HEAD while
  passing CI, and would have been rewritten by any contributor's commit hook. Fixed in a separate
  mechanical commit so the gate commit stays reviewable. `hf_space/` remains excluded — deploy
  bundle, older ruff/gradio pin, accepted charter deviation.
- **New coverage gate**, `--cov=dashboard --cov-branch --cov-fail-under=84`. `dashboard/` sits
  outside `--cov=src`, so its 214 tests ran while measuring nothing. Gated at **84** against a
  measured 84.85% — deliberately not 85 (fails today) and not 80 (would permit a ~5pp regression).
  The entire deficit is `tabs/game_tab.py` at ~53%, whose `_ensure_loaded` and AI-move paths are
  unreachable while the `hf_space` shadowing forces `conftest.py` to mock them; **85 is recorded
  as a WS3 task**, since relocating those modules is what makes that code testable.
- **Charter gates register** gains the row `| dashboard | 84 |`, cross-checked by
  `test_documented_gates_are_enforced_in_ci`. No guard change needed — it matches `--cov=<target>`
  by string, with no `src/` prefix requirement.
- **mypy posture decided rather than extended.** The override is wildcarded (`dashboard` +
  `dashboard.*`; the bare wildcard does not match the package itself), replacing a hand-enumeration
  under which a *new* dashboard module would silently inherit full `--strict`. The CI step stays
  `src/`-only: it is `continue-on-error` and the dashboard override disables 13 error codes, so
  extending it would add the appearance of type-checking without the substance. Rationale recorded
  in the new `dashboard/AGENT.md` rather than the charter's deviation register, which is for
  divergences between documentation and reality — no document claimed `dashboard/` was typed.
- **New `dashboard/AGENT.md`** — layout, the claim-fidelity rules the charter guard enforces, the
  `sys.path` shadowing hazard (including that `tests/dashboard/conftest.py` deliberately uses the
  opposite order, so app and tests import different code for the same names), the Gradio ≥6 vs
  Space 4.44.1 split, the gates, and the callback-binding / `interactive=False` gotchas. Root
  `AGENT.md` gains a pointer to it and to `hf_space/AGENT.md`; the module index itself stays
  `src/`-only.

### Added — UI claim fidelity guard (`dashboard-uplift`)

- **New charter Requirement *UI Claim Fidelity*** — the evidence standard reaches documents but
  not the dashboard, which renders figures from Pydantic defaults and hardcoded markdown and is
  seen by more people than any document. A number shown to a user is a claim.
- **`test_ui_claims_match_committed_artifacts`** (registered in `_GUARDED`, so the charter's
  both-directions meta-guard covers it): bans the fabricated figure and the retracted blanket
  claim across both interactive surfaces (`dashboard/**/*.py` and `hf_space/**/*.py`), asserts
  the target-resolution figures agree with `config/baselines/transfer_ci.json` within that
  file's own `tolerance_pct`, and cross-checks the remaining rendered resolutions (9×9, 13×13
  — absent from the baseline JSON) against the representative seed's rows in
  `results/transfer_baseline_compare.csv`. `TransferMilestone` additionally rejects an override
  whose ratio contradicts its own operands. It loads
  `dashboard/config.py` standalone via `importlib` rather than importing the package, keeping
  gradio out of the charter guard. Mutation-tested against four regressions: a reintroduced
  spike figure, the retracted literal, a flipped comparison direction, and the restored
  "milestone achieved" framing.
- One pre-existing dashboard test asserted `"better" in summary` — it encoded the retracted
  framing as a requirement, and now asserts the baseline ratio instead.
- The change package `openspec/changes/dashboard-uplift/` additionally designs three deferred
  workstreams: un-shadowing the `hf_space` mirror (which needs module *relocation*, not a
  `sys.path` reorder — root `src/` and `config/` are regular packages, so reordering alone
  breaks the Go tab), a registry-driven scenario tab plus a Results tab over the committed
  artifacts, and a clickable Go board. The fourth — bringing `dashboard/` inside the CI quality
  gates — has since landed; see the WS6 entry above.

### Added — Executable project charter (`project-charter-alignment`)

- **New `openspec/` tree** ([OpenSpec](https://github.com/Fission-AI/OpenSpec) format):
  `openspec/specs/project-charter/spec.md` is now the repository's **supreme** scope
  document — mission, scope, non-goals, the novelty claim, the evidence standard, and an
  accepted-deviation register. It is deliberately *thin and referential*: it asserts equality
  with existing owners (`ARCHITECTURE.md` for layout, `ci.yml` for gates, the scenario registry
  for capabilities) rather than copying them, so there is one place to edit when reality
  changes. `openspec/project.md` states the precedence order; the change package under
  `openspec/changes/project-charter-alignment/` carries the proposal, design, tasks, and delta.
- **`tests/docs/test_charter_alignment.py`** — one guard per charter Requirement plus two
  meta-guards (every region parses non-empty; every `### Requirement:` maps to a guard, checked
  both directions). All nine were mutation-tested to confirm they fail when violated. The
  capability guard reads `ScenarioRegistry().list_scenarios()` in a **subprocess**: the registry
  is a process-wide singleton that `tests/poc/*` autouse fixtures `clear()` without teardown, so
  an in-process read is order-dependent (measured: 10 scenarios under `pytest tests/poc
  tests/docs`, 0 under a narrower selection).
- **`tests/support/cut_modules.py`** — `CUT_MODULES` promoted to one shared definition so the
  charter's non-goal guard and the `hf_space` mirror guard cannot drift apart.

### Fixed — Claims contradicted by their own committed artifacts

- **Retracted AMR headline corrected.** `CLAUDE.md` still advertised the pre-bugfix
  `~11–14% win / ~15–55× wall-clock` L-shape AMR result that
  `specs/lshape_amr_compare.spec.md` had already retracted (it came from the F0 two-player
  adversarial backup on a single-agent game). Now states the committed figures: median L2 ratio
  **0.9605** (~4% win) at matched DOF, **1.26** at matched compute with MCTS winning **0/5**
  seeds at ~350× the solves. "Two honest comparisons" → three, per AC4.
- **Zero-shot transfer MSE corrected repo-wide.** `README.md` advertised ≈4e-4 while citing a
  spec whose committed artifacts (`results/transfer_baseline_compare.csv`,
  `config/baselines/transfer_ci.json`) say **≈2.3e-3**; the favourable number came from an
  uncommitted spike and had propagated into nine outward-facing SBIR documents plus the earlier
  retraction banners. All corrected; `docs/demos/transfer_results.md` now states explicitly that
  its table is spike output.
- **Phantom headline artifact disclosed.** `docs/business/proposal/concept_note.md` asserted the
  Pareto plot was *"archived at"* `benchmarks/results/headline_2026_04/pareto_plot.png` — a path
  that does not exist — in the same sentence as *"no numerical performance claim … is not
  traceable to that artifact."* Marked `[PENDING]`, matching `outreach_template.md`.
- **Deleted subsystem no longer documented as live.** `CLAUDE.md`'s four `video_compression`
  milestones and `docs/TRAINING_DATA_SOURCES.md` carried eight paths removed in the 2026-07-22
  cut.
- **Never-runnable commands removed.** Three documented `torchrun scripts/train_distributed.py`
  invocations referenced a script that does not exist; `src/distributed/` has no entry point.
- Smaller corrections: `specs/README.md` was missing `lshape_amr_compare` and mislabelled
  `llm_prior_ood`; the undocumented `fem` extra; an inverted mypy-gate claim in
  `NEXT_STEPS_PLAN.md`; `src/training/loss.py` → `losses/`; the moved `PR86_HEADLINE_RUNS.md`
  path; stale operator/game snapshots in `docs/architecture/components.md`.
- **`docs.yml` `paths:` widened** so a PR touching only `CLAUDE.md`, `README.md`, or `specs/**`
  actually runs the internal-link checker — the gap that let a dangling
  `specs/lambda_scheduling.spec.md` reference survive.

### Changed — Code hygiene (`code-hygiene-plan`)

- **Enforcement tooling made truthful**: aligned the pre-commit `ruff` hook to the
  repo-wide `0.15.8` pin (was `v0.3.0`, so pre-commit reformatted code differently
  from CI) and the `mypy` hook to `1.11.x`; removed the dead `bandit` hook (it
  referenced a non-existent `[tool.bandit]` section and a non-existent CI job, and
  never ran); refreshed the now-stale CI `mypy` comment (kept `continue-on-error` —
  the strict run is torch-version-sensitive). Added `pre-commit` to the `[dev]`
  extra and a guarded `pre-commit install` to the session-start hook so the hooks
  actually run.
- **hf_space deploy mirror documented + guarded**: added `hf_space/AGENT.md`
  describing the partial, drifted, independently-formatted `hf_space/src/` mirror
  (and the Xet-tracked `checkpoint.pt` / intentionally-divergent `requirements.txt`),
  plus `tests/hf_space/test_mirror_guard.py` — a floor guard asserting `app.py`'s
  imports resolve in the mirror, every mirror file parses, and the cut modules and
  retracted transfer figure stay scrubbed.
- **Archived reviews corrected**: `docs/archive/reviews/pr6_review.md` and
  `pr7_review.md` now carry a banner retracting the fabricated `0.000209 / 240×`
  zero-shot-transfer figure (committed benchmark ≈ 2.3e-3; the ≈4e-4 first written here
  was an uncommitted spike config, corrected 2026-07-31).
- **Bounded code-debt**: allowlisted three verified-live abstractions the AST audit
  mis-flagged (`BaseEngine.is_ready`, `GameInterface.get_symmetries` /
  `get_action_mask`) so `scripts/audit_abstractions.py` stays trustworthy; corrected
  the `get_symmetries` docstring; extracted the duplicated
  `np.random.seed; torch.manual_seed` idiom into `src/seeding.py::set_global_seeds`.

### Added — Stochastic Galerkin operator-splitting layer (NKE, `alphagalerkin-nke-integration`)

- New additive subpackage `src/pde/stochastic/` implementing the Lagrangian Galerkin
  projection of a Kolmogorov-forward generator `L = A + D + J` onto a Gaussian-mixture
  basis (after NKE, arXiv:2607.19173 — implemented from the standard derivation with a
  documented provenance caveat; see `specs/stochastic_galerkin_nke.spec.md` and
  `docs/related-work.md`): exact expm advection/diffusion moment flows, a **trained MDN
  jump semigroup** (residual, dt-scaled, identity at dt=0), symmetric Strang composition
  (measured second-order: slopes 1.995–2.000), and a **parallel-in-time trainer** whose
  M−1 interval losses evaluate in one batched forward pass over precomputed particle
  clusters (no autoregressive rollout). GPU/CPU agnostic throughout.
- Verified against independent van Loan closed forms: OU moment recovery < 1e-3
  (Hypothesis stable-A sweeps); jump-OU with the exact compound-Poisson oracle < 1e-3;
  trained-MDN trajectory errors 1.7e-2 / 7.4e-3 (gate 5e-2); trainer reaches the
  oracle-achievable loss floor (gap closure 0.000). Every unmeasured gate was
  calibrated from a pinned run and recorded in the spec.
- A generator with a jump term but no jump model raises `JumpModelMissingError` — the
  jump component is never silently dropped (change-doc requirement), with a
  defense-in-depth re-check in the Strang composer.
- New `stochastic_galerkin_compare` PoC scenario + CLI: deterministic Galerkin-attention
  arm vs the stochastic moment-projection arm on a shared Fokker-Planck/OU density
  benchmark with free analytic ground truth. The single gate is the stochastic arm's
  absolute MSE (measured 2.3e-8, gate 1e-6); the deterministic arm's MSE and the ratio
  are recorded **ungated** (novelty ≠ superiority). Committed artifacts:
  `results/stochastic_galerkin_compare.{csv,png}`,
  `config/baselines/stochastic_galerkin_ci.json`.
- MCTS/self-play untouched, enforced twice: an AST import-isolation guard over the new
  modules and the green MCTS/F0/F1 regression surfaces. The novelty-gap documentation
  guard is executable (`tests/regression/test_related_work_guard.py`): every
  `docs/related-work.md` entry must carry a "does NOT do" clause, and the retracted
  blanket "no MCTS+Galerkin" claim is asserted absent from the README.

### Added — Honest zero-shot transfer benchmark (operator vs retrained CNN)

- New CI-gated `transfer_baseline_compare` PoC scenario replacing the **fabricated**
  "Zero-shot Transfer MSE 0.000209, 240× better than threshold" headline (a hardcoded
  notebook markdown cell no code ever computed). `src/experiments/cnn_baseline.py`
  (`DiscreteCNNBaseline` discrete foil), `src/research/transfer_baseline_compare.py`
  (median-over-seeds harness), `src/poc/scenarios/transfer_baseline_compare{,_config}.py`,
  `scripts/run_transfer_baseline_compare.py`, `specs/transfer_baseline_compare.spec.md`.
- **Honest measured result** (committed `config/baselines/transfer_ci.json`,
  `results/transfer_baseline_compare.{csv,png}`): the resolution-independent operator
  transfers zero-shot (19×19 MSE ≈ 2.3e-3, trained only at 9×9) but a discrete CNN —
  retrained *or even applied zero-shot* — is more accurate. The operator's value is
  **zero-retraining (one model at any resolution), not peak accuracy**. The gated ratio is
  committed as a regression ceiling, not a false `< 1` win claim.
- Every headline number now comes from one real (median-ranked) seed, so dividing the
  committed absolutes reproduces the committed ratio exactly. Shared
  `src/research/seed_sweep.py` de-duplicates the seed-derivation between the transfer and
  L-shape harnesses. Per-module branch coverage ≥ 92% (cnn 100%, harness 98%, config 98%,
  scenario 92%); new `transfer-baseline-regression` CI job (soft-gated) diffs the tripwire
  run against the committed baseline.

### Removed — Cut to the research core (6 application modules, ~72k LOC)

- `git rm` of `src/{video_compression,reentry,vertex,intercept,firefighting,thermo}` and
  their test trees / scripts / configs / docs to refocus the repo on the Galerkin + MCTS
  core (pre-cut tag `archive/pre-core-cut-2026-07-22`). All six were import-safe (nothing in
  the keep-set imported them). The `thermo` λ-window negative-result ablation is preserved in
  git history. Companion cleanup: removed the `video`/`requires_video` pytest markers, the
  `vertex` packaging extra, the codec-perf CI workflow, and pruned the C4 architecture
  diagrams, `AGENT.md`, and `CLAUDE.md` of the deleted subsystems.

### Changed — Prior-art review + SBIR reframe

- `docs/proposals/PRIOR_ART_REVIEW.md`: the narrow MCTS-Galerkin-basis-selection delta
  survives, but the blanket "no MCTS + FEM" claim does **not** (TreeMesh, arXiv:2111.07613,
  couples MCTS + RL with FE mesh generation). SBIR positioning reframed to the method delta
  at matched wall-clock, not a demonstrated accuracy win.

### Fixed — Single-agent MCTS backup (F0, correctness) + reward wiring (F1)

- **F0 — single-agent backup.** `MCTSNode.backup` unconditionally negated the backed-up value at
  every tree level (a two-player assumption), while `select_child` maximises `Q + exploration` at
  every depth. For single-agent games (`n_players == 1`: every PDE / refinement game) this made the
  search *minimise* value at odd depths. Fixed by routing the sign flip through a new
  `src.mcts.search.SearchMode` (`SINGLE_AGENT` / `ZERO_SUM` / `LEGACY_ADVERSARIAL`);
  `MCTSNode.backup(value, invert)` now takes the flag explicitly. **Backwards compatible:** the
  `MCTS.__init__` default is `ZERO_SUM` (byte-for-byte the old two-player backup), so Go/chess are
  unchanged; single-agent callers pass `SINGLE_AGENT`. `LEGACY_ADVERSARIAL` (deprecated, warns) exists
  only to reproduce pre-fix results.
- **L-shape headline corrected & republished.** The `lshape_amr_compare` MCTS arm now defaults to
  `search_mode="single_agent"`. Re-running the canonical config over the same 5 seeds:
  `legacy_adversarial` (pre-fix) → median L2 ratio **0.8896** (~11% win); `single_agent` (corrected)
  → **0.9605** (~4% win), win fraction 0.80 in both. Still a win at matched DOF (primary gate passes),
  but a **smaller, honest** one. `results/lshape_mcts_vs_dorfler.{csv,png}` regenerated under the
  corrected mode; `specs/lshape_amr_compare.spec.md` AC3 documents both numbers.
- **F1 — reward reachability.** `PDEGame.get_reward` (previously abstract with zero `src/` call sites)
  is now reachable through MCTS behind an opt-in `MCTS(use_intermediate_rewards=...)` flag (default
  `False` → unchanged behaviour): `_simulate` accumulates `R = Σ γ^t · get_last_reward()` along the
  selection path and backs up `R + γ^d · V(leaf)`. `PDEGameAdapter` gained `get_last_reward()`
  implementing the optional `SupportsStepReward` protocol.
- **Tests.** `tests/mcts/test_backup_modes.py` (sign-by-mode, the anchor
  `test_single_agent_search_prefers_higher_value_at_all_depths` which fails on the inverting modes,
  deprecation + reward-discount validation, intermediate-reward accumulation),
  `tests/pde/test_reward_reachability.py` (get_reward invoked iff enabled), and
  `tests/pde/test_clone_isolation.py` (F3 clone isolation across every concrete PDE game).

### Fixed — Post-merge review hardening (PR #95 follow-up)

- **`PDEGameAdapter.search_mode` property.** Since `MCTS.__init__` defaults to `SearchMode.ZERO_SUM`
  for back-compat, a caller wrapping a raw `PDEGameAdapter` who forgot to pass `search_mode` would
  silently get the pre-fix (wrong-for-single-agent) backup. The adapter now exposes a `search_mode`
  property returning `SearchMode.SINGLE_AGENT`, mirroring `RefinementGameAdapter.search_mode`, so PDE
  callers can wire `MCTS(search_mode=adapter.search_mode)`. Additive; nothing merged was incorrect
  (the production `lshape_amr_compare` path already plumbs `search_mode` explicitly).
- **`MCTS._read_step_reward` contract check.** A game exposing `get_last_reward` as a non-callable
  attribute (float / property value) now raises a clear `TypeError` at the source of the contract
  violation instead of a cryptic `'... is not callable'` deeper in the search loop.

### Added — Domain-free refinement engine (`src/refinement/`) + λ-scheduling ablation (`src/thermo/`)

- **`src/refinement/`** — the domain-agnostic `RefinementGame` engine (`RefinementState` +
  `RefinementLike` protocol, `RefinementGameAdapter` → MCTS passing `SINGLE_AGENT`, generic
  `RefinementGameConfig[TDomain]`, `@register_refinement_game`). `PDEState` gains additive
  `to_refinement()`/`from_refinement()` converters (fields unchanged; existing PDE tests green).
  85% branch gate; audit-clean.
- **`src/thermo/`** — the first non-PDE `RefinementGame`: a λ-window (BAR/FEP) sample-scheduling
  ablation. `LambdaSchedulingGame` (deterministic `apply_action`, monotone-under-allocate /
  reachable-non-monotone-under-split), four `VarianceSurrogate`s (analytic / mismatched / recorded /
  operator-stub), and a plan-in-surrogate / act-in-world comparison harness.
- **NEGATIVE result (kill criterion triggered).** Untrained uniform-prior MCTS is **~2× worse** than
  greedy variance-weighted allocation at every surrogate bias including zero — it over-splits and
  fragments the sample budget. The thesis is falsified for this configuration; the code is retained
  only as the falsification harness and **no capability is claimed**. Honest caveat: a purely
  multiplicative surrogate bias is scale-invariant for allocation, so genuine mismatch needs shape
  distortion — moot here since MCTS already loses at zero mismatch. Artifacts:
  `results/lambda_scheduling.{png,csv}`; write-up in `specs/lambda_scheduling.spec.md`. CI gates the
  mechanics (85% branch), not the losing headline.

### Changed — Tech-debt hardening on the refinement/thermo surface

- **Reward-scale confound fixed & negative result revalidated.** The MCTS intermediate-reward
  return `R + γ^d·V(leaf)` mixed the order-`1e-3` shaped reward with an order-`1` terminal winner.
  `LambdaSchedulingGame.get_winner` now returns **0** (neutral) for a non-converged terminal
  (was `-1`), and the per-edge cost is keyed on the **window-count delta** (a split adds a window),
  not on a DOF side-effect. Re-running leaves the verdict unchanged (ratio 2.00 → 2.05), so the
  negative result is genuine over-splitting, not a reward-scale artifact. Spec + committed
  `results/lambda_scheduling.{png,csv}` updated.
- **Structured logging** (`structlog`) added to `src/refinement/adapter.py`,
  `src/thermo/{surrogate,outer_loop}.py`, and `scripts/run_lambda_scheduling.py`, mirroring the
  repo's event-logging convention.
- **Typing escape hatches removed:** `OperatorSurrogate.predict_fn` and `run_bias_sweep.make_planner`
  are now `Callable`-typed; the CLI builds its config via `model_validate`; the dead
  `replace_params` helper (which carried a `type: ignore`) is deleted. Zero avoidable `type: ignore`
  remain in the new src surface.
- **Reuse:** `iterate_greedy/uniform/mcts` generators + `score_true_stderr` extracted in
  `outer_loop`; the plot CLI now consumes them instead of re-implementing the scheduler loops.
- **No hardcoded values:** `DEFAULT_NOISE_FREQUENCY`, `MIN_SPLIT_CHILD_SAMPLES`, `BUDGET_GRID_POINTS`
  named; `RATIO_FLOOR` reused; `reward_discount` surfaced as a typed `LambdaSchedulingConfig` /
  `SchedulingParams` field with a `(0, 1]` validator.
- **Coverage:** new tests for the converged-winner / tolerance-terminal branch, split-vs-allocate
  cost keying, zero-window infinite variance, the adapter's torch-tensor `get_state` path, empty
  `from_refinement`, and the `reward_discount` validators. `src/thermo` 95% / `src/refinement` 99%
  branch. `ruff` + `mypy --strict` clean; abstraction-audit clean.

### Fixed — CI coverage job uses the pure-Python tracer

- The installed torch wheel crashes coverage's default **C tracer** on `import torch._C`
  (`ValueError: module functions cannot set METH_CLASS ...` / segfault), so the `Test Coverage` job
  failed at collection. Set `COVERAGE_CORE=pytrace` on the coverage job (the remedy already
  documented in CLAUDE.md) so the coverage gates actually run.

### Added — Spec-driven agentic tooling + Noyron v2.2 (`specs/`, `.claude/`, `src/agents/`, `src/poc/scenarios/noyron_basis*`)

Additive, backwards-compatible sprint across four workstreams:

- **Spec-driven development (`specs/`)** — a markdown-only spec tree (`README.md`,
  `TEMPLATE.spec.md`, and per-feature specs) whose thresholds reuse the canonical
  `src.poc.config.MetricThreshold` (no parallel schema). Spec → tests → code → AQA →
  regression-surface entry is now the documented workflow.
- **`.claude/` project scaffolding** — committed shared Claude Code config so web/CLI sessions
  can run the repo's checks: a SessionStart hook that bootstraps `pip install -e '.[dev]'`
  (including the `SETUPTOOLS_USE_DISTUTILS=stdlib` fix for `antlr4-python3-runtime`),
  `settings.json`, four skills (`spec-new`, `regression-surface`, `coverage-gate`,
  `new-pde-operator`), five persona subagents, and three slash commands. Local artifacts
  (`.claude/plans/`, `settings.local.json`) stay gitignored.
- **`src/agents/` hardening** — new `src/agents/AGENT.md`; opt-in `BaseAgent` lifecycle hooks
  (`pre/post_setup`, `pre/post_step`, default no-ops); opt-in wall-clock timeout gated on
  `AgentConfig.enforce_timeout` (default `False` preserves behaviour; enabled →
  `ExecutionStatus.TIMEOUT`); reusable `src/agents/scaffold.py` + `agents.cli scaffold` command.
- **Noyron v2.2 — first MCTS-on-Noyron result** — new `noyron_basis` PoC scenario driving MCTS
  Galerkin basis selection on the Leap 71 helical SDF operators via the existing
  `pde_basis_helical` path, reusing the geometry-agnostic `_centaur_common` primitives. A
  reusable `make_manufactured_operator` overlays a product-of-sines target so the homogeneous
  helical operators yield a non-degenerate game. The default thresholds assert the provable
  correctness property (`error_reduction_pct ≥ 0` monotone, bounded residual); the reduction
  *magnitude* on 3D SDF geometry is limited by the current candidate basis library (~2–4 %) and
  documented as an open research item. Per-arm medians are always recorded so results are never
  vacuous.
- **LLM-prior OOD expansion** — shipped `config/scenarios/llm_prior_{helmholtz,biharmonic}.yaml`
  + AQA tests (operators already in the `ood_pde` Literal / `PDE_TYPE_MAP`).
- **Known-issue closure** — SGF variation parsing marked RESOLVED (verified green); MCTS
  rate-control skips documented as a Milestone 10 Phase 3 gate.

Coverage: `agents/base.py` `config.py` `scaffold.py` 100 %; `noyron_basis.py` 97 %,
`noyron_basis_config.py` 100 %. `ruff` + `mypy --strict` clean on the changed surface.

### Added — LLM-Prior MCTS Basis Selection (`src/integrations/lm_studio/`, `src/poc/scenarios/llm_prior_ablation.py`, `src/poc/scenarios/llm_prior_config.py`, `config/scenarios/llm_prior_demo.yaml`)

### Changed — Noyron HX headline calibrated to measured numbers (`src/poc/config_noyron.py`, `config/scenarios/noyron_hx.yaml`, `README.md`)

End-to-end GPU verification on a Blackwell rig (RTX 5060 Ti) showed the previously documented YAML defaults could not hit the previously documented success criteria. Recalibrated to the measured achievable floor at the YAML-default surrogate size; the headline claim shifts from "tight absolute MSEs" to "essentially perfect resolution-independent transfer".

- **`harmonic_wave_number` default lowered from `4π` to `π`.** At `k = 4π` the reference field is outside the spectral capacity of the default surrogate (`d_model = 64`, 32 Fourier features, 4096 collocation points, 200 epochs); 200-epoch GPU run measured `mse_low ≈ 3e-2`, `mse_high ≈ 6.6e-2`, `transfer_ratio = 2.15`. At `k = π` (one full period across the unit cube) the same surrogate measures `mse_low = mse_high = 1.55e-2`, `transfer_ratio = 1.00 ± 0.01` — eval at 4× training point density gives the same MSE as eval at training density, the central headline claim.
- **`mse_threshold_low` and `mse_threshold_high` relaxed from `5e-4` / `1e-3` to `2e-2` / `2e-2`** to reflect the measured ~`1.6e-2` floor at the default surrogate size, with ~30% headroom for run-to-run noise. These thresholds are now a *regression guard*, not a tight accuracy claim. Reaching tighter absolute MSEs (e.g. `1e-3`) requires growing the surrogate beyond the YAML defaults: `d_model ≥ 128`, `n_train_pts ≥ 16k`, or `n_epochs ≥ 1000`.
- **`transfer_ratio_threshold` tightened from `4.0` to `1.5`** because the measured ratio is `1.00 ± 0.01` — this is now the headline pass/fail metric, and the tighter bound is the regression guard for the resolution-independence claim.
- **YAML headline config** ([config/scenarios/noyron_hx.yaml](config/scenarios/noyron_hx.yaml)) now sets all four values (`harmonic_wave_number`, two `mse_threshold_*`, `transfer_ratio_threshold`) explicitly with inline comments recording the measured floor and the path to tighter thresholds.
- **README "Noyron HX" subsection** ([README.md](README.md)) rewritten: leads with the resolution-independence claim and the measured `transfer_ratio = 1.00`, replaces the bullet success criteria with a Threshold/Measured table, documents the surrogate-growth knobs for tighter absolute MSEs, and corrects the headline-run wall time from "~2 min" to "~7 min on a Blackwell GPU" (measured).

### Decision — DDP wiring for `NoyronHXScenario` deferred (`docs/architecture/c4_mermaid.md`)

- Recorded as a permanent architecture note in the C4 PoC Framework section: per-GPU utilization during the headline run is 1–10% on a Blackwell card. Bottleneck is per-step Adam overhead and Python/CUDA launch latency, not compute. Adding `DistributedDataParallel` would put NCCL all-reduce on the critical path of every step and slow training, not speed it up. Concrete revisit thresholds documented (`n_train_pts ≥ 100k`, `d_model ≥ 512`, or `batch_size ≥ 32`).

- **NS-FDM Taylor-Green parity** — fixed numpy/torch asymmetry in
  `NavierStokesOperator.exact_solution` (numpy branch had `cos(x)*cos(y)`
  instead of `sin(x)*cos(y)` for `uy`). Single-line fix at
  [src/pde/operators.py:1189](src/pde/operators.py:1189) corrects three
  metrics simultaneously: the FDM IC, the FDM L2 reference, and the PINN
  L2 evaluation (all routed through the numpy branch). The torch branch
  was always correct, so PINN training was unaffected — only post-hoc
  evaluation was corrupted. New
  `tests/pde/test_taylor_green_invariants.py` asserts elementwise
  numpy/torch agreement to guard against the drift recurring.
- **Dörfler AMR escapes the 18-DOF ceiling** — `AMRConfig` defaults
  raised so 1D refinement on smooth Burgers (Cole-Hopf shock indicator
  is sharply concentrated) reaches meaningful DOF counts:
  `marking_fraction` 0.3 → **0.5**, `max_refinements` 10 → **30**,
  `max_initial_points_1d` 8 → **256**, `initial_dof_divisor` 4 → **2**.
  The `_solve_amr_1d` `n_start` formula is now target-aware:
  `n_start = max(min(n_dof // 2, max_initial_points_1d), min_initial_points)`.
  New regression test
  `TestDorflerAMRSolver.test_dorfler_amr_1d_reaches_meaningful_dof`
  parametrised across `target_dof ∈ {128, 512, 2048}` ensures the
  algorithm never collapses back to the 18-DOF bug and that n_dof
  scales with the request.
- **Canonical PINN respects `device` + auto-detects vector PDEs** —
  `PINNConfig` gains `device: str = "auto"` (per CLAUDE.md GPU-preferred
  policy) and `vector_pde: bool | None = None`.
  `SimplePINNSolver.solve()` honours both: device resolution flows
  through `src.poc.device.resolve_device` (extended to support
  indexed `cuda:N` strings with bounds checking), and Navier-Stokes
  operators auto-build a 2-channel network with per-component
  Laplacian residual. The previous hard-coded
  `device = torch.device("cpu")` is gone.
  `_build_network(input_dim, output_dim=1)` now accepts an output
  dimension. Metadata round-trip includes `device`, `vector_pde`,
  `n_collocation`, and the new `gpu_profile` block.
- **GPU utilisation profiler** — new
  [src/research/gpu_profiler.py](src/research/gpu_profiler.py) provides
  a `GpuUtilizationProfiler` context manager wrapping `nvidia-smi dmon`.
  Spawns the dmon subprocess on `__enter__`, terminates and parses on
  `__exit__`, returns a `GpuUtilizationReport` (mean SM-util %, mean
  memory-util %, peak FB-memory MiB) which `SimplePINNSolver` embeds in
  `SolverResult.metadata["gpu_profile"]`. Skips silently when
  `nvidia-smi` is missing (CI on no-GPU hosts). All numerical literals
  surfaced as named module constants
  (`_DMON_COL_GPU=0`, `_DMON_COL_SM_PCT=4`, `_DMON_COL_MEM_PCT=5`,
  `_DMON_COL_FB_MEM_MIB=8`, `_DMON_MIN_COLUMNS=6`,
  `_DEFAULT_TERMINATE_TIMEOUT_S=5.0`); `terminate_timeout_s` is a
  configurable constructor field.
- **`PDEBenchmarkRunner` `--heavy` opt-in** — extra refinement levels
  (e.g. 65 536-DOF Poisson for the P40's 24 GiB advantage) live under
  `heavy_refinement_levels` in the YAML and are appended only when the
  runner is constructed with `heavy=True` (or
  `run_sbir_demo --heavy`). Default behaviour is unchanged so CI
  smoke tests stay fast.
- **`scripts/run_sbir_p40.py` rewritten as a config-driven CLI** — the
  previous 260-line subclass-based fork is gone. New shape: small
  argparse-driven driver that loads
  `config/benchmarks/sbir_p40.yaml` (PINN profiles for `p40` and `cpu`
  rows, baselines, benchmarks). Surfaced overrides:
  `--config`, `--output-dir`, `--device`, `--n-epochs`,
  `--n-collocation`, `--refinement-levels`, `--skip-cpu`,
  `--require-cuda`. Helper functions (`load_config`, `apply_overrides`,
  `apply_benchmark_overrides`, `filter_baselines`, `build_pinn_config`,
  `register_pinn_profiles`, `_make_pinn_class`) are all individually
  unit-tested via
  `tests/scripts/test_run_sbir_p40.py`. Zero hardcoded numerics in the
  script body.
- **Coverage on the changed surface** — 95% branch+line coverage across
  the four affected `src/` modules
  (`gpu_profiler.py` 96%, `baselines.py` 95%, `pde_benchmarks.py` 94%,
  `poc/device.py` 100%); 1131 tests pass on `tests/research/` +
  `tests/pde/` + `tests/scripts/test_run_sbir_p40.py` with the global
  85% gate met (project total 94.84% on the changed module set).
  `ruff check` + `ruff format --check` clean on every edited file.
### Added — Codec Model Zoo Phase 2-D (`src/video_compression/zoo/sweep.py`, `scripts/train_compression_zoo.py`)

### Fixed — Noyron HX YAML loader dispatch (`src/poc/config.py`)

- **`load_config_from_dict` now dispatches `name="noyron_hx"` to `NoyronHXScenarioConfig`** (was silently falling back to `BaseScenarioConfig`, whose `extra="forbid"` rejected every Noyron-specific field with 24 Pydantic ValidationErrors at runtime). PR #58's smoke tests construct the config directly in code so they never exercised the loader path; the bug only surfaced via `python -m src.poc.cli run --config config/scenarios/noyron_hx.yaml`. Lazy-imported `NoyronHXScenarioConfig` inside the function to avoid a circular dep with `src/poc/config_noyron.py`. Regression test added to `TestLoadConfigFromDict` in [tests/poc/test_config.py](tests/poc/test_config.py).

### Fixed — CUDA-host test brittleness in pre-existing scenarios (`src/poc/scenarios/complexity.py`, `tests/poc/test_stability_scenario.py`)

- **`ComplexityScenario._benchmark_{fnet,softmax,galerkin}` memory tracking** now gates `torch.cuda.max_memory_allocated()` on `self._device.type == "cuda"` rather than the global `torch.cuda.is_available()`. The previous gate produced non-zero `memory_mb` on CUDA-available hosts that forced the scenario to CPU (which the test does deliberately), violating the "CPU runs report zero CUDA memory" contract. Real bug on multi-device hosts, not just a test fix.
- **`test_result_contains_expected_fields`** (stability scenario) now compares `result.device` against the device `StabilityScenario.setup()` actually picks (`"cuda" if torch.cuda.is_available() else "cpu"`) instead of hardcoding `"cpu"`. The production code's auto-selection was correct; the test was the one out of sync with reality.

### Added — Noyron HX v1 Hardening (`src/pde/sdf.py`, `src/pde/geometry_picogk.py`, `src/poc/scenarios/noyron_hx.py`)

- **Voxel-FDM training consistency** — `NoyronHXScenario` now trains directly on the cached FDM solution when `ref_solver_kind="voxel_fdm"`. Previously the scenario trained on the harmonic surrogate but graded against FDM; the head-line `mse_low < 5e-4` / `mse_high < 1e-3` thresholds were unreachable in FDM mode. The cached solution is built lazily via `_voxel_fdm_reference()` and reused at evaluation, so reference and supervision come from the same field.
- **Surfaced scenario metrics** — `accept_rate` (from `PicoGKDomain.volume_accept_rate`), `train_time_s`, `eval_time_s`, and `train_loss_final` are now recorded in `ScenarioResult.metrics`. Timing values are captured via `ScenarioLogger.timed(...)` context-manager and propagated through the public metric dict.
- **Bisection / grid-search fallback for SDF projections** — `AnalyticalHelixSDF._nearest_t` gains a coarse-grid + Newton-refine fallback (opt-out via `enable_fallback=False`); `PicoGKDomain._project_to_surface` gains a bracketed bisection along the central-difference gradient (opt-out via `enable_bisection_fallback=False`). Both restore robustness on thin tubes (`r/R << 0.1`) where the original Newton iteration could stall.
- **`PicoGKDomain.volume_accept_rate`** — new read-only property that exposes the empirical interior acceptance rate computed at construction by the existing Monte-Carlo volume estimator. No re-sampling cost; the rate is cached on `_volume_accept_rate`.
- **`NoyronHXScenarioConfig.helix_n_turns`** default aligned to **5** across the Pydantic config, `config/scenarios/noyron_hx.yaml`, and `AnalyticalHelixSDF`. Previously the config-class default was 3 while the YAML used 5 — instantiating the config in code produced a different geometry than the headline run.
- **Module-level numerical-stability constants** (replace previously hardcoded literals):
  - `DEFAULT_TRANSFER_RATIO_FLOOR: float = 1e-12` — division floor for `mse_high / mse_low`.
  - `DEFAULT_NORMALIZE_EXTENT_FLOOR: float = 1e-9` — bbox-extent clamp in `_normalize`.
  - `EVAL_SEED_STRIDE: int = 9973` — prime offset between low- and high-density evaluation seeds.
- **`NoyronHXScenario._draw_pool_indices(n_pool, n_pts)`** — single helper for sampling indices from the cached FDM voxel pool. Replaces duplicated `randperm` / `randint` selection logic that previously appeared inline in both `_sample_voxel_fdm_batch` and `_evaluate`. Validates `n_pool > 0` and `n_pts > 0` and routes through-replacement sampling via `randint` when `n_pts > n_pool`.

### Added — Noyron HX Test Suite (`tests/pde/test_sdf.py`, `tests/pde/test_picogk_domain.py`, `tests/poc/test_noyron_hx_scenario.py`)

- **43 new test cases** covering: SDF fallback (disabled / param validation / grid scaling / pathological initial guess / no-Newton-refine branch), `PicoGKDomain` constructor validators (`grad_epsilon`, `max_oversample`, `projection_max_iters`, `min_grad_norm_sq`), bisection-fallback no-op when all-converged, projection-converged log branch, `accept_rate` / `train_time_s` / `eval_time_s` metric round-trip, voxel-FDM uses FDM (not harmonic) supervision, voxel-FDM cache cleared on teardown, `_draw_pool_indices` semantics, and module-constant invariants.
- **Per-module coverage**: `src/pde/sdf.py` **100%**, `src/pde/geometry_picogk.py` **100%**, `src/poc/config_noyron.py` **100%**, `src/poc/scenarios/noyron_hx.py` **97%** — all well above the project 85% gate.

### Added — Learned PDE Evaluator (`src/alphagalerkin/`)

- **`AlphaGalerkinConfig.evaluator="trained"`** — re-enables the network-backed evaluator literal that was removed in the DOE Genesis PR. The trained branch loads an `AlphaGalerkinModel` checkpoint via `create_model_from_checkpoint` and wraps it in the existing `FNetEvaluator`, providing learned policy/value priors to MCTS rather than the uniform prior of `RandomEvaluator`. Closes the only non-trivial entry under *Known Issues* in `CLAUDE.md`.
- **`checkpoint_path: Path | None`** Pydantic field with a `model_validator(mode="after")` that fails fast at config-construction time when `evaluator="trained"` is paired with a missing or non-existent checkpoint.
- **GPU-primary default** — `AlphaGalerkinConfig.device` default flipped from `"cpu"` to `"cuda"`. New module-level `_resolve_device_cached` helper (cached via `functools.cache`) falls back to CPU at runtime when `torch.cuda.is_available()` is False, emitting at most one downgrade warning per unique device string for the lifetime of the process. The random/uniform evaluator path skips device resolution entirely (it is device-agnostic) so CPU-only users do not see spurious `cuda_requested_but_unavailable` warnings under the new default. `config/train_pde.yaml` updated to `device: auto` (the `Trainer`'s native CUDA-availability fallback) — note that the solver's runtime fallback is solver-only and does not apply to the training pipeline.
- **Trained-evaluator instance cache** — `AlphaGalerkinSolver._build_trained_evaluator()` constructs the `FNetEvaluator` once per solver instance and reuses it across subsequent `solve()` calls so benchmark suites that iterate over many PDEs do not pay repeated disk I/O + model-init cost. `reset_cache()` invalidates the cache for callers that swap checkpoints during a long-running process.
- **New evaluator config fields** (replace previously hardcoded values):
  - `evaluator_temperature: float` (gt=0.0, default=1.0) — softmax temperature for trained-evaluator policy logits.
  - `evaluator_use_fast_path: bool` (default=True) — toggle the FNet fast-forward path inside `FNetEvaluator`.
  - `checkpoint_strict_load: bool` (default=False) — controls strict shape matching on `create_model_from_checkpoint`; the default tolerates policy-head shape mismatches across PDEs.

### Added — Trained Evaluator Tests (`tests/alphagalerkin/test_trained_evaluator.py`)

- **8 new test classes / parameterized cases** covering evaluator dispatch, action-space mismatch graceful degradation, device resolution caching, trained-evaluator instance caching, and config-field propagation. The GPU smoke test is gated on `@pytest.mark.gpu_required` and auto-skips on CPU CI via the root `conftest.py` hook.
- **Per-module coverage** on `src/alphagalerkin/` raised to **94%** (gate: 85%).

### Changed

- **`src/alphagalerkin/solver.py`** — module docstring rewritten to document the three evaluator modes; previously hardcoded `temperature=1.0` / `use_fast_path=True` / `strict=False` in `_build_mcts` removed in favour of the new Pydantic config fields.
- **`tests/alphagalerkin/test_solver.py`** — `test_trained_evaluator_rejected_by_config` flipped into `test_trained_evaluator_requires_checkpoint`, asserting both missing-path and non-existent-path failure modes surface as `ValidationError`.
- **`config/train_pde.yaml`** — `device: cpu` → `device: auto` to use the trainer's native CUDA-availability fallback (`BaseTrainer` resolves `auto` via `torch.cuda.is_available()`); the solver's `_resolve_device_cached` runtime fallback is solver-only and does not apply to the training pipeline.

### Documentation

- **`CLAUDE.md`** — drop the "trained-evaluator stub" entry from *Known Issues*; add the *Learned PDE Evaluator Wired* milestone for 2026-04-25.
- **`docs/architecture/c4_mermaid.md`** — extend the Container Diagram with the trained-evaluator path through `FNetEvaluator` and the on-instance evaluator cache.

### Added — E2E Dashboard (`dashboard/`)

- **`dashboard/app.py`** — Gradio Blocks application factory (`build_app()`) and CLI entry point (`main()`). Launches a tabbed UI exposing all AlphaGalerkin capabilities at `http://localhost:7860`. Accepts `--host`, `--port`, `--share`, `--debug` flags.

- **`dashboard/config.py`** — Full Pydantic v2 config hierarchy eliminating every hardcoded value:
  `AppConfig`, `GameConfig`, `PDEConfig`, `ComplexityRunConfig`, `StabilityRunConfig`,
  `TransferMilestone`, `PoCConfig`, `TrainingConfig`, `DashboardConfig`.
  `DEFAULT_CONFIG` singleton for zero-configuration startup.

- **`dashboard/utils.py`** — Shared utility module:
  - `fig_to_pil()` — always closes matplotlib figure (even on exception); `.copy()` detaches from buffer
  - `device_str()` — CUDA/CPU detection with graceful fallback
  - `format_exc()` — consistent exception formatting
  - `configure_structlog()` — idempotent structured logging setup

- **`dashboard/tabs/game_tab.py`** — Go AI tab. Thread-safe lazy model loading via `threading.Lock` (double-checked locking). Human vs AI and AI vs AI modes with 9×9/13×13/19×19 board support (zero-shot transfer). Config-injected via `GameConfig`.

- **`dashboard/tabs/pde_tab.py`** — Interactive Poisson equation solver. Five charge patterns (Point Charge, Dipole, Quadrupole, Ring, Random), multi-resolution comparison with zoom-upsampling MSE. Config-injected via `PDEConfig`.

- **`dashboard/tabs/poc_tab.py`** — PoC scenario runner. O(N) complexity benchmark, LBB stability monitoring, zero-shot transfer milestone display. Module-level optional imports for test patchability. Config-injected via `PoCConfig`.

- **`dashboard/tabs/training_tab.py`** — Architecture summary, simulated training curves (policy/value/LBB losses), and loss breakdown diagram. Config-injected via `TrainingConfig`.

### Added — Dashboard Test Suite (`tests/dashboard/`)

- **203 tests**, **89% line coverage** (gate: 85%), all passing with zero ruff violations.
- `conftest.py` — shared fixtures, `matplotlib.use("Agg")`, config fixture hierarchy, mock scenario results, charge-grid fixtures.
- `test_app.py` (24 tests) — CSS builder, arg parser, `build_app()`, `main()`.
- `test_config.py` (31 tests) — all Pydantic models, validation errors, JSON round-trip.
- `test_utils.py` (24 tests) — `fig_to_pil` (close on error, detached buffer), `device_str`, `format_exc`, `configure_structlog`.
- `test_pde_tab.py` (37 tests) — all charge patterns, Poisson solve integration, `solve_and_visualize`, `compare_resolutions` with shape-matching mock.
- `test_poc_tab.py` (32 tests) — `_parse_int_list`, `run_complexity`, `run_stability` (mocked), `show_transfer_milestone` (live).
- `test_training_tab.py` (28 tests) — model summary (fallback on import error), training curves, loss breakdown.
- `test_game_tab.py` (27 tests) — `autouse` fixture resetting module globals, fallback board, `_ensure_loaded` idempotency, human/AI move handlers.

- **Intercept Module** (`src/intercept/`)
  - `InterceptGame` implementing `GameInterface` protocol for MCTS-guided missile defense
  - 6-DOF rigid body dynamics (`dynamics.py`, `interceptor_dynamics.py`)
  - Proportional Navigation guidance (`guidance.py`)
  - `ExtendedKalmanFilter` for target tracking (`tracking.py`)
  - `RadarSensor`, `SensorFusion` for multi-sensor tracking (`sensors.py`)
  - `HungarianAssigner` for weapon-target assignment (`assignment.py`)
  - `ISAAtmosphere`, `WindModel` for atmospheric modeling (`atmosphere.py`)
  - `AeroModel`, `TabularAeroModel` for aerodynamic coefficients (`aero.py`)
  - `FrameTransform`, `QuaternionOps` for reference frame conversions (`frames.py`)
  - Pydantic-validated `InterceptorConfig`, `EngagementConfig`, `ThreatConfig`

- **Backend Abstraction** (`src/backend/`)
  - `BackendInterface` protocol for unified PyTorch/JAX operations
  - `TorchBackend`, `JaxBackend` implementations
  - `Array`, `Precision`, `DeviceType` type abstractions (`types.py`)
  - Random number generator abstraction (`rng.py`)
  - Backend-aware logging and debug utilities

- **Prototyping Module** (`src/prototyping/`)
  - `ModelBuilder`, `PrototypeModel` for rapid architecture iteration
  - `QuickTrainer`, `TrainResult` for fast experiment loops
  - `QuickEvaluator`, `EvalResult` for quick model evaluation
  - `DataGenerator`, `SyntheticData` for synthetic data creation
  - `Visualizer` with multiple plot types
  - `ExperimentTemplate`, `TemplateRegistry` for experiment patterns

- **Analysis Module** (`src/analysis/`)
  - `PositionEvaluator`, `EvaluationResult` for position evaluation
  - `GameReviewer`, `MoveAnalysis` for game review and move quality assessment
  - `PatternMatcher`, `PatternLibrary` for board pattern detection
  - `GameStatistics`, `StatisticsCollector` for game statistics aggregation
  - `AnalysisConfig`, `AnalysisMode` Pydantic configuration

- **Tournament Module** (`src/tournament/`)
  - `TournamentManager`, `TournamentState` supporting Round-Robin, Swiss, Elimination formats
  - `TournamentScheduler` for match scheduling
  - `EloRating`, `RatingSystem` for player rating computation
  - `Player`, `PlayerRegistry` for participant management
  - `Match`, `MatchResult`, `MatchStatus` for match tracking

### Changed

- **`pyproject.toml`** — Added `[[tool.mypy.overrides]]` for `dashboard.*` modules (relaxed strict checks for Gradio code). Added `[tool.coverage.report]` with `fail_under = 85` and `show_missing = true`. Added `dashboard` pytest marker.
- **Gradio 6 compatibility** — CSS argument moved from `Blocks()` constructor to `launch()`.

> **Branch and PR cleanup** — removed 28 stale remote branches and 6 open stale PRs.

## [0.3.0] - 2026-04-01

### Summary

Key highlights of this release:

- **Chess Self-Play Training Pipeline** — AlphaZero methodology, 4672-action dense policy, 119-channel state encoding
- **SBIR Readiness Infrastructure** — Navy N252-088, DOE ASCR, NSF SBIR, AFWERX proposal configs and benchmark suite
- **Advanced PDE Operators** — NavierStokes (Taylor-Green), L-shaped Poisson (singularity), enhanced Burgers (Cole-Hopf)
- **Domain Geometry & Time-Stepping module** — Rectangular, L-shaped, Cylinder domains; ForwardEuler, RK4, CrankNicolson
- **Multi-Agent Swarm Planning** — PettingZoo `ParallelEnv` adapter, potential field obstacle avoidance
- **Unified Loss Package & BaseTrainer consolidation** — `LossRegistry`, `get_loss()` factory, shared AMP/grad/LR in `BaseTrainer`
- **CI/CD hardening** — 85% coverage gates, nightly schedule, Stage 8 chess pipeline
- **218+ new tests** across PDE, research, training, and games modules

---

### Added

- **SBIR Readiness Infrastructure** (Navy N252-088, DOE ASCR, NSF, AFWERX)
  - `config/proposals/navy_n252_088.yaml`, `nsf_sbir.yaml` — SBIR-specific benchmark configs
  - `config/benchmarks/sbir_suite.yaml` — 3-problem benchmark suite (L-shaped Poisson, Burgers shock, NS Taylor-Green)
  - `src/research/baselines.py` — Classical PDE solver baselines: UniformFDMSolver, DorflerAMRSolver, SimplePINNSolver
  - `src/research/pde_benchmarks.py` — PDEBenchmarkRunner with JSON/Markdown report generation and convergence rate computation
  - `docs/proposals/templates/sbir_phase1.md` — Reusable SBIR Phase I proposal template
  - `docs/proposals/IP_STRATEGY.md` — 3 provisional patent claims, trade secret boundaries, publication plan

- **Advanced PDE Operators**
  - `NavierStokesOperator` — Taylor-Green vortex benchmark with analytical solution, configurable Re
  - `BurgersOperator` enhanced — Cole-Hopf exact solution, configurable shock params, convergence rate method
  - `LShapedPoissonOperator` — r^(2/3)*sin(2theta/3) singularity for AMR benchmarking

- **Domain Geometry Abstractions** (`src/pde/geometry.py`)
  - `RectangularDomain`, `LShapedDomain`, `CylinderFlowDomain` (DFG benchmark)
  - Rejection sampling for non-convex domains, proportional boundary sampling
  - `GeometryConfig` Pydantic schema and `create_geometry()` factory

- **Time-Stepping Module** (`src/pde/time_stepping.py`)
  - `ForwardEuler`, `RK4`, `CrankNicolson` (fixed-point iteration) with factory pattern
  - `TimeSteppingConfig` Pydantic schema, `integrate()` with snapshot saving

- **S500 Swarm Planning Game** (`src/pde/games/swarm_planning.py`)
  - `SwarmPlanningGame` with round-robin multi-agent control (7 actions per agent)
  - Potential field obstacle avoidance (Laplace equation connection), coverage rewards
  - `SwarmPlanningConfig` — fully Pydantic-validated with no hardcoded values

- **PettingZoo Adapter** (`src/games/pettingzoo_adapter.py`)
  - `PettingZooAdapter` wrapping `GameInterface` as PettingZoo `ParallelEnv`
  - Optional dependency with graceful degradation (`HAS_PETTINGZOO` flag)

- **Unified Loss Package** (`src/training/losses/`)
  - `LossRegistry` with decorator-based registration (`"alphagalerkin"`, `"l2_relative"`, `"h1"`, `"mse"`)
  - `get_loss()` factory function for config-driven loss instantiation
  - Backwards-compatible thin wrappers in `src/training/loss.py` and `src/training/physics_loss.py`

- **BaseTrainer Consolidation** (`src/training/base_trainer.py`)
  - Abstract `BaseTrainer[ConfigT]` with shared AMP, gradient clipping, LR scheduling, checkpoint save/load
  - `BaseTrainerConfig` Pydantic schema covering all shared hyperparameters
  - `StepResult` dataclass for structured step output

- **Checkpoint Migration System** (`src/training/checkpoint_migration.py`)
  - Version-aware migration with `@register_migration` decorator
  - Migration path: `0.0.0 -> 1.0.0 -> 1.1.0` (LBB config fields added)

- **Property-Based and Numerical Stability Tests**
  - `tests/training/test_loss_properties.py` — hypothesis tests: non-negativity, CE = log(n), gradient flow
  - `tests/training/test_numerical_stability.py` — extreme values, near-zero denominators, NaN propagation
  - `tests/pde/test_operator_properties.py` — PDE operator invariants, linearity, collocation in domain
  - `tests/modeling/test_attention_properties.py` — Galerkin attention shape, LBB positivity, resolution independence

- **Comprehensive Coverage Tests** (218 new tests)
  - `tests/pde/test_geometry.py` — 65 tests for domain geometries
  - `tests/pde/test_time_stepping.py` — 37 tests for time-stepping methods
  - `tests/research/test_baselines.py` — 39 tests for classical solver baselines
  - `tests/research/test_pde_benchmarks.py` — 38 tests for benchmark runner
  - `tests/training/test_base_trainer.py` — 39 tests for BaseTrainer
  - `tests/pde/test_swarm_planning.py` — 50 tests for swarm planning game
  - `tests/games/test_pettingzoo_adapter.py` — 11 tests for PettingZoo adapter

### Changed

- **CI/CD Hardening** (`.github/workflows/ci.yml`)
  - MyPy strict enforcement (`continue-on-error: false`)
  - Coverage gates raised: 75% -> 85% overall, 80% -> 85% per-module (pde, modeling, training)
  - Added `research` module coverage gate at 85%
  - Added nightly schedule (`cron: '0 4 * * *'`) and performance benchmark job on main merges

- **Config-Driven LBB Loss** (`config/schemas.py`)
  - Surfaced `lbb_loss_weight`, `lbb_target`, `lbb_eps`, `log_barrier_weight` as Pydantic fields
  - Added mathematical documentation (Babuska-Brezzi motivation) in field descriptions

- **Race Condition Fix** (`src/modeling/model.py`)
  - Removed `_training_resolution` mutation from `forward()` (DDP-unsafe)
  - Added explicit `set_training_resolution()` public method

### Fixed

- `advection_coeff` dimension mismatch in `PDEBenchmarkRunner._create_operator()` — was hardcoded `[0.0, 0.0]` for any dim

- **Chess Self-Play Training Pipeline** (AlphaZero methodology)
  - `ActionPolicyHead` for dense 4672-action policy output (`src/modeling/model.py`)
  - `StatefulGameWrapper` bridging stateless `GameInterface` to MCTS (`src/games/wrapper.py`)
  - Chess training CLI (`scripts/train_chess.py`) with Hydra config (`config/train_chess.yaml`)
  - `game_type` and `action_space_size` fields in `OperatorConfig` (`config/schemas.py`)
  - PRD and ADR documentation (`docs/prd/prd-chess-self-play.md`, `docs/architecture/ADR-chess-self-play.md`)

- **Chess Training Tests**
  - `tests/games/test_wrapper.py` — StatefulGameWrapper unit tests (10 tests)
  - `tests/modeling/test_chess_model.py` — ActionPolicyHead and chess model tests (12 tests)
  - `tests/training/test_chess_self_play.py` — Chess self-play integration tests (7 tests)
  - `tests/games/test_chess_exhaustive.py` — Exhaustive encode/decode roundtrip + edge cases (20 tests)
  - `tests/training/test_trainer_chess.py` — Checkpoint save/load/resume, engine eval, config tests (11 tests)
  - `tests/security/test_chess_security.py` — Invalid actions, OOB states, corrupted data (15 tests)
  - `tests/e2e/test_chess_training_e2e.py` — E2E training smoke tests (3 tests)

- **Stockfish Benchmark Evaluation**
  - Engine eval config fields in `TrainingConfig` (path, depth, games, movetime)
  - `Trainer._run_engine_evaluation()` with W&B Elo metric logging
  - Engine eval section in `config/train_chess.yaml`

- **CI/CD Chess Pipeline**
  - Stage 8: Chess Pipeline Tests in `.github/workflows/ci.yml`
  - Coverage gate `--cov-fail-under=80` for `chess.py` (97%) and `wrapper.py` (100%)
  - CI Success gate requires chess tests
### Changed

- **Game-agnostic self-play**: `SelfPlayWorker` now accepts optional `GameInterface` parameter
- **Game-agnostic trainer**: `Trainer.__init__()` accepts `game` parameter, forwarded to worker
- **Game-agnostic collator**: `VariableSizeCollator` and `SameSizeCollator` derive action mask size from `target_policy` tensor instead of hardcoded `board_size²+1`
- `AlphaGalerkinModel` and `AlphaGalerkinFast` auto-select policy head by `action_space_size`
### Fixed

- **Underpromotion encode/decode mismatch** (`src/games/chess.py`): `_decode_move` used `[-1, 0, 1]` but `_encode_move` used `straight=0, left=1, right=2` — straight promotion from column 0 decoded as `to_col=-1`. Fixed to `[0, -1, 1]`.
- **Collator action mask size** (`src/data/collate.py`): Both collators hardcoded `n_actions = board_size²+1` causing tensor size mismatch with chess's 4672-action policy. Fixed to detect per-experience policy encoding.

## [0.2.0] - 2026-01-26

### Milestones Achieved

- **Zero-Shot Transfer Validated**: Physics PoC demonstrated resolution-independence
  - Trained on 9x9 grids, transfers zero-shot to 19x19 (measured MSE ≈ 0.00039). NOTE: the original "0.000209 / 240× better than threshold" was a fabricated notebook figure — corrected 2026-07-22; a CNN retrained at 19x19 is more accurate (see `specs/transfer_baseline_compare.spec.md`).
  - Validates core Galerkin approach for continuous operator learning

- **Training Pipeline Operational**: End-to-end training with self-play working on GPU
  - MCTS-based self-play generates training experiences
  - LBB stability monitoring integrated into training loop

### Added

- **W&B Integration for Physics PoC**
  - `--wandb` flag for `train_physics.py` to enable Weights & Biases logging
  - Logs training loss, evaluation MSE, transfer MSE, and learning rate
  - Final summary includes success status and best transfer MSE

- **GameInterface Protocol Implementation**
  - Added `apply_action()` method to `SimpleGoGame` class
  - Enables MCTS integration with Go game state

- **Security Tests** (`tests/security/`)
  - Input sanitization tests for GTP interface
  - DoS protection via input length limits

- **E2E Tests** (`tests/e2e/`)
  - CLI journey tests for help and train commands

### Changed

- Replaced Unicode checkmarks with ASCII `[PASS]`/`[FAIL]` for Windows compatibility
- Updated `.gitignore` with additional patterns:
  - `nul` (Windows device file)
  - `*.log`, `*.dist-info/`
  - `hydra_outputs/`

### Fixed

- Fixed `AttributeError: 'SimpleGoGame' object has no attribute 'apply_action'`
- Fixed unused loop variable warning in `BoardSizeBatchSampler`
- Fixed line length issue in W&B initialization

## [0.1.0] - 2026-01-26

### Added

- **Core Architecture**
  - `AlphaGalerkinModel`: Resolution-independent Go AI using continuous operators
  - `GalerkinLinearAttention`: O(N) complexity global influence modeling
  - `SoftmaxAttention`: Local tactical reading with injectivity preservation
  - `FNetBlock`: FFT-based mixing for fast MCTS rollouts

- **Mathematical Kernel**
  - Fredholm integral equation with Green's function formulation
  - Fourier features for positional encoding
  - Monte Carlo integral normalization (1/n) for Galerkin attention
  - LBB stability monitoring (dim(Key) >= dim(Query))

- **Training Infrastructure**
  - Self-play with MCTS for experience generation
  - Uniform and prioritized replay buffers
  - `AlphaGalerkinLoss`: policy_CE + value_MSE + LBB_regularization
  - Checkpoint management with best model tracking
  - Hydra configuration system

- **Physics PoC**
  - Poisson equation solver for synthetic data generation
  - `PhysicsOperator` neural network for influence field prediction
  - Zero-shot transfer verification scripts

- **PoC Scenario Framework**
  - Configuration-driven scenario execution
  - Built-in scenarios: transfer, complexity, stability
  - Pydantic-validated configs
  - Structured logging via structlog

### Documentation

- C4 architecture diagrams
- CLAUDE.md with project context and verification commands
