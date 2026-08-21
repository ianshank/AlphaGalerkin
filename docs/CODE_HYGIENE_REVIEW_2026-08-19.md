# Code Hygiene & Correctness Review — 2026-08-19

> **What this is:** a hands-on, execution-verified senior-engineering pass across
> `src/mcts/`, `src/pde/`, `src/refinement/`, `src/integrations/`, and `src/data/` —
> real commands run, real bugs found and fixed, real coverage gaps closed with new
> tests. This is a companion to `docs/CODE_HYGIENE_AUDIT.md` (2026-08-14, backlog/
> triage-oriented) and `docs/NEXT_STEPS_REVIEW_2026-08-18.md` (strategic/tiered); this
> doc records what was actually *done* in one pass, not just recommended.
>
> **Method:** four of this repo's own specialist subagents (`mcts-engineer`,
> `pde-solver`, `integration-engineer`, `sqe`) independently audited disjoint,
> non-overlapping subsystems in parallel — each ran its own regression surface,
> wrote and verified new tests, and fixed narrow, well-tested bugs in-scope while
> flagging (not unilaterally resolving) ambiguous design questions. A fifth agent
> (`reviewer`) then ran an adversarial pass over the combined diff, specifically
> trying to break the four highest-stakes correctness claims. I independently
> verified lint/format/type-check/test results across the aggregate before writing
> this document; no number below is taken from a subagent's self-report without an
> independent re-run.

## Headline: four real bugs fixed, one root-cause fix landed, one new critical defect surfaced

1. **MCTS crashed on a terminal-at-root game state.** `mcts.get_action()` raised an
   unhandled `ValueError` (temperature ≠ 0) or an unhelpful failure (temperature = 0)
   whenever `search()` was called on a game that was already terminal at the root —
   e.g. a degenerate PDE config whose initial state already satisfies the termination
   condition. Now guarded with a clear, actionable error. (`src/mcts/search.py`)
2. **A NaN/Inf value from a broken evaluator silently corrupted the search tree,
   with zero observability.** Neither the single-simulation nor the batch path
   checked evaluator output for finiteness before backing it up. **Detection was
   added** (a shared `_check_finite_evaluation` helper on both paths, emitting a
   structured warning) — the adversarial review pass confirmed this is accurately
   scoped as detection-only, not a value fix: `_check_finite_evaluation`'s own
   docstring states neither field is sanitized, and `MCTSNode.backup()` still adds
   `total_value` unconditionally with no finiteness guard, so a NaN still poisons
   every ancestor's value — it is now just visible in the logs while doing so,
   rather than silent. (`src/mcts/search.py`, `evaluator.py`)
3. **NaN from a diverging PDE solve resolved to the *best possible* MCTS leaf value.**
   `EncodedValueEvaluator.evaluate` (`src/pde/games/lshape_amr.py`) clamped its output
   via `max(-1.0, min(1.0, value))`; because NaN comparisons are always `False` in
   Python, `value=nan` resolved through the clamp to exactly `+1.0` — the single best
   score a leaf can have — which would have actively steered MCTS search *toward* a
   computational failure rather than away from it. Fixed with an explicit
   `np.isfinite` guard (neutral `0.0` fallback + warning).
4. **A broken CUDA driver crashed LM Studio preflight instead of degrading
   gracefully.** `torch.cuda.mem_get_info()` failures were already caught per-device,
   but `torch.cuda.is_available()`/`device_count()` themselves raising (a present-
   but-broken driver, distinct from "no GPU") propagated a raw `RuntimeError`,
   contradicting the function's own documented "skip cleanly" contract. Fixed with the
   same try/except pattern already used for the per-device probe.
   (`src/integrations/lm_studio/preflight.py`)

Plus one narrow, mechanical fix: a retry-branch log/sleep ordering inconsistency in
`src/integrations/lm_studio/client.py` (one of three retry branches slept before
logging, delaying the diagnostic event relative to the other two).

**Root-cause fix landed**: the Tier-0 P0-1 Burgers OOD-reward defect
(`docs/CODE_HYGIENE_AUDIT.md:576-614`, previously scoped in
`docs/NEXT_STEPS_REVIEW_2026-08-18.md` item 1 as "2-4 engineer-days, not yet done")
is now fixed. `BurgersOperator.__init__` (`src/pde/operators.py`) now checks
`"is_time_dependent" in config.model_fields_set` before overriding the class-level
default, so an unset config keeps `is_time_dependent = True` (and a real
`exact_solution()`) while an explicit `True` or `False` is still honored exactly as
before — `tests/pde/test_operators.py::test_steady_returns_none` (the test that
proved the naive "just honor the class default" fix would have broken something
real) passes unmodified.

**Critical finding, surfaced by fixing the above — live today, not hypothetical.**
Making `BurgersOperator.exact_solution()` reachable at the default `t=0` exposed a
**pre-existing, separate defect** in the Cole-Hopf approximation itself. At `t=0`,
the truncated-series formula evaluates `phi ≈ -0.5` (negative), which the
`clamp(min=1e-10)` guard floors up — producing solution magnitudes of order
**1e10–1e13** (large but finite, not NaN/Inf, so nothing currently catches it).

This was initially reported by `pde-solver` with appropriate caution ("must be
resolved before anyone re-derives thresholds"), but the adversarial review pass
traced the actual call chain and found it is **not a future concern — it is live on
the default configuration of the shipped scenario right now**:
`src/poc/scenarios/llm_prior_config.py`'s default `ood_pde` is `"burgers"`, exactly
matching `config/scenarios/llm_prior_demo.yaml` — CLAUDE.md's own documented
headline GPU command. Directly measured: `build_pde_operator("burgers")` +
`build_basis_game(...)` → `get_initial_state().error_estimate = 4.29e12`, with the
same tensor serving as both the RMS baseline *and* the literal least-squares
regression target for every basis-fit action in the game.

**The regression test added alongside the fix was itself silently vacuous**: it
only asserted `torch.isfinite(...).all()`, which trivially passes on a
~3e10-magnitude nonsense value — exactly the "test doesn't test what it claims"
failure mode this whole pass was watching for, caught by the adversarial pass one
level up rather than by the original author. **Closed in this pass**: added
`test_cole_hopf_t0_magnitude_is_a_known_defect_not_a_correctness_claim` right next
to it, which pins the magnitude as a documented, tracked defect (`u.abs().max() >
1e6` — the opposite assertion direction from what a "this works" test would assert)
so a future reader can't mistake "isfinite passes" for "this is fine," and so a real
fix has an explicit regression target to flip.

**Practical consequence, not yet acted on**: the next real run of
`python -m src.poc.cli run --config config/scenarios/llm_prior_demo.yaml` will
compute `ood_llm_residual`/`ood_trained_residual` against this numerically
meaningless ~1e12 "ground truth" instead of either the old flat-zero or a
physically meaningful solution — very likely flipping the documented
`ood_llm_residual ≤ 1e-2` threshold to FAIL for reasons that have nothing to do with
LLM quality. **This needs an explicit decision before that command is next run for
real** — options include patching the Cole-Hopf floor, having the basis-selection
game avoid querying at exactly `t=0`, or gating the headline rerun until resolved.
None of those were chosen unilaterally here: this is genuinely Cole-Hopf-math work
(same category of decision as the explicitly out-of-scope Heat/AdvectionDiffusion
operators), not a hygiene-pass fix. This revises
`docs/NEXT_STEPS_REVIEW_2026-08-18.md` item 1's stated next step and should be read
before that item's "re-derive thresholds" instruction is acted on.

## Documentation drift found (independent of the four subsystem audits)

**`CLAUDE.md:115` currently states a false claim.** Its banner says the
`video_compression` subsystem "was deleted from the repository" and that
`src/video_compression/**`, `scripts/benchmark_codec.py`,
`scripts/train_compression_zoo*.py`, and `config/video_compression/**` "no longer
exist." Verified directly: **four of the five named paths exist and are under active
development** (the Self-Hosted Transcoder Phase 0-2D milestones, already documented
*elsewhere* in the same file, reinstated it after the 2026-07-22 cut).
`tests/support/cut_modules.py`'s own docstring already correctly documents the
reinstatement — only the CLAUDE.md:115 banner itself is stale. This is a distinct
failure class from the numeric-fabrication incidents CLAUDE.md already tracks (a
stale *existence* claim, not a stale *number*) and was independently confirmed by two
different findings in this pass:

- `mypy --strict` surfaces **28 previously-undocumented real type errors**, all in
  `src/video_compression/` (operator-type mismatches, `Tensor`-vs-`Module` confusion,
  ndarray dtype-parameter errors) — not mentioned in CLAUDE.md's mypy discussion or
  any CI comment, because the file claiming the package doesn't exist makes it
  invisible to the tooling that would normally track this.
- `src/video_compression` has **no per-module coverage gate** anywhere in
  `.github/workflows/ci.yml`'s coverage job or CLAUDE.md's Regression Surface table,
  despite running in CI's plain `pytest tests/` — it is structurally excluded from
  the repo's own quality-gate discipline.

This is flagged for a follow-up fix (correct the CLAUDE.md:115 banner, decide whether
`video_compression` gets a coverage gate) — not fixed in this pass, since it's a
documentation/governance call, not a code bug in the four audited subsystems.

## Coverage improvements (measured, not estimated)

| Module/package | Before | After | Gate |
|---|---|---|---|
| `src/data/physics_dataset.py` | 23% branch | **100%** branch | — |
| `src/data` (package) | 79.81% | **98.14%** | 77% |
| `src/refinement` | 96% | **100%** | 85% |
| `src/mcts` | 96.29% | **96.95%** | 90% |
| `src/pde` | (baseline not re-measured pre-fix) | **92.72%** | 85% |
| `src/integrations/lm_studio` | 94.77% | **95.62%** | 85% |

All gates verified passing with the exact CLAUDE.md-documented commands
(`COVERAGE_CORE=pytrace` prefix where required).

## Hardcoded values — fixed

- **`src/mcts/{search,node,gumbel,evaluator}.py`**: 7 call sites hardcoded the
  literal `1.0` instead of using the already-defined `DEFAULT_TEMPERATURE` constant
  (which had zero consumers before this fix). Pure value-preserving substitution,
  verified via full regression before/after.

## Hardcoded values — found, flagged for follow-up (not fixed; each needs a config-shape decision beyond this pass's scope)

- `src/pde/operators.py`: Cole-Hopf `n_terms=50` and the `1e-10` clamp epsilon are
  duplicated across tensor/numpy branches, not named constants or config fields —
  the same epsilon implicated in the Cole-Hopf t=0 finding above.
- `src/pde/operators.py:903,924,1031`: `sigma = 0.1 * np.mean(self.domain_size)` (the
  synthetic Gaussian-pulse width fraction) is duplicated verbatim across three
  operators, not a config field.
- `src/pde/games/basis_selection.py:235-236`: RBF candidate-basis centers are sampled
  via `rng.uniform(0, 1)`, hardcoding a `[0,1]` domain assumption regardless of the
  operator's actual `domain_min`/`domain_max` — silently wrong for any non-unit-square
  domain (e.g. `LShapedPoissonOperator`'s `[-1,1]²`).
- `src/pde/games/basis_selection.py:445` and `src/pde/games/mesh_refinement.py:749`:
  `cost = 1.0` per action decrements `budget_remaining` (seeded from
  `computational_budget`, default `1e6`), completely decoupled from the config's own
  `cost_per_dof` field used elsewhere — at default scale, `BUDGET_EXHAUSTED` is
  practically unreachable.
- `src/pde/games/basis_selection.py:453`: a hardcoded, non-scale-normalized
  EXPLORING/REFINING phase threshold — notably, `PDEGame.get_phase()`
  (`src/pde/game.py:524-556`) **already implements this correctly** (config-driven,
  scale-normalized), but neither `BasisSelectionGame` nor `MeshRefinementGame` calls
  it; each hand-rolls its own (worse, in basis_selection's case) inline version.
  Diagnostic-only impact today (only read by serialization, not reward/termination).
- `src/pde/games/mesh_refinement.py:257`: a hardcoded `level < 2` hp-refinement
  switchover, unlike its sibling `max_refinement_level`/`max_polynomial_degree`
  config fields.
- `src/pde/games/swarm_planning.py:440`: obstacle-distance floor hardcoded to `0.1`
  despite a docstring claiming it uses the config's own `obstacle_radius` field.

`src/pde/games/lshape_amr.py` and all of `src/refinement/*.py` were checked and are
already clean (named constants / Pydantic fields throughout) — no findings there.

## Design ambiguities flagged for a human decision (correctly not unilaterally resolved)

- **`fallback_to_uniform_on_parse_error` doesn't cover permanent SDK errors.** A
  non-retryable SDK failure (auth revoked mid-run, model unloaded) surfaces as the
  bare `LMStudioError` parent, which isn't caught by the evaluator's fallback
  except-tuple — so it always propagates uncaught rather than degrading to
  uniform-random, unlike parse/mismatch/connection failures. A one-line broadening to
  `except LMStudioError` would make this symmetric, but the field's name specifically
  says "parse_error," so this is a real product-behavior call, not a bug. Regression
  tests were added locking in the *current* contract either way.
- **`src/mcts/constants.py` (and its `src/physics/`, `src/training/` siblings) is
  dead re-export scaffolding** — nothing imports from any of the three; every real
  consumer imports `src.constants` directly. Resolving this is a cross-cutting,
  three-package architectural call, out of scope for a single-subsystem audit.
  **Resolved (2026-08-19, follow-up hygiene wave):** re-verified zero-consumer status
  repo-wide (`src/`, `tests/`, `dashboard/`) and deleted all three files. No dedicated
  test files existed for them (`tests/mcts/`, `tests/physics/`, `tests/training/` had no
  `test_constants.py`), so no test deletions were needed. `src/constants.py` is
  unaffected and remains the canonical module every consumer already imported from.
- **`MCTSNode.select_child` raises a misleadingly-labeled error when every child's
  Q-value is NaN** (`"node has no children to select from"` when it does have
  children — NaN comparisons are always `False`, so no child ever wins the
  selection). The real fix is catching NaN at its source (the evaluator-output guard
  added in this pass reduces how often this can happen, but doesn't structurally
  prevent it). Locked in via a test documenting the current message rather than
  silently changed.

## Edge cases closed with new tests (representative highlights, not exhaustive — see full diff)

- MCTS: empty/single legal actions at the full-tree level (evaluator-level was
  already covered); `c_puct`/temperature boundary values (0.0, very large);
  `reward_discount` just-above-zero; `BatchMCTS` homogeneous all-terminal/
  all-non-terminal batches (only the interleaved case existed before).
- PDE/refinement: repeated/nested clone isolation (`clone().clone()` — relevant
  because MCTS clones along every simulation path, so a depth->1 tree walk clones a
  clone, and no existing test went beyond one level); zero-DOF terminal states
  reachable from a real, unmodified `get_initial_state()` (not just a synthetically
  mutated state) for all three PDE games, each additionally proven not to crash a
  real `MCTS.get_action()` micro-run; zero-measure/degenerate domain rejection at the
  Pydantic validation boundary (`>=` not just `>`); `RefinementGameAdapter.error_
  reduction`'s zero-division guard (previously unexercised, now `src/refinement` is
  at 100% branch coverage).
- Integrations: a real multi-failure retry sequence (transient-recovers /
  permanent-fails-fast / transient-exhausts) asserting exact call counts and sleep
  values, **mutation-tested** by temporarily forcing `_retryable` to always return
  `True` to confirm the new tests actually fail under the bug they guard against, then
  reverting; structured log event field assertions (`lm_studio_retry`/`lm_studio_
  call`) that previously took a fixture but never read it.
- Data: `PhysicsDataset`'s `cache=False` on-demand path, the `_compute_stats()` /
  `get_stats()` normalize-without-cache silent-no-op gotcha, `n_samples=0`'s
  previously-undocumented `ValueError` crash under the default `normalize=True`, and
  a dtype-casting asymmetry (`coords` is never cast to float32, unlike input/output).

## Untested-code findings (report-only, not fixed — outside the four subsystems' scope)

- `src/games/go.py`: illegal-move rejection, White-stone-counting/White-territory
  scoring branches, and `get_winner`'s mid-game guard are all never exercised by any
  test.
- `src/games/chess.py`: queenside castling is claimed tested by a docstring but only
  the kingside path and queenside's *initial flag* are actually driven end-to-end;
  threefold-repetition is claimed tested by two file docstrings but no test actually
  drives the repetition counter.
- `src/games/sgf/converter.py`'s `create_analysis_tree` has zero test references.
- `src/deployment/quantize.py::CalibrationDataReader` — pure Python/numpy, no
  optional-dependency gating needed, yet untested (unlike the rest of the package,
  which is legitimately `importorskip`-gated on absent `onnx`/`onnxruntime`).
- `src/distributed/worker.py` — the entire file (`SelfPlayWorker`,
  `SelfPlayCoordinator`) has zero test references, despite not requiring a live
  `torch.distributed` process group to construct or exercise (confirmed by reading
  `__init__`). Sharpest example: `_serialize_experiences`/`_deserialize_experiences`
  is plain `pickle.dumps`/`loads` with no distributed dependency at all.

## Skills, hooks, and loops — findings

**Existing inventory confirmed healthy**: 9 skills, 5 subagents, 4 slash commands
under `.claude/`. The `new-pde-operator` skill convention referenced by two specs
(`llm_prior_ood.spec.md`, `verified_error_certificate.spec.md`) is verified real and
correctly wired — not a dangling reference.

**Verified, concrete tooling bugs**:
- `Makefile`'s `test-stoch` target has **silently drifted** from what CI/CLAUDE.md
  actually gate — it omits 3 of 4 required `--include=` paths and 5 of 6 required
  test paths, so `make test-stoch` reports a different (likely inflated) number than
  the real gate.
- `make demo` is **currently broken** — it references a scenario name
  (`transfer_darcy_to_poisson`) that was never registered; it only ever existed as an
  illustrative placeholder in a migration doc, copy-pasted into the Makefile without
  being run.
- `make lint` is narrower than CI (missing `dashboard/ scripts/ config/ conftest.py
  deploy_space.py`), and `make check`/`pre-pr` never invoke any coverage gate at all
  — a green `make check` asserts nothing about the 85% global gate or ~20 per-module
  gates the PR checklist requires.
- The PR template's checklist item "`pre-commit run --all-files` is green (ruff,
  ruff-format, yamllint, commitizen…)" is slightly inaccurate — `commitizen` only
  runs at the `commit-msg` git-hook stage, never on `--all-files`.

**SessionStart hook verified present, correct, and doing more than documented**
(also runs `pre-commit install` and prints tool-version banners, not just the
documented `pip install -e '.[dev]'`).

**Fabricated/stale-claim lint-hook feasibility**: a general "suspiciously precise
number" regex was already prototyped and abandoned elsewhere in this repo (105 false
positives across 21 files, per CLAUDE.md's own Next-Steps table) — not tractable. But
a **narrow, mechanical existence-claim checker is tractable and would have caught the
CLAUDE.md:115 finding above**: extend `scripts/check_doc_links.py`'s existing
backtick-path-stripping primitives to flag a path near phrases like "no longer
exist"/"REMOVED"/"deleted" whose existence-on-disk contradicts the claimed direction.
Recommended as a follow-up, not built in this pass.

**Loop audit** (excluding the four already-audited subsystems): `src/training/
trainer.py:602`'s `_fill_buffer` while-loop has no iteration cap or wall-clock bound
— if self-play ever nets zero new experiences per call, it hot-loops indefinitely
(logged, but not bounded). **No SIGINT/SIGTERM handling exists anywhere in the
training stack** — the only exception handlers are `except Exception`, which does not
catch `KeyboardInterrupt`, so a killed run has no emergency-checkpoint path and zero
test coverage of that scenario. `checkpoint_migration.py:87`'s migration-path search
loop has no runtime guard that each step strictly advances version — currently
dormant (today's 2 registrations are both correctly forward) but would spin silently
forever under a future mis-registration. Everything else checked (self-play move
bounds, distributed comms, checkpoint atomicity, `BaseAgent`'s opt-in timeout
enforcement) was verified already-correct.

## Adversarial review pass — what it broke, what it confirmed

A fifth agent (`reviewer`) independently re-derived and tried to break the four
highest-stakes correctness claims above, with direct code execution rather than
trusting descriptions:

- **Burgers sentinel fix (item 1 above): verified sound**, including two specific
  attack angles that could have broken it — confirmed Pydantic's `model_fields_set`
  correctly distinguishes "explicit `False`" from "never set" (not just "differs
  from default"), and confirmed YAML-sourced (`model_validate`) construction tracks
  fields-set identically to direct kwargs, so the fix behaves the same regardless of
  config-loading path.
- **Cole-Hopf t=0 finding: confirmed and escalated** — see above; this is the one
  place the adversarial pass found the original finding understated the severity,
  and it directly caught a vacuous test that the fix-author's own verification had
  missed.
- **MCTS NaN/Inf detection (item 2 above): confirmed correctly scoped**, with the
  precision correction now reflected above (detection, not sanitization).
- **`EncodedValueEvaluator` NaN-to-neutral fix (item 3 above): verified sound**,
  including direct reproduction of the underlying Python quirk being fixed
  (`max(-1.0, min(1.0, float('nan'))) == 1.0`) and confirmation that `±inf` was
  already handled correctly pre-fix (only NaN misbehaves under `min`/`max`), so the
  fix is precisely targeted rather than overclaimed.
- **Scope-boundary integrity: verified sound** — no two agents touched the same
  file; `src/constants.py` itself (source of `DEFAULT_TEMPERATURE`) was untouched by
  this diff, so the 7 substitutions carry zero collision risk.
- **Test-quality spot checks across all four subsystems: verified sound** — every
  spot-checked test constructs real objects and asserts real behavior; the new
  `tests/data/test_physics_dataset.py` suite was independently re-run (21/21 pass,
  100% branch coverage independently re-measured, matching the claimed figures).

## Verification (independently re-run, not taken from subagent self-reports)

```
ruff check <all touched src+tests>           → All checks passed!
ruff format --check <all touched src+tests>  → clean
mypy --strict --ignore-missing-imports <all touched src>  → Success: no issues found in 8 source files

pytest tests/mcts/ tests/pde/ tests/refinement/ tests/integrations/ tests/data/ \
       tests/poc/test_llm_prior_ablation_config.py tests/poc/test_llm_prior_ablation_scenario.py \
       -m "not gpu_required"
  → 1505 passed, 11 skipped, 8 deselected, 0 failed  (20.6s)

pytest tests/training/test_losses_physics.py tests/training/test_physics_integration.py \
       tests/training/test_trainer_physics.py
  → 101 passed, 0 failed

python -m scripts.audit_abstractions src/mcts src/refinement src/pde --fail-on-missing
  → OK: every abstract method / protocol member has a call site.
```

No file was touched by more than one agent — confirmed via `git diff --stat`
(23 modified + 1 new file, cleanly partitioned by subsystem).

## Environment note (affects anyone reproducing these commands)

Bare `mypy`/`pytest` on `$PATH` in this sandbox resolve to isolated `uv tool` shims
missing `pydantic`/`hypothesis`/`pytest-cov` — they fail immediately with confusing
import errors unrelated to any real code issue. Use `python3 -m mypy ...` /
`python3 -m pytest ...` instead; all numbers in this document were produced that way.

## What this pass deliberately did not do

- Did not fix the Cole-Hopf t=0 defect itself (only added a regression-lock test
  pinning it as a known, tracked defect — see above), the Heat/AdvectionDiffusion
  P0-1 slices, the `fallback_to_uniform_on_parse_error` scope question, the
  `constants.py` dead re-export, the CLAUDE.md:115 stale-existence-claim, the
  Makefile drift/breakage, or any of the report-only untested-code findings — each
  is flagged above with enough specificity to act on, but each also needs either a
  design decision or a larger-than-hygiene-pass change.
- Did not touch `src/video_compression/`'s 28 real mypy errors — real, newly
  surfaced tech debt, but a subsystem none of the four dispatched agents were scoped
  to and too large for an unplanned addition to this pass.

---

# Round 2 — same day, broader sweep

> **What changed since the section above:** round 1 covered 5 of ~25 `src/` packages
> and deliberately left a list of things flagged-but-unfixed. Round 2 executed that
> list plus the packages round 1 never reached, using 8 parallel agent waves on
> disjoint file scopes. **Most of round 1's "did not do" list above is now done** —
> the `constants.py` dead re-export, the `CLAUDE.md:115` stale-existence claim, the
> Makefile drift, and `src/video_compression`'s mypy debt all landed here.
>
> **Verification note:** three agent waves were killed mid-run by an API spend limit.
> All three had already finished their edits and died during final reporting, not
> mid-edit — confirmed by re-running every gate independently rather than trusting
> any agent's self-report. Every number below was produced by a command run after all
> waves stopped.

## What round 2 fixed

**Real defects**

1. **Thread-safety bug** — `src/prototyping/templates.py`'s `TemplateRegistry.__new__`
   copied `src/templates/registry.py::BaseRegistry`'s double-checked-locking singleton
   pattern but **omitted the lock entirely**, so concurrent first-access could
   construct more than one "singleton." Fixed to mirror the working pattern exactly.
   Honest caveat on the test: a raw N-thread timing race could *not* be made to fail
   reliably against the pre-fix code (the racy window is a few bytecodes, well under
   CPython's GIL switch granularity), so a second, deterministic lock-presence test
   was added alongside it — that one does fail against the bug.
2. **Unbounded self-play loop** — `Trainer._fill_buffer` had no iteration or
   wall-clock bound; if self-play ever netted zero usable experiences per call it
   would re-invoke full MCTS generation forever. Now bounded by a new, documented
   `TrainingConfig.max_buffer_fill_iterations` field (no hardcoded literal) and
   raises a typed `BufferFillError` with an actionable message rather than silently
   training on an under-filled buffer.
3. **`src/distributed/worker.py` `games_completed` over-count** — found by the
   first-ever test of that file, **reported not fixed** (the wave that found it was
   test-only by design): `SelfPlayWorker.generate_batch` adds the *requested* game
   count unconditionally, ignoring the `_should_stop` early `break`, so calling
   `stop()` before a batch yields `experiences_generated == 0` but
   `games_completed == 5`. Characterized by a test that pins current behavior rather
   than silently normalizing it.

**Hardcoded values eliminated** (all value-preserving unless noted)

- `basis_selection.py` RBF candidate centers were sampled from a hardcoded `[0,1]`
  unit square regardless of the operator's real domain — wrong for any non-unit
  domain (e.g. `LShapedPoissonOperator`'s `[-1,1]²`, where centers landed partly
  outside the domain). Now sampled from the operator's actual `domain_min`/`domain_max`,
  with explicit 1D handling.
- **Behavior-changing, deliberately:** `basis_selection.py` / `mesh_refinement.py`
  decremented the budget by a flat `cost = 1.0` while the *reward* path in the same
  files already computed `cost_per_dof * dof_added`. The budget path now mirrors the
  reward path. Note the direction: at the default `cost_per_dof=0.01` the old flat
  cost exhausted the budget ~100× faster than the cost the reward was accounting for
  — reusing the existing correct pattern (rather than naively substituting
  `cost_per_dof` alone) was the load-bearing detail here.
- `basis_selection.py` hand-rolled a hardcoded, non-scale-normalized EXPLORING/
  REFINING phase threshold while `PDEGame.get_phase()` already implemented it
  correctly (config-driven, scale-normalized). Now delegates to the base class.
- `mesh_refinement.py`'s `level < 2` h-vs-p switchover became a typed
  `hp_switchover_level` config field; `swarm_planning.py`'s obstacle-distance floor
  and `operators.py`'s duplicated Cole-Hopf `n_terms`/clamp-epsilon and
  `sigma = 0.1 * domain_size` became named constants.
- 7 call sites across `src/mcts/` hardcoded `1.0` instead of the already-defined
  `DEFAULT_TEMPERATURE` (which had zero consumers).

**Dead and duplicated code removed**

- `src/mcts/constants.py`, `src/physics/constants.py`, `src/training/constants.py` —
  three re-export modules added in a prior sprint to fix hardcoded-value hygiene, but
  with **zero consumers** (every real call site imports flat `src.constants`
  directly). Deleted after verifying zero-consumer status per file. A migration guide
  that actively recommended these as the "preferred v0.4+" import path was corrected.
- `BaseTrainer.evaluate()` and both concrete stubs (`Trainer.evaluate`,
  `DistributedTrainer.evaluate`) — an abstract method with no call site anywhere,
  each subclass stubbing it with `NotImplementedError`. Deleted, with an explanatory
  comment left at the removal site.
- `FNetMixingLayer` was declared twice with identical bodies; `benchmark_fnet.py` now
  imports the canonical `src.modeling.fnet` version. Verified first that no committed
  doc cites a specific number this benchmark produces.
- `select_child`'s error message claimed the node "has no children" when the real
  cause is all-NaN child scores (NaN comparisons are always `False`). Message
  corrected; selection semantics deliberately unchanged.

**Type safety** — `mypy --strict` on `src/`: **31 → 8 errors**. The 23 fixed were all
in `src/video_compression/` (missing `register_buffer` companion annotations, a
systemic `np.ndarray[np.int32, ...]` shape/dtype type-parameter typo, list/dict
annotations, a return type needing narrowing, a stale ignore). The 8 remaining are
the 5 `codec/codec.py` errors that need real interface design (a factory returning
bare `nn.Module` where a Protocol over the real variants is needed) plus 3
pre-existing torch-version-dependent `unused-ignore`s CI already documents as
accepted.

**Coverage and tooling gates**

- `src/video_compression` had **no coverage gate at all** despite 933 tests running
  in CI — because `pyproject.toml` still omits it from `--cov=src` (a leftover from
  when the package was believed retired). Now gated at 83 against a measured 85.43%,
  using the same inline-coveragerc technique `phase2-zoo-validation.yml` already uses
  for the same collision. Charter gates register updated.
- `src/distributed` coverage rose 68.91% → 82.34% (`worker.py` specifically 22% → 99%).
- `make test-stoch` had silently drifted to 1 of 4 required `--include=` paths and 1
  of 6 test paths — it reported a different, inflated number than the gate it claims
  to mirror. `make demo` was outright broken (referenced a scenario name that was
  never registered). `make lint` was narrower than CI. `make check`/`pre-pr` never
  invoked any coverage gate. All fixed and each verified by actually running it.
- `CLAUDE.md:115`'s claim that `video_compression` "no longer exists" was false for 4
  of the 5 paths it named; corrected, along with a second copy of the same false claim
  elsewhere in the file. Only `config/perf/*` is genuinely gone.

**Tests added** for previously-unexercised reachable code: Go illegal-move rejection,
`get_result()`'s White/draw branches and `get_winner()`; Chess queenside-castling
*execution* and threefold-repetition (both claimed by docstrings but never actually
driven); `CalibrationDataReader`; and all of `src/distributed/worker.py`.

## Verification (all re-run independently after every wave stopped)

```
ruff check  src/ tests/ dashboard/ scripts/ config/ conftest.py deploy_space.py  → clean
ruff format --check  (same paths)                                                → 895 files formatted
mypy src/ --strict --ignore-missing-imports                                      → 8 errors (from 31)
audit_abstractions src/mcts src/refinement src/pde --fail-on-missing             → clean

pytest tests/pde/ tests/refinement/                                → 1045 passed
pytest tests/training/ tests/distributed/                          → 1167 passed,  3 skipped
pytest tests/video_compression/                                    →  962 passed, 24 skipped
pytest tests/mcts tests/games tests/deployment tests/prototyping \
       tests/experiments tests/modeling tests/physics tests/data \
       tests/docs                                                  → 1844 passed, 33 skipped
                                                                     ────────────────────────
                                                                     ~5,018 passed, 0 failed
```

## Still open after round 2

- **The Cole-Hopf t=0 defect from round 1 is unchanged** and remains the most
  urgent item in this document — see the round-1 section above. Round 2 did not
  touch it (it is PDE-math work, not hygiene).
- `src/distributed/worker.py`'s `games_completed` over-count (new, above).
- A **pre-existing, unrelated** failing test surfaced while wiring `make check`:
  `tests/security/test_checkpoint_safety.py::test_checkpoint_path_validation`
  (`UnpicklingError: unpickling stack underflow`). Not caused by any round-2 change
  and deliberately left untouched, but it currently halts the `make check`
  prerequisite chain.
- `codec/codec.py`'s 5 remaining mypy errors (need a Protocol/Union over the entropy
  model variants).
- Deferred by explicit decision: SIGINT/SIGTERM handling for the training stack
  (`self_play.py` uses `torch.multiprocessing.Pool`, so a flag-check handler would
  not cover an interrupt arriving mid-self-play — needs a design spike, not a quick
  add); the `B10` package deletions; `tests/demos/` CI wiring (verified passing
  151/151 but invisible to every CI job); `src/backend`'s dual exclusion from both
  coverage and the abstraction gate.

## Process note

Round 1's adversarial `reviewer` agent pass caught a real severity understatement
(the Cole-Hopf finding) and a vacuous test. Round 2 substituted a **targeted
self-review of the highest-risk diff** (the training control-flow changes) plus the
full empirical re-verification above, because the spend limit that killed three
waves made another full reviewer-agent pass a poor trade. That is the one quality
step consciously traded away this round, recorded here rather than left implicit.

## Round-2 adversarial review — and what it caught

The round-2 section above initially shipped *without* the adversarial `reviewer`
pass (an API spend limit had just killed three agent waves, and a targeted
self-review was substituted). That pass was run afterward, on the pushed diff. It
was worth it — it found a **real regression introduced by the round-2 work itself**,
which the ~5,018 green tests did not catch.

**Confirmed defect, now fixed: `Path()` coercion corrupted GCS checkpoint URIs.**
One of the 23 "behavior-preserving" `mypy` annotation fixes wrapped
`artifacts.checkpoint_path` in `Path(...)` at `zoo_trainer.py:364`. But
`EntryArtifacts.checkpoint_path` is `Path | str` by design — the filesystem backend
returns a `Path`, and `GCSZooStorage` returns a `gs://` URI *string* (GCS objects
have no filesystem path, and `storage.py`'s own docstring says so). `Path()`
collapses the double slash, so `gs://bucket/…` silently became
`gs:/bucket/…` — a value `parse_gcs_uri` rejects outright and
`scripts/train_compression_zoo_entry.py` writes straight into `metrics.json`. Every
existing test uses the filesystem backend, where `Path(Path)` is a no-op, which is
exactly why it stayed green. Fixed by widening `ZooTrainingReport.checkpoint_path`
to `Path | str` to match, and passing the value through unchanged. A regression test
was added and **mutation-tested** (it fails when the `Path()` wrap is reintroduced).

**Three comments that were factually wrong — corrected.** All three claimed more
than the code delivered, which is the specific failure mode this whole exercise
exists to prevent:

- The 1D RBF branch claimed `BasisFunction.evaluate` "never reads `center_y`" for 1D
  coords. It does: `evaluate` substitutes `y = 0`, so `r_sq` becomes
  `(x - center_x)² + center_y²` and any nonzero `center_y` multiplies the entire
  basis column by a constant `exp(-center_y²/2σ²)`. At σ=0.1 a `center_y` of 0.7
  scales the column by ~2e-11, which `lstsq` then discards as rank-deficient —
  silently shrinking the effective basis on every 1D problem (Burgers included).
  **Fixed properly rather than just re-worded**: `center_y` is now pinned to 0 for 1D
  domains, making the candidate a true 1D Gaussian. The draw is still taken so the
  RNG stream stays aligned and seeded 2D results remain bit-identical.
- The phase-delegation comment (and its test docstring) claimed the change bought
  "config-driven, scale-normalized" phase detection. `PDEGame.get_phase` normalizes
  by `self._initial_error`, which `BasisSelectionGame` never sets — so it falls back
  to `1.0` and the comparison is exactly as absolute as the literal it replaced. The
  real win is one code path instead of two. Corrected, along with the two genuine
  behavior deltas delegation introduces (`INITIAL` for the first few steps; *any*
  non-converged terminal reported as `BUDGET_EXHAUSTED`).
- The mesh-refinement cost comment claimed a flat "~100× slower budget drain." The
  ratio is dimension-dependent and can invert: 2D at p=1 gives cost 0.12 (cheaper
  than the old 1.0), but 3D at p=3 gives 4.48 (more expensive). Corrected, and the
  lost strict-monotonicity of `budget_remaining` under coarsening is now noted
  explicitly (`max_steps`, not budget, is the real episode bound).

**Also fixed:** `CLAUDE.md`'s 2026-08-16 milestone still listed the three deleted
`constants.py` modules as a delivered capability — the same class of stale claim
round 2 corrected in three *other* files, missed in the file that matters most.
`make format` was still narrower than the widened `make lint`, so
`make format && make lint` could fail on a file `format` never touched.

**Verified sound by the same pass** (checked, not assumed): `dof_added` has no
off-by-one at either budget site; `BUDGET_EXHAUSTED` was already unreachable at every
shipped config *before* the change, so nothing regressed; `budget_remaining` reaches
no committed artifact; the RBF change is bit-identical for the default unit square;
the buffer-fill counter is unconditional and its bound is unreachable-by-accident at
all shipped configs; all three `constants.py` deletions and the `evaluate()` removal
are genuinely consumer-free (including dynamic/string imports and `dashboard/`,
`hf_space/`, `scripts/`); the two `FNetMixingLayer` copies were byte-identical apart
from inert jaxtyping annotations; and the new coverage gate measures 85.43% under
CI's exact conditions with `.coveragerc.*` already gitignored.

**Open, tracked, not fixed here:** `hp_switchover_level` has a lower bound but no
upper bound or cross-validation against `max_refinement_level`;
`POTENTIAL_FIELD_MIN_DISTANCE` is a domain-scale length masquerading as a module
constant and belongs on the config; and `tests/distributed/test_worker.py`
deliberately pins the `games_completed` over-count, so whoever fixes that bug must
update the test rather than read its failure as a regression.
