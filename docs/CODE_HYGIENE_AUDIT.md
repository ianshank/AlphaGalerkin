# Code Hygiene & Modularity Audit — 2026-08

> Prose/historical material excluded from the MkDocs site (see `mkdocs.yml`
> `exclude_docs`) — browse this on GitHub. It is not part of the API
> reference or the project charter; where it disagrees with
> [`openspec/specs/project-charter/spec.md`](../openspec/specs/project-charter/spec.md),
> the charter wins.

## 1. Executive summary

A structural, hygiene, and tests/CI/tooling scan of `src/` (280 files,
~95.5k LOC as of this audit — `find src -name '*.py' | xargs wc -l | tail -1`)
plus `tests/`, `scripts/`, `config/`, and CI. `hf_space/` (the HuggingFace
deploy bundle) is out of scope — it is a known, already-drifted mirror of
`src/`, tracked by `tests/hf_space/test_mirror_guard.py`, and is its own
follow-up rather than a hygiene item here.

Three headline findings drove this PR's priorities:

1. **`mypy --strict` was reported as "enforced nowhere"** (CI step
   `continue-on-error: true`, pre-commit hook `stages: [manual]`) and every
   exploration pass ranked fixing that as the top priority. Measuring it
   (`mypy src/ --strict --ignore-missing-imports`, 2026-08) found only **3
   errors, all `unused-ignore`** — stale `# type: ignore` comments whose
   underlying diagnostics no longer fire under the pinned toolchain. The
   enforcement gap is real; the error volume was not. Both are addressed
   below at their actual scale (§4 commit 2 fixes the 3; §5 B6 reframes the
   backlog item around gate determinism, not a "ratchet").
2. **The codebase is unusually clean** on markers, exception handling, and
   logging discipline: zero TODO/FIXME/HACK backlog outside a code
   generator's template strings, zero bare `except:`, 149 modules on
   structlog (`grep -rl structlog src/ --include=*.py`) vs. 3 legitimate
   `logging.basicConfig` bootstraps in `src/` (6 including `scripts/`), and 64
   well-scoped `except ImportError` optional-dependency gates. The debt is
   concentrated in **enforcement gaps** (gates that exist but don't run),
   **duplicated boilerplate** (the same lifecycle/config/registry pattern
   reimplemented per-scenario instead of shared), and **layering drift**
   (a `poc` ↔ `research` import cycle papered over by 122 in-function
   imports).
3. **Reproducibility hazard**: a per-seed derivation constant
   (`_SEED_PRIME_STRIDE`) exists under the identical private name in six
   modules with two different values (1009 in four, 7919 in two). This PR
   deduplicates the *arithmetic* without touching the *values*, because
   unifying the values would change derived seeds and invalidate results
   already committed to `config/baselines/*.json`. The value decision is
   deliberately deferred (§5 B9).

This PR lands 8 low-risk quick wins (§4) and documents 20 larger backlog
items (§5, B1–B20) that were evaluated and intentionally not attempted here —
several because a first pass classified them as safe and a peer review
subsequently found evidence they weren't (§6 explains each reversal).

## 2. Method & scope

Three independent passes — structural/modularity, code-level hygiene, and
tests/CI/tooling — searched `src/`, `tests/`, `scripts/`, `config/`, and
`.github/workflows/` for file:line evidence. A design pass then verified
every candidate quick win against the live repo (reproducing counts,
checking test coverage, tracing imports), and an adversarial peer review
re-verified the design pass's own claims, overturning several before
implementation started (§6). Every quick win below was implemented and its
guarding test suite run green before commit; commit SHAs are on the PR.

Numeric claims in this document state the command used to measure them so
they can be reproduced rather than trusted. Per the charter's evidence
standard (`specs/transfer_baseline_compare.spec.md` documents the
2026-07-22 correction of a fabricated benchmark figure), this audit does
not restate that retracted number or the retracted blanket
no-MCTS-with-Galerkin-basis-selection claim — see the charter and
`docs/business/proposals/PRIOR_ART_REVIEW.md` for the corrected framing.

## 3. Findings by theme

### 3.1 God modules and duplicated scenario boilerplate

Nine files carry either a single outsized class or an unrelated multi-class
grab-bag (`wc -l`, 2026-08):

| File | LOC | Shape |
|---|---|---|
| `src/pde/operators.py` | 1841 | 10 PDE-family classes in one flat namespace; `src/pde/registry.py` already has an unused `@register_pde_operator` decorator that would let this split cleanly |
| `src/training/trainer.py` | 1476 | One 1304-line `Trainer` class that re-overrides 5 methods `BaseTrainer` already provides |
| `src/research/baselines.py` | 1460 | Config schemas + solver implementations + registry in one file; the sibling `src/research/extra_solvers/*.py` already demonstrates the intended split-file + `SOLVER_REGISTRY` pattern |
| `src/games/chess.py` | 1242 | One 1114-line `ChessGame` class mixing move generation, action encoding, tensor/symmetry, and rules/termination |
| `src/pde/games/mesh_refinement.py` | 1117 | A pure quadtree data structure (`Mesh`, ~313 lines) and the MCTS game (~700 lines) in one file |
| `src/training/losses/physics.py` | 819 | 6 registered losses; siblings in the same package are already one-loss-per-file |
| `src/agents/config.py` | 703 | 16 config classes for 14 sibling modules — every module in `src/agents/` reaches back into this one hub |

The three `*_compare` scenario families (`lshape_amr_compare`,
`transfer_baseline_compare`, `stochastic_galerkin_compare`) are ~90%
identical across `src/poc/scenarios/` and `src/research/`: byte-identical
`teardown`/`_record_metrics`/`_write_artifacts`/`setup` lifecycle methods,
field-for-field Config→Params transcription (15–29 fields copied
one-to-one per scenario), the same name-lock validator copy-pasted into 6
config modules, and `export_csv`/`export_plot`/`MultiSeed*Comparison`
triplicated in `src/research/` with identical lazy-matplotlib preambles.
`src/poc/scenarios/_centaur_common.py` already extracts several shared
primitives for a *different* scenario family (llm_prior/scaling_law/noyron)
— proof the pattern works — but the `*_compare` triplet never adopted the
equivalent extraction. This PR's commit `refactor(seeding)` fixed one small
instance of exactly this gap (a duplicated `_median`/`median_of` helper);
the full `CompareScenarioBase` extraction is backlog B2.

### 3.2 Rejected internal standards

`src/templates/` provides a thread-safe singleton registry
(`create_registry`), a Pydantic config base (`BaseModuleConfig`), and
structured-logging helpers — used by 8 call sites. Seven other registries
exist that duplicate its singleton machinery instead of using it, three of
them (`ScenarioRegistry`, `GameRegistry`, `TemplateRegistry`) copying the
exact double-checked-locking pattern verbatim (`TemplateRegistry` drops the
lock entirely — a strictly worse copy). `src/poc/config.py::BaseScenarioConfig`
independently re-implements `BaseModuleConfig`'s `compute_hash()` rather
than subclassing it. `src/poc/logging.py` re-implements three helpers
(`log_timing`, `log_call`, `DebugContext`) that already exist in
`src/templates/logging.py`. None of this is touched by the quick wins in
this PR — see backlog B3/B5.

### 3.3 Layering: the `poc` ↔ `research` import cycle

`src/research/baselines.py` imports `src.poc.device` at module level;
`src/poc/scenarios/{lshape_amr,transfer_baseline,stochastic_galerkin}_compare.py`
import `src.research.*` twice each (once under `TYPE_CHECKING`, once
deferred inside a method body, with a comment explaining the import is
deferred specifically to avoid the cycle). Repo-wide there are roughly 101
in-function `from src.` imports — most of them break-the-cycle workarounds
rather than genuine laziness. `src/data/dataset.py` also imports
`src.training` at module level, a low-level package depending on a
high-level one. None of this is touched here; backlog B1 proposes promoting
`src/poc/device.py` to a neutral `src/device.py` (mirroring the existing
`src/seeding.py` precedent this PR's `derive_seeds` helper now also
follows) as the first cut at the cycle.

### 3.4 Enforcement gaps

- **mypy**: see §1. The CI step at `.github/workflows/ci.yml` (`Run MyPy
  type checking`) is `continue-on-error: true`; the pre-commit hook is
  `stages: [manual]`. This PR fixes the measured 3 errors (commit `types:`)
  but leaves the gate non-blocking — the documented reason (torch-version-
  dependent `unused-ignore` diagnostics across environments) is a
  determinism problem, not a volume problem, and needs a policy decision
  before it can be flipped safely (backlog B6).
- **Dead CI job**: the `benchmark` job ran `pytest tests/ -m "benchmark" ...
  || true` against zero matching tests (`benchmark` was never a registered
  pytest marker) and swallowed its own exit code. Removed in this PR.
- **CI/pre-commit lint-scope divergence**: pre-commit's ruff hooks had no
  `files:` filter (so they linted `scripts/`, `config/`, and root-level
  modules), but the CI lint job only checked `src/ tests/ dashboard/`.
  Verified live drift before this PR (`ruff check scripts/ config/
  conftest.py deploy_space.py`): 5 real errors CI could not see. This PR
  extends CI's scope to match and adds `exclude: ^hf_space/` to pre-commit
  so both directions agree (previously only CI excluded it).
- **CLAUDE.md Regression Surface table vs. CI**: only the openspec charter's
  11-row gate register is drift-guarded
  (`tests/docs/test_charter_alignment.py::test_documented_gates_are_enforced_in_ci`).
  CLAUDE.md's much larger Regression Surface table is not — at least the
  Noyron HX/basis and L-shape AMR coverage-gate rows are documented but
  absent from `ci.yml`, and two rows document a dotted `--cov=module.path`
  form that a comment in `ci.yml` itself says collides with the torch C
  extension. Not touched by this PR; backlog B8.

### 3.5 Reproducibility hazard: seed-stride divergence

See §1 item 3 and the `refactor(seeding)` commit in this PR.

### 3.6 Test hygiene

`tests/poc/` (37 files, no shared `conftest.py`) had ~9 modules each
hand-roll an autouse `ScenarioRegistry().clear()` fixture under 2 different
names and at least 3 different semantics (clear-only; clear + purge
`sys.modules` so `@scenario` decorators re-fire; clear + re-register for an
identity check), none restoring what they cleared. This PR added
`tests/poc/conftest.py` with one additive snapshot/restore fixture that
wraps the whole package without changing any local fixture's behavior —
`test_charter_alignment.py`'s own docstring documents the exact
order-dependence this works around (it currently reads
`ScenarioRegistry().list_scenarios()` in a subprocess specifically to avoid
it). Pruning the now-partially-redundant local fixtures is deferred
(backlog B16): two of them also purge `sys.modules`, which a save/restore
fixture does not reproduce, and a safe prune needs per-file analysis.

### 3.7 Docs & config

12 of 26 `src/` packages have an `AGENT.md` (`ls src/*/AGENT.md | wc -l`
vs. `ls -d src/*/ | grep -v __pycache__ | wc -l`, 2026-08) despite the root `AGENT.md` describing
per-module documentation as universal. CLAUDE.md's own Next Steps table
under-reported this gap, naming only 2 packages
(`src/refinement`, `src/alphagalerkin`); this PR corrects that row to name
all 14 (§4 commit 9). The largest undocumented package is `src/research`
(≈9.9k LOC). A grep for `config/scenarios/stochastic_galerkin_compare_demo.yaml`
found zero literal references and it was deleted in an earlier commit in
this PR — CI's `Unit Tests (Fast)` job then failed on all three Python
versions: `tests/poc/test_stochastic_galerkin_compare_config.py` builds the
path via an f-string (`f"stochastic_galerkin_compare_{basename}.yaml"` over
`basename in ["ci", "demo"]`), which a literal grep cannot see. The file was
restored byte-identical and the deletion dropped from this PR's quick wins;
see §6. Several hydra-addressable training configs (`train_5hr.yaml`,
`train_experiment.yaml`, `config/presets/*`) have zero *textual* references
but are reachable via `python -m scripts.train --config-name=...` and are
mirrored into `hf_space/config/` — deleting them was evaluated and rejected
as a quick win (§6) in favor of flagging them in backlog B11.

## 4. Quick wins delivered in this PR

| Commit | Change | Guarding surface run before commit |
|---|---|---|
| `chore(lint)` | Drop removed ruff rules `ANN101`/`ANN102`; add `RUF100` to select (autofixed 71 stale `noqa`); fix the 5 live errors the extended scope surfaced; extend CI + pre-commit to the same file scope in both directions | `ruff check`/`ruff format --check` over the full extended scope; `pytest tests/docs/` |
| `types` | Fix the 3 measured `unused-ignore` errors; `mypy src/ --strict` now exits 0 | `mypy src/ --strict --ignore-missing-imports`; `pytest tests/training/test_base_trainer.py tests/research/test_baselines.py tests/research/test_ns_baseline.py` |
| `test(poc)` | New `tests/poc/conftest.py` — a **structlog** global-config save/restore fixture. A `ScenarioRegistry` snapshot/restore fixture was also attempted here and **reverted before merge**: it made the end state strictly worse (see §6). | `pytest tests/poc`; the order-stress trio (`test_complexity_scenario.py test_registry.py test_cli_commands.py`); `pytest tests/docs/ tests/regression/test_related_work_guard.py tests/hf_space/` |
| `chore(ci)` | Delete the dead `benchmark` job; add `--strict-markers`; delete 6 redundant marker registrations (5 in `tests/conftest.py`, 1 in `tests/dashboard/conftest.py`) | `pytest --collect-only`; `pytest tests/dashboard/`; `check_doc_links.py` |
| `refactor(seeding)` | `derive_seeds(base_seed, n_seeds, stride)` in `src/seeding.py`; 4 config modules + `seed_sweep.py` delegate to it; **strides unchanged**. `stochastic_galerkin_compare_config.py` does **not** adopt it — see the CI-caught correction below | The config-test files asserting seed derivation |
| `refactor(poc)` | `resolve_device("auto", ...)` in `stability.py`/`transfer.py`/`complexity.py` (byte-equivalent to the prior inline expression); `_median = median_of` shim in `llm_prior_ablation.py` | LLM-prior mocked-CPU surface + `test_centaur_regression.py`; `test_stability_scenario.py test_complexity_scenario.py test_runner.py test_device.py` |
| `refactor(constants)` | Wire 2 exact concept+value twins (`DEFAULT_LBB_THRESHOLD`, `DEFAULT_DROPOUT`) to their call sites; delete 2 twinless dead constants + their test assertions | `tests/test_constants.py tests/modeling/` |
| `chore(observability+config)` | Log 4 genuinely-silent exception swallows; add a `viz` optional-dependency extra; remove the dead `doc8` pre-commit hook | Mesh-refinement/preflight/CLI tests; SBIR P40 surface |
| `docs(audit)` | This document + `mkdocs.yml` exclude entry + CHANGELOG + CLAUDE.md milestone bullet + AGENT.md row correction | `check_doc_links.py`; `pytest tests/docs/` |

Every commit above ran `mypy src/ --strict --ignore-missing-imports`
(clean throughout) and the full extended ruff scope before landing.

**A CI-caught correction, made after the commits above landed and after
this document was first written**: the `chore(observability+config)`
commit's YAML deletion and the `refactor(seeding)` commit's adoption in
`stochastic_galerkin_compare_config.py` both turned out to be wrong, and
both were caught by GitHub Actions' `Unit Tests (Fast)` job failing on all
three Python versions within minutes of the PR opening — see §6 for the
two root causes and the fixes. Left in the git history rather than
squashed, since a hygiene PR silently self-correcting its own history
would undercut the point.

### 4.1 Follow-up PR: B17 (dead abstraction) + B18 (CI gate)

Landed after the merge of the PR above, on a branch restarted from the
updated default. Two backlog items, closed together because B18's gate
cannot be turned on for `src/pde` until B17 clears it.

**B17 — `PDEGame.get_result` deleted, not wired.** The audit deferred this
item precisely because "wire it or delete it" is a contract change that
wants evidence rather than taste. The evidence came out one-sided:

* **No caller, and the shape fits none of the five that exist.** Five
  independent episode-terminal paths (`src/pde/trainer.py::_run_episode`,
  `src/alphagalerkin/solver.py::solve`,
  `_centaur_common.run_basis_selection_cell`,
  `lshape_amr_compare.run_mcts_arm`, `src/agents/solver.py`) each build
  their own result object, and **every one needs a field `PDEResult` does
  not carry** — `actions`, `solution`/`grid_points`/`wall_time_seconds`,
  `rollouts_used`, `n_solves`. Wiring `get_result` could not have replaced
  any of them; it could only have run *in addition*, to hand back a struct
  the caller then destructures for 2–3 values it already holds.
* **Six of its seventeen fields had no reader anywhere in `src/`** —
  `compute_efficiency`, `dof_efficiency`, `error_reduction_rate`,
  `best_error`, `average_error`, `residual_norm`.
* **Everything real was already reachable.** Error norms via
  `compute_exact_error` (a sibling abstract method that *does* have a call
  site, `solver.py:423`); the trajectory via `PDEGameAdapter.error_history`
  (consumed at `solver.py` as `METADATA_KEY_ERROR_HISTORY`); the rest off
  `PDEState`.
* **The sibling precedent argues the other way, not the same way.** F1 wired
  `get_reward` up because the engine consumes its *semantics* — `MCTS._simulate`
  accumulates `R + γ^d·V(leaf)`, and wiring it moved committed numbers
  (the L-shape median ratio 0.8896 → 0.9605). `get_result` changes no
  algorithm, no search, no number. It is reporting, not mechanism.
* **It documented a flow that never executed.** `docs/architecture/pde_game_c4.md`
  showed `MCTS->>PDEGame: get_result(final_state, history)`. MCTS has never
  called it. That sequence diagram is corrected in this PR (the terminal
  calls belong to `AlphaGalerkinSolver`, not MCTS).

**What was kept.** Deleting `get_result` would have orphaned
`lshape_amr._termination_reason` — a genuinely useful classifier whose only
production consumer was the dead method — and the same ladder was inlined a
third and fourth time inside the two other `get_result` overrides. So the
ladder was promoted to `PDEGame.termination_reason`, with the one
game-specific rung behind a `_capacity_reason` hook (basis count for
`basis_selection`; DOF with `>` for `mesh_refinement`, `>=` for
`lshape_amr`, each matching its own `is_terminal`). It is **concrete, not
abstract**, so no existing subclass breaks.

It now has a real consumer: `AlphaGalerkinSolver` records it under
`METADATA_KEY_TERMINATION_REASON`, which previously read `"is_terminal"` —
a label that collapsed converged / max_dof / budget_exhausted into one
uninformative value. While in that call site, `SolverResult.h1_error` was
also populated: the field existed, `to_dict()` serialised it, and
`compute_exact_error` had been returning `h1` next to `l2` all along, so the
exported column was permanently null for no reason.

Net: −17-field struct and 3 overrides (~150 lines), +1 shared ladder and 3
four-line hooks; the ladder gained coverage it never had (`max_basis` and
the `>` vs `>=` boundary were untested under `get_result`).

**B18 — the gate.** `lint` now runs
`audit_abstractions src/mcts src/refinement src/pde --fail-on-missing`, plus a
`continue-on-error` pass over the whole of `src/` (the `src/backend` domain-PoC
backlog is untriaged, so it stays advisory). The script is AST-only with
stdlib imports, so it runs inside that job's deliberately minimal dependency
set. Promoting the report-only step to blocking is what remains.

## 5. Prioritized backlog

Documented, not implemented. Ordered by suggested sequencing.

| # | Item | Effort | Risk | Notes / guards |
|---|---|---|---|---|
| B1 | Break the `poc`↔`research` cycle: promote `src/poc/device.py` → `src/device.py` (re-export from the old path), migrate `baselines.py`/`comparison.py` and opportunistically the 122 in-function imports (55 files) | M | Low | Gate paths in `ci.yml`/CLAUDE.md cite `src.poc.device` at 100% coverage — update in lockstep |
| B2 | `CompareScenarioBase` + Config→Params + name-lock-validator unification across the three `*_compare` triplets | L | Med | The three per-scenario 85%-branch regression surfaces, run together |
| B3 | Registry consolidation onto `src/templates/registry.py` (`ScenarioRegistry` last — most consumers) | L | Med | Charter capability guard (subprocess `ScenarioRegistry` read); PDE end-to-end; MCTS evaluator protocol |
| B4 | God-module splits at the seams in §3.1 — one module per PR, import-compatible `__init__` re-exports mandatory | XL | High | Per-module coverage gates (pde 75, training 85, research 85, mcts 90 branch); mypy's per-module override block names old paths and must move in lockstep. **[2026-09-01: PARTIAL — `src/pde/operators.py` (2233 lines, the largest single-class-per-file offender named in §3.1) split into `src/pde/operators/` (`base.py` + one file per operator), PR #140. Every public (non-dunder) name in `dir(m)` unchanged before/after — not literal `dir()` byte-identity, which a package's `__path__`/`__all__` dunders make impossible; a peer review caught this PR's own commit message and CHANGELOG originally overclaiming the stronger form, corrected (see CHANGELOG.md and `tests/pde/test_operators.py::TestOperatorsPackagePublicAPI`). `mypy --strict` error set byte-identical; `pyproject.toml`'s override glob updated to `src.pde.operators.*`. The other 6 files named in §3.1 were untouched by this slice.]** **[2026-09-01: second slice — `src/pde/games/mesh_refinement.py` (1117 lines, the second B4-named file) split into `src/pde/games/mesh_refinement/` (`mesh.py`: the pure quadtree — `ActionKind`, `MeshElement`, `Mesh`, confirmed to have zero import of `src.mcts`/`src.pde.mcts_adapter`; `game.py`: the MCTS-facing `MeshRefinementGame`), following the B21 recipe verbatim. Same public-API guarantee (`tests/pde/test_mesh_refinement.py::TestMeshRefinementPackagePublicAPI`), same byte-identical `mypy --strict` error set, `pyproject.toml`'s override extended to both the bare package name and `src.pde.games.mesh_refinement.*`. One test fix required by the split itself: `test_degenerate_triangulation_is_logged` monkeypatched the package `__init__.py`'s re-exported `logger` attribute, but the `logger.warning(...)` call it exercises moved into `game.py`'s own independent module-level `logger` — patched at the new location. The remaining 5 files named in §3.1 — `trainer.py`, `baselines.py`, `chess.py`, `losses/physics.py`, `agents/config.py` — are untouched; B4 stays PARTIAL/open for those.]** |
| B5 | `BaseScenarioConfig`/`BaseModuleConfig` + `poc`/`templates` logging unification | M | Med | Verify `compute_hash()` stability across the merge (it feeds artifact/log identity) |
| B6 | Make the mypy gate deterministic, then flip `continue-on-error` off | L | Low | Decide a `warn_unused_ignores` policy first — the flakiness is torch-version-dependent, not volume-dependent (§1, §3.4). **[2026-08-14 status: `mypy src/ --strict --ignore-missing-imports` is clean; the flip is an owner decision (§7 register #3), prereq = pin torch in the lint job]** |
| B7 | CI composite action for the ×10 checkout/setup/install preamble + a shared `--ignore`/`--deselect` args file (currently duplicated verbatim between `test-fast` and `coverage`) + `COVERAGE_CORE` centralization | M | Low-Med | Preserve step names — the charter's Quality-Gate guard parses `ci.yml` by `- name:` |
| B8 | Extend the drift guard in §3.4 to CLAUDE.md's Regression Surface table, not just the charter's | M | Med | New test under `tests/docs/`; don't conflict with the existing charter meta-guards |
| B9 | Seed-stride value unification (1009 vs 7919) | S per scenario | Med | Requires re-recording `config/baselines/*.json` via the existing `record-baseline`/`diff` CLI, one scenario at a time |
| B10 | Dead-package decisions: ~14k LOC (15% of `src/`) across `prototyping`, `tournament`, `analysis`, `curriculum`, `deployment`, `demos` have zero inbound `src/` references and are held alive only by their own tests | M each | Med | A cut needs the charter's scope register amended first; a keep-and-wire needs its own PR (e.g. `trainer.py`'s `_run_checkpoint_tournament` re-implements tournament logic instead of importing `src/tournament`). **[2026-08-14 reclassification: only 4 of the 6 are actually dead (`prototyping`, `tournament`, `analysis`, `curriculum` — zero inbound refs, 2026 commits are repo-wide sweeps only). `deployment` is CI-exercised every run (test-extras ONNX suites, green) → keep or explicit-deprecate. `demos` is a live dashboard dependency (`dashboard/app.py:58-60` imports 3 demo tab factories; substantive Aug-2026 commits) → keep. Tournament-duplication claim verified.]** |
| B11 | YAML dedup (the `lm_studio` block copy-pasted across 6 scenario configs; `llm_prior_*` triplets differing by 4/75 lines) plus a decision on the flagged-not-deleted hydra configs (`train_5hr.yaml`, `train_experiment.yaml`, `config/presets/*`, `darcy_poc.yaml`, `transfer_ablation.yaml`) with the `hf_space/config/` mirror in view | S | Low | Demo-YAML validation tests per scenario surface |
| B12 | AGENT.md authoring for the 14 uncovered packages, starting with `src/research` (§3.7) | M | None | `check_doc_links.py` |
| B13 | CLAUDE.md/CHANGELOG restructure — CHANGELOG's `[Unreleased]` section is the large majority of the file; cut a release. CLAUDE.md's milestone log is append-only by its own header, so this is about the drift guard (B8), not moving content | M | Med | `tests/docs/` suite, docs CI workflow |
| B14 | `hf_space/` single-sourcing | — | — | Out of scope for this effort; owned by `tests/hf_space/test_mirror_guard.py`'s tracked follow-up |
| B15 | `scripts/run_{lshape_amr,transfer_baseline_compare,stochastic_galerkin_compare}.py` duplicate CLI/config-loading/baseline-diff boilerplate that `src/poc/cli.py` and `src/templates/cli.py` already provide but no script imports | M | Med | Each script has a dedicated `tests/scripts/` file; shrinks after the dedup |
| B16 | Prune the `tests/poc/` local registry-clear fixtures now that a save/restore wrapper exists (needs per-file analysis of the `sys.modules`-purging ones, §3.6); unify the drifted helix-geometry test fixture; a deprecation-timeline policy for the repo's ~19 back-compat shims; a config-driven `device` field (fail-loud default) for the 3 classic PoC scenarios | S-M | Low-Med | Respective module test suites |
| ~~B17~~ ✅ **DONE** | **Resolved the dead abstraction `PDEGame.get_result`** — see §4.1 below. Deleted (not wired), and replaced by the genuinely-consumed `PDEGame.termination_reason` | S-M | Med | `python -m scripts.audit_abstractions src/pde --fail-on-missing` now exits 0; PDE end-to-end + reward-reachability + clone-isolation surfaces re-run green |
| ~~B18~~ ✅ **DONE** | **Wired the abstraction audit into CI** — see §4.1 below. `lint` job now gates `src/mcts src/refinement src/pde --fail-on-missing`, with the rest of `src/` report-only | S | Low | The audit script's own tests (`tests/scripts/test_audit_abstractions.py`) |
| B19 | **Document the 8 optional extras.** `fem`, `viz`, `dev`, `test-extras`, `jax`, `jax-gpu`, `picogk`, `lm-studio`, `docs` are undocumented in `README.md`, `CONTRIBUTING.md`, and `docs/getting-started.md`, all of which show only `pip install -e ".[dev]"`. Separately, `dashboard*` ships in the wheel (`pyproject.toml` `include`) but imports matplotlib **and** gradio at module level with no extra covering it — `pip install alphagalerkin && python -m dashboard.app` fails. Consider a `dashboard` extra | S | Low | None (packaging metadata + prose) |
| ~~B20~~ ✅ **DONE** | **Add the missing per-module coverage gates.** `src/poc/cli.py`, `src/poc/visualization/*`, the 3 classic scenarios, `src/constants.py` and `src/seeding.py` are covered only by the global 85% gate. Note `CLAUDE.md` documents a `noyron_basis` gate (97%/100%) that **is not wired in `ci.yml`** — the charter guard passes because it asserts documented thresholds ⊆ CI values, not that the gate step exists | S-M | Low | `tests/docs/test_charter_alignment.py`; add the step or drop the claim. **[2026-08-14: CLOSED-WIDER — the phantom-gate class was 3 gates, not 1 (`noyron_basis`, Noyron HX per-module, SBIR P40 per-module: zero `noyron`/`geometry_picogk`/`gpu_profiler` mentions existed in `ci.yml`), plus a fourth, *degraded* gate: under coverage 7.x, file-path `--cov=path.py` specs are silently dropped, so the llm_prior file-level gates enforced nothing. All four wired/repaired in native-runner form; see §7.]** **[2026-09-01: CLOSED — the remaining 5 (`poc/cli.py`, `poc/visualization/*`, the 3 classic scenarios, `constants.py`, `seeding.py`), PR #140. Measured before any new tests: cli.py 90%, visualization 89%, the 3-scenario set 71% (dragged by `transfer.py` at 25% — `execute()`/`_train_model()` had zero coverage), constants.py/seeding.py 100%. New tests closed the real gaps (`transfer.py` micro-run suite, `cli.py`'s untested `cmd_eval_harness`, `visualization`'s untested `pareto_frontier`); all five now measure 85%+ and gate at 85 in `ci.yml`.]** |

| B21 | **Formalize a `god-file-split` skill.** PR #140's `src/pde/operators.py` split (B4) worked out a repeatable recipe the hard way — this session's own attempt failed pre-commit's `check-doc-links` hook on the first try (a dangling markdown link in `docs/PLAN_2026-04-27.md` a repo-wide `grep` for the path prefix had missed) and a second stale reference (`docs/architecture/components.md`'s bare `` `operators.py` `` mention, no path prefix to grep for) surfaced only in a follow-up review pass. The recipe is now proven and worth codifying before the other 6 files named in B4 repeat the same trial-and-error: convert to a package with `__init__.py` re-exporting the full public surface (verify via a before/after diff of *public* names only — `[n for n in dir(m) if not n.startswith("_")]` — not the raw `dir()` output, which a package's `__path__`/`__all__` dunders will always change; the operators.py split's own commit message and CHANGELOG got this distinction wrong the first time, caught by peer review); update the `pyproject.toml` mypy override from the bare module name to `<module>.*` (overrides do not cascade to submodules); grep for the old path both with and without a directory prefix, since prose sections already scoped to the parent directory often drop it; check `ARCHITECTURE.md`'s drift guard and `docs/architecture/*` for path citations; re-run `check_doc_links.py`. | S | Low | Would guard the next B4 split from repeating the same two near-misses; no code impact by itself |
| B22 | **Concurrent-subagent working-tree safety.** Dispatching 3 background agents against non-overlapping file sets in the same working tree (rather than isolated git worktrees) hit a real collision during PR #140: one agent's own `git stash`/`git reset` (used internally, not instructed) stashed away a second agent's uncommitted, unrelated test-file edits alongside its own in-progress work. No data was lost this time — HEAD and prior commits were unaffected (`git reset` with no target is index-only), and the second agent recovered its own files via `git checkout stash@{0} -- <its files>` — but the mechanism is a real hazard: any subagent with Bash access can run working-tree-wide git commands that affect every other concurrently-running agent, whether or not their file scopes overlap. Options: default to `isolation: "worktree"` for any multi-agent dispatch that gives subagents Bash+git access, or add an explicit instruction in dispatched prompts restricting subagents to `git status`/`git diff`/`git add <pathspec>` and forbidding `git stash`/`git reset`/`git checkout` without a narrow pathspec. | S | Low | Process/orchestration convention, not a code change; worth a line in `AGENT.md` or the relevant `.claude/agents/*.md` frontmatter for agents commonly run concurrently |
| B23 | **PR and CI schedule hygiene.** Two items observed while driving PR #140 to green. (2) **DONE (2026-09-01)**: a monthly Routine now lists open PRs and flags ones whose base SHA is 20+ commits behind their base branch's tip or with no activity in 30+ days (reporting-only by design — it does not comment/label/close, a human decides what to do with a flagged PR). One residual uncertainty: `create_trigger` warned that self-bound routines "store no MCP connectors," which if it turns out to matter in practice would mean the fired session lacks working GitHub tools when it runs next month — self-binding to the session that already holds this repo's GitHub MCP access was the mitigation, but this is unverified until the first firing. (1) **NOT DONE, re-scoped after investigation**: the nightly `schedule`-triggered CI run genuinely burns ~53-58 real minutes on an unchanged commit (measured via the GitHub Actions API's workflow-run listing for the `CI` workflow, comparing `created_at`/`updated_at` on 4 consecutive cancelled runs — not the `gh` CLI, which is not available in every environment this repo is worked from), confirming real waste, not a near-instant cancellation. Root cause, precisely: `schedule` exists only so the nightly-only `test-slow` job (`ci.yml`'s own `if:` names `github.event_name == 'schedule'` explicitly) runs at all; every *other* job has no such exclusion, so the full ~12-job fast/lint/coverage matrix re-runs as a side effect on every nightly firing regardless of whether anything changed. The safe fix (scoping every non-`test-slow` job to exclude `schedule`) means touching ~12 job definitions in the same file that gates every open PR's mergeability — deliberately deferred to its own dedicated, carefully-tested change rather than bundled into an unrelated PR under time pressure, per this project's own git-safety convention about wide-reaching edits to shared CI config. | S | Low | (2): the Routine itself, first firing 2026-10-01. (1): `.github/workflows/ci.yml` — needs its own PR, ~12 jobs' `if:` conditions, verified against a real scheduled run before merging |
| ~~B24~~ ✅ **DONE (2026-09-01)** | **Fail-fast validation for `BasisSelectionGame` operators with no exact solution** (`id_pde`/`PDEName`/`ResearchPDEName`), landed via a **different mechanism than originally proposed here**. The graceful-arm-skip pattern this row originally suggested (mirroring the LLM-preflight-failure skip) turned out to be the wrong shape on investigation: `ExactSolutionUnavailableError` is raised inside `_centaur_common.run_basis_selection_cell`'s `PDEGameAdapter(game)` construction, and that function's `CellOutcome` return value is unpacked positionally (`rollouts_used, final_residual = ...`) at every one of its 5 call sites (`llm_prior_ablation.py`, `scaling_law.py`, `agents/research_loop.py`, `noyron_basis.py`, `src/integrations/eval_harness/target.py`) — adding a third `skipped` field would break that unpacking everywhere, and representing "skipped" via a sentinel inside the existing 2 fields (e.g. `final_residual=nan`) risks a NaN silently entering a median/aggregate calculation and being reported as a real metric, which is a *worse* defect than the current loud crash (it would be exactly the class of silent-degenerate-metric bug P0-1 itself exists to eliminate). Implemented instead: a `@field_validator` on each of the three config classes rejecting `heat`/`advection_diffusion` at **construction time** with a message naming `ExactSolutionUnavailableError` and this row, so the user gets an immediate, actionable error instead of a crash minutes into a run after other arms/seeds already spent compute — arguably better UX than either the original crash or a graceful mid-run skip. The rejection list (`_PDES_WITHOUT_EXACT_SOLUTION`, one frozenset per file) is duplicated across the three files rather than imported from a shared module, matching `agents/config.py`'s own pre-existing, documented decision to stay decoupled from the heavy MCTS/PDE import surface. A pre-existing test (`test_research_loop.py::test_config_problem_names_must_be_unique`) used `pde="heat"` incidentally while testing an unrelated thing (duplicate-name detection) and was updated to `pde="burgers"`. 6 new tests (2 per config class: reject both unsupported values, accept the `poisson` default). | S-M | Low | `pytest tests/poc/test_llm_prior_ablation_config.py tests/poc/test_scaling_law_config.py tests/agents/test_research_loop.py -v` |

## 6. Rejected or deferred, with reasons

- **The `ScenarioRegistry` snapshot/restore fixture — attempted, measured,
  reverted.** It was the headline "quick win" of the `test(poc)` commit and an
  adversarial review found it was a *net regression*. Probing the live registry
  immediately after `tests/poc/test_cli_commands.py`:

  | teardown semantics | registry afterwards |
  |---|---|
  | no fixture (baseline) | 10 real scenarios |
  | `clear()` + restore snapshot | `[]` — **empty for the rest of the process** |
  | restore-only-what's-missing | `['alpha','beta']` — real scenarios lost |
  | skip-restore-when-snapshot-empty | `['alpha','beta']` — same |

  Root cause: several modules here also purge
  `sys.modules['src.poc.scenarios*']` so their `@scenario` decorators re-fire on
  re-import. A per-test snapshot taken *before* that re-import is empty or
  partial, so restoring it deletes registrations the test legitimately created —
  and with `sys.modules` repopulated the decorators cannot fire again. Three
  different semantics each traded one failure mode for another; none beat doing
  nothing. Removed, and the local fixtures left untouched. The genuine fix is to
  rework those heterogeneous local fixtures (backlog **B16**); until then the
  subprocess read in `test_charter_alignment.py` remains the right mitigation.
  The structlog half of the same conftest is kept — it was probe-verified and
  has no such interaction.
- **`structlog.testing.capture_logs` is inert against this repo's module-level
  loggers — the fourth ordering defect on this branch.** A test added here to
  cover the new `interpolator_build_failed` warning asserted through
  `capture_logs` and failed CI on all three Python versions with `assert 0 == 1`
  — while the CI log's own "Captured stderr call" section showed the warning
  being emitted. `capture_logs` swaps structlog's *global* processor chain, but
  `src/poc/logging.py` sets `cache_logger_on_first_use=True`, so once any earlier
  test calls `configure_logging`, a module's
  `logger = structlog.get_logger(__name__)` proxy has cached a stdlib-backed
  bound logger and never consults the chain again. Passes alone, fails in a full
  run. Fixed by patching the module's `logger` attribute with a recording
  double. Counting the two `sys.modules`-purge failures and the reverted
  registry fixture, **four of this PR's own defects were global-singleton state
  crossed with collection order** — all four invisible to per-file verification.
  The three resulting rules are written into `tests/poc/conftest.py`'s docstring
  rather than left as folklore.
- **Two errors that made it past local verification into CI, caught and
  fixed within minutes.** `git grep` for the literal filename
  `stochastic_galerkin_compare_demo.yaml` found nothing before deleting it
  as an orphan — but `tests/poc/test_stochastic_galerkin_compare_config.py`
  builds that path from an f-string
  (`f"stochastic_galerkin_compare_{basename}.yaml"` parametrized over
  `["ci", "demo"]`), which a literal grep cannot find. Separately, adopting
  `derive_seeds` in `stochastic_galerkin_compare_config.py` added
  `src.seeding` as a new import, tripping
  `tests/pde/stochastic/test_import_isolation.py`'s explicit, curated
  allowlist of that layer's dependency surface (by design — its own error
  message reads "extend the allowlist deliberately if this is intended").
  Both were real regressions, both failed CI's `Unit Tests (Fast)` job on
  all three Python versions, and both were fixed the same way they'd be
  fixed in any PR: the YAML was restored byte-identical from git history,
  and `stochastic_galerkin_compare_config.py`'s `resolved_seeds()` was
  reverted to its original inline body rather than widening a guard that
  exists specifically to keep this one module's dependencies narrow and
  audited (CLAUDE.md documents that narrow surface explicitly). The lesson
  for future work in this vein: a "zero references" grep is only as good
  as its ability to see constructed paths, and any new import into
  `src/poc/scenarios/stochastic_galerkin_compare*.py` specifically needs
  the import-isolation test checked, not just the general test suite.
- **`_PDE_TYPE_MAP` deletion** — considered dead (never read in
  `llm_prior_ablation.py`), but it is pinned by
  `tests/regression/test_centaur_regression.py`'s identity assertion
  (`llm_prior_ablation._PDE_TYPE_MAP is PDE_TYPE_MAP`) and named explicitly
  in CLAUDE.md's Centaur test pyramid regression-surface row. Left in
  place.
- **Deleting the hydra-addressable config YAMLs** — `train_5hr.yaml`,
  `train_experiment.yaml`, and `config/presets/*` have zero textual
  references but are runnable via `python -m scripts.train
  --config-name=...` and are mirrored into `hf_space/config/`. "Zero
  references" is the wrong test for a CLI-addressable entry point; demoted
  to flag-only (backlog B11).
- **Pruning all 9 `tests/poc/` local registry-clear fixtures** — they are
  not homogeneous. At least two also purge `sys.modules` so `@scenario`
  decorators re-fire on re-import, which a registry snapshot/restore does
  not reproduce; pruning those would be a behavior change, not a cleanup.
  Only the additive fixture landed (backlog B16 for the prune).
- **Dropping `hydra-core`** — considered as an unused dependency (zero
  imports in `src/` or `config/`), but `scripts/train.py` and
  `scripts/train_chess.py` both use `@hydra.main`. It is scripts-only, not
  unused; left in place.
- **`ScaleNorm`'s `eps: float = 1e-5` → `LAYER_NORM_EPSILON`** — same
  numeric value as the constant, but `ScaleNorm` is a different
  normalization concept than the constant's name asserts provenance for.
  Wiring it would create a misleading name-binding to save one literal;
  skipped (the other two constant-reconciliation wirings in commit
  `refactor(constants)` are both exact concept+value twins).
- **Docstring `print()` example edits in `src/training/stability.py`** —
  these are `print()` calls inside class docstring usage examples, not
  runtime code; "fixing" them would be review noise with zero behavioral
  effect. Skipped.
- **`src/deployment/validate.py` and `src/tools/verify_invariance.py`
  exception handling** — initially flagged alongside the 4 sites fixed in
  this PR, but both actually *propagate* the caught error to their caller
  (`{"error": str(e)}` / appended to a returned `errors` list) rather than
  swallowing it silently. Not a defect; left unchanged.

## 7. Phase-1 follow-up (2026-08-14, PR after #119)

The tech-debt PR that consumed this audit's backlog landed the following. Every
change was adversarially peer-reviewed by five parallel review passes before
implementation; all numbers below are measured at HEAD, not estimated.

### 7.1 CI enforcement fixes (the real "get CI green" work — CI was green but unenforced)

- **CI never ran on pull requests**: `on.pull_request.branches: [main, develop]`
  named branches that do not exist (the default branch is
  `claude/alphagalerkin-implementation-4zGEN`), so all open PRs merged with zero
  checks. The filter is removed; `test-slow`'s `if:` had the same dead branch
  names and now keys on `github.event.repository.default_branch`.
- **Phantom gates closed (B20)**: 3 documented-but-unwired gates added in
  native-runner form — noyron_basis (measured 98% combined), Noyron HX surface
  (99%), SBIR P40 surface (94%).
- **Degraded gate repaired**: coverage 7.x silently drops file-path
  `--cov=path/to/module.py` specs (only a `CoverageWarning`, slug
  `module-not-imported`); the llm_prior
  gate passed on the `lm_studio` directory alone. The file-level pair now runs
  as a native-runner step. **Decay evidence**: with the gate unenforced,
  `llm_prior_ablation.py` drifted to a measured 77% branch (81% combined with
  its config) vs the documented 86%. Gated at 79 = measured − 2; ratcheting
  back to 85+ is Phase-2c work. A CAUTION comment in `ci.yml` bans the
  file-spec form.
- **Unenforced guarded surface re-enabled**: `tests/pde/test_mcts_adapter.py`
  (documented F1/F3 Regression Surface) ran in NO CI job since the 2026-04
  triage — the `--ignore` is removed from both blocks (37 tests, green). The 2
  CUDA deselects that duplicate in-source `skipif` markers are removed; the
  remaining 9 deselects + 6 ignores all pass at HEAD (fixes landed on parallel
  branches weeks after the triage — the list is stale, not broken) and are
  Phase-2 staged deletions (matrix-verified, mcts pair last).
- Dead `.pytest_cache/` artifact-upload step deleted; `--no-cache-dir` dropped
  (it defeated the job's pip cache); Stage labels renumbered.

### 7.2 Hardcoded-value fixes (zero numeric change, verified by assertion)

- Boundary tolerances named, not unified: `DEFAULT_BOUNDARY_TOLERANCE` (1e-6)
  now used by `operators.py`; new `DEFAULT_PICOGK_BOUNDARY_TOLERANCE` (1e-5)
  documents the SDF-band semantic and the pre-existing picogk operator/domain
  divergence. `is_boundary_point` has zero production call sites — the risk was
  drift, not baselines.
- LR scheduler had **three** copies of `min_lr_ratio`/`warmup_start_factor`:
  `BaseTrainerConfig` fields (0.01/1e-6), `_create_scheduler` static defaults
  (0.01/1e-6), and `Trainer`'s bare literals (0.1/0.1). The first two now bind
  to `DEFAULT_MIN_LR_RATIO`/`DEFAULT_WARMUP_START_FACTOR`; the third is now two
  typed `config.schemas.TrainingConfig` fields (defaults 0.1/0.1 — value-
  preserving) that `Trainer` passes through; keys added to `config/train.yaml`.
- Checkpoint-migration defaults (4 literals, `checkpoint_migration.py`) are
  **intentionally frozen** — a v1.1.0 migration must inject v1.1.0's defaults
  forever — with a drift-alarm test that fails if a live default is retuned.
- Gumbel epsilons split by semantic: `GUMBEL_NORMALIZATION_EPSILON` (inert
  division guard) vs `GUMBEL_LOG_PRIOR_FLOOR` (algorithmic — sets zero-prior
  actions' ≈−18.4 score); `FNetEvaluator` gets `_SOFTMAX_NORMALIZER_FLOOR`,
  mirroring `lm_studio/evaluator.py` by name as documented there.
- `[9, 13, 19]` literals at 13 code sites → `list(DEFAULT_BOARD_SIZES)` /
  `default_factory=lambda: list(...)` (never the bare mutable module list).

### 7.3 Dead abstraction demoted

`BaseTrainer.compute_loss/generate_data/evaluate` were `@abstractmethod`s that
both production subclasses stubbed with `NotImplementedError` — a dead
contract. Demoted to concrete `step()`-loop hooks that raise with guidance;
subclass stubs KEPT (their messages are asserted by
`tests/training/test_trainer_coverage.py` and document each trainer's real
entry points). Pre-existing `audit_abstractions` baselines recorded:
`src/training` → `BaseLoss.forward` (1 hit), `src/pde` → `PDEGame.get_result`
(B17, resolution decided: demote + hoist the 3 near-verbatim implementations
into a concrete base with a `termination_reason` hook — production wiring was
rejected, it would add per-episode `compute_exact_error` solves).

### 7.4 Coverage actuals for ungated packages (branch %, measured at HEAD)

Gate-setting input for Phase 2c: gate at actual−2, ratchet toward 85.

**[2026-08-15: 6 of 8 CLOSED.]** `src/agents`, `src/tools`, `src/experiments`,
`src/curriculum`, `src/engines`, `src/data` are now gated in `ci.yml`'s `coverage`
job at floor(re-measured branch %)−2 (capped at 85), mirrored into CLAUDE.md's
Regression Surface and the charter gates register
(`openspec/specs/project-charter/spec.md`). `src/templates` and `src/math_kernel`
were re-measured and deliberately **not** gated — triage found a real gap in each
rather than dead code: `src/templates/cli.py` measures 0% directly but is live
production code (`src/agents/cli.py` imports it; exercised indirectly at ~66% via
`tests/agents/test_cli.py` + `tests/e2e/test_centaur_e2e.py`) with no test in
`tests/templates/` itself; `src/math_kernel` is uniformly low (55–64%) across
`basis.py`/`integral.py`/`spectral.py`, driven almost entirely by the
`HAS_JAX`-guarded `Jax*` classes that no test anywhere in the repo (including the
dedicated `test-jax` CI job) exercises, plus a handful of untested
input-validation branches. Both remain open Phase-2c candidates, now with a
concrete test-writing scope instead of an unmeasured guess. This closes the
`src/agents`/`src/tools`/`src/experiments`/`src/curriculum`/`src/engines`/`src/data`
half of Phase 2c only — B20's own literal list (`src/poc/cli.py`,
`src/poc/visualization/*`, the 3 classic scenarios, `src/constants.py`,
`src/seeding.py`) is a separate set and remains untouched.

| Package | Branch % | Tests | Notes |
|---|---|---|---|
| src/agents | 95.4 | 292 | path-form `--cov=src/agents` measures the whole package fine; the CI native-runner step gates only research_loop+config (85). `-p no:cov` there is the dotted-cov/torch workaround, NOT disabled coverage |
| src/tools | 91.9 | 171 | |
| src/experiments | 89.8 | 153 | benchmark_fnet.py 61% |
| src/curriculum | 89.0 | 128 | |
| src/engines | 84.3 | 133 | match.py 68%, elo.py 73% |
| src/data | 79.8 | 82 | physics_dataset.py 23% is the drag |
| src/templates | 72.5 | 107 | **cli.py 0% (121 stmts)** — triage before gating |
| src/math_kernel | 61.5 | 155 | uniform 55–64% — triage before gating |

### 7.5 Mypy override debt (measured)

8 override blocks in `pyproject.toml` (not ~10 as §1 estimated). The 51-module
block masks **207 errors** (`arg-type` 90, `assignment` 48, `attr-defined` 17;
tail: `pde/operators.py` 28, `training/evaluation.py` 18; 1 module already
clean; 32 modules ≤3 errors). Ratchet plan: PR 0 = free trim (delete the
0-error `math_kernel` decorator block, shrink the `no-untyped-call` block to
`research.baselines` (1 error), drop clean `physics.solver` from the big
list), then ~5 modules/PR; 49 of the 207 errors sit in B10-candidate packages
— decide B10 first; `operators.py`'s 28 ride the B4 split.

### 7.6 Follow-up roadmap and owner-decision register

**Phase 2** (first PRs under enforced PR CI): 2a B1 device promotion
(lockstep: the 2 `--cov=src.poc.device` citations AND the Trained-evaluator
Regression-Surface row naming `_resolve_device_cached`); 2b seeding/layering
dedup (stochastic layer EXCLUDED per §6); 2c coverage gates at actual−2
(agents 93, tools 90, experiments 87, curriculum 87, engines 82, data 78),
owner decisions #2/#3, mypy free-trim, staged deselect/ignore deletions,
dependabot #103–#107 (all `mergeable_state: clean`).
**Phase 3**: B10 batch (register #1), `src/backend/rng.py` + test-only plot
types + `templates/cli.py` deletions, B17→B18 (audit tool extension rule:
@abstractmethod-only, declaration-body excluded, "every concrete override
raises" quantifier verbatim, `_KNOWN_LIVE` preserved), mypy ratchet,
math_kernel/templates triage, B16/B19/B12.
**Phase 4**: B4 splits (operators.py: registration stays centralized in
`registry.py` — per-class decorators would cycle; `__init__` re-exports
`DEFAULT_HELMHOLTZ_WAVENUMBER` + `PDEResidual`; hf_space mirror keeps its
monolith), B3 registries, B7 composite action, B8, B15, B13.

| # | Owner decision | Recommended default |
|---|---|---|
| 1 | B10: 4 dead packages + tournament wire-vs-delete | Delete prototyping/analysis/curriculum; tournament = owner's call; deployment deprecate-don't-delete; demos keep |
| 2 | Gate integration/jax/chess/extras in `ci-success` | Yes — 13+ consecutive green default-branch runs; ONNX step stays step-level continue-on-error |
| 3 | mypy flip to hard gate | Yes, after pinning torch in the lint job |
| 4 | Coverage ratchet targets | actual−2 now, +2/quarter toward 85 |
| 5 | test-slow on PRs | Keep nightly + `[full-test]` opt-in |
| 6 | Shim deprecation window | 2 minor releases |
| 7 | Release cut (B13) | After Phase 2 |

### 7.7 Post-Phase-1 peer review — findings and dispositions (2026-08-15)

A four-lens review (adversarial diff review, SQE coverage/edge-cases, hygiene sweep with extended
ruff rule sets, skills/reusability architecture) ran against the Phase-1 branch. Fixed-in-branch
items are marked ✅; the rest are prioritized backlog with the evidence needed to act on them.

#### P0-1 — Flat MCTS reward on 3 of 8 PDE operators (degenerate acceptance criteria)

**Independently verified by execution, not inspection.** Three links, each confirmed:

1. `src/pde/operators.py` — `BurgersOperator.__init__` assigns
   `self.is_time_dependent = config.is_time_dependent`, **overwriting** the class-level
   `is_time_dependent = True`. `PDEConfig.is_time_dependent` defaults to `False` and
   `_centaur_common.build_pde_operator` never overrides it. Measured: class-level `True`,
   instance `False`. Same pattern in `AdvectionDiffusionOperator`.
2. → `BurgersOperator.exact_solution` hits its `if not self.is_time_dependent: return None`
   guard. Measured: Burgers → `None`, Poisson → `ndarray`.
3. → `BasisSelectionGame.compute_exact_error` falls back to RMS of `state.residuals`, and those
   residuals are structurally zero: `compute_derivatives` early-returns all-zero derivatives when
   `u` is grad-free, and the caller always passes `torch.from_numpy(state.solution)` — grad-free
   by construction. With a zero source term the error is identically 0.0.

Measured per-operator error trajectories: `burgers`, `heat`, `advection_diffusion` → `[0.0, 0.0,
0.0, 0.0]` (FLAT); `poisson`, `helmholtz`, `biharmonic` → non-degenerate.

**Consequence**: `_centaur_common.run_basis_selection_cell` early-returns before constructing MCTS
(`current_error <= target_residual`), so the Burgers OOD cell runs **zero rollouts on every arm and
seed**. `config/scenarios/llm_prior_demo.yaml`'s headline OOD gates are therefore degenerate:
`ood_llm_residual <= 1e-2` passes trivially and unfalsifiably (0 ≤ 0.01) while
`ood_trained_residual > 1e-1` fails unconditionally. The shipped demo's OOD claim is evidence in
neither direction. Helmholtz/Biharmonic carry manufactured solutions, which is why the
OOD-expansion tests pass and this stayed invisible.

**Why not fixed here**: the fix changes solver semantics and invalidates the shipped OOD
thresholds, which must be re-derived from a real measurement. That is a dedicated PR, not a
rider on a CI/constants change. Landed now: a `logger.debug("derivatives_skipped_u_disconnected", …)`
at the zero-derivative branch, so the condition is diagnosable instead of silent.

**Fix order when taken up**: (1) stop letting `config` silently downgrade a class-level
`is_time_dependent = True` — honour the class default or raise on the contradiction; (2) make
`compute_exact_error` refuse the residual fallback when `_exact_solution is None` rather than
reporting a constant as convergence; (3) re-run `llm_prior_demo.yaml` and re-derive `ood_*`.

> **Status (2026-08-21).** Step (1) was done in the 2026-08-19 hygiene pass, which made
> `BurgersOperator.exact_solution` reachable and thereby exposed a *second*, independent
> defect in the Cole-Hopf series itself — every Fourier coefficient was hardcoded to 1
> (the transform of a Dirac comb, not of a sinusoid), so the reachable "ground truth" was
> ~1e12–1e13 rather than flat zero. **That second defect is now fixed**: the coefficients
> are `2*(-1)^n*ive(n, R)`, `R = 1/(2*pi*nu)`, and `initial_condition` / `boundary_value` /
> `exact_solution` were unified onto the single benchmark `u(x,0) = -sin(pi*x)` on `[0,1]`
> with homogeneous Dirichlet data (they previously described three different problems).
> Measured for the `ood_pde="burgers"` arm: `BasisSelectionGame` initial
> `error_estimate` **4.20e12 → 0.7071**; the operator is no longer degenerate on either
> the flat-zero or the 1e12 axis. **Step (2) is now done (2026-09-01, PR #140)**:
> `BasisSelectionGame.get_initial_state`/`compute_exact_error` raise a new
> `ExactSolutionUnavailableError` instead of falling back to the degenerate
> zero-residual for operators with no exact solution (`HeatOperator`;
> `AdvectionDiffusionOperator` without a time arg). `get_initial_state` is
> the one that mattered in practice — it hits the identical bug first, at
> episode start, before `compute_exact_error` is ever reached. Step (3) is **still open by
> design** — the `ood_*` thresholds were deliberately left untouched because re-deriving
> them requires the real GPU run, but they are now measurable against a physically
> meaningful solution (`ood_llm_residual <= 1e-2` demands a ~70x reduction from 0.707,
> which is demanding-but-attainable, where before it was arithmetically impossible).
> See `docs/CODE_HYGIENE_REVIEW_2026-08-19.md` for the full before/after and the
> documented `nu -> 0` conditioning limit of the Fourier-Bessel representation.
Re-measure the `noyron_basis` "~2–4% best-case reduction" open research item afterwards — it is
plausibly a symptom of link 3 rather than of basis/geometry mismatch.

#### P1 — Silent failures in numeric paths (partially fixed)

- ✅ `src/modeling/stability.py` — an SVD `RuntimeError` returned `beta = 0`, which is *exactly*
  the LBB-violation alarm value, making a numerical crash indistinguishable from a genuine
  stability finding on the metric the novelty claim rests on. Now logs `lbb_svd_failed` with shape
  and an explicit note.
- ✅ `src/poc/tuning/sampler.py` — a missing `optuna` made `sampler="tpe"` silently run **random
  search** while still reporting TPE. Now warns with a remedy; `optuna` is also now a declared
  `[tuning]` extra rather than an undeclared import.
- Open: `src/pde/games/basis_selection.py` (singular Galerkin system → pinv fallback; a singular
  system is precisely the LBB inf-sup failure the project claims to detect),
  `src/experiments/physics_model.py`, `src/research/fem_baseline.py`,
  `src/training/loss_balancing.py`.

#### P1 — Reproducibility: two disjoint numpy RNG worlds

`src/seeding.py::set_global_seeds` seeds the **legacy** `np.random` global, but ~15 modules use the
Generator API (`np.random.default_rng(None)`), which draws fresh OS entropy and is untouched by it —
including `PDEOperator.generate_collocation_points(seed=None)`, called inside the physics-loss
training loop. Either thread explicit seeds into every `default_rng` call site or add a child-seed
helper; until then `src/seeding.py`'s docstring overstates its reach. Note the legacy API itself is
deliberate (converting it would change derived seeds and invalidate committed baselines — same
hazard as B9).

#### P1 — CUDA correctness (partially fixed)

- ✅ `src/research/stochastic_galerkin_compare.py` — `field.numpy()` on a device tensor would raise
  on any CUDA run of the artifact path; CPU-only tests structurally cannot see it. Now
  `.detach().cpu().numpy()`.
- Open: 12 more bare `.numpy()` in `src/pde` (safe today only because `sample_interior` defaults to
  CPU and no caller passes a device — the picogk/Noyron path is GPU-preferred). Normalise on the
  codebase's own correct idiom (`PDEResidual.to_numpy`). Also `src/pde/operators/base.py`'s (moved
  from the pre-split `src/pde/operators.py`, 2026-09-01) in-place `coords.requires_grad_(True)`
  mutates a caller-owned tensor that shares memory with a numpy array; the same pattern recurs in
  `operators/biharmonic.py` and `operators/navier_stokes.py`.

#### P2 — Config fields that are declared but never read

~~Wire or delete: `GumbelMCTSConfig.use_mixed_value` (the *defining* feature of Gumbel AlphaZero —
setting it changes nothing), `GumbelMCTSConfig.discount` (gumbel search ignores its own discount)~~
**[2026-09-01: DONE, PR #140 — both wired.** `use_mixed_value` now gates a new
`_gumbel_mixed_value()` implementing the v_mix estimator (Danihelka et al. 2022, Appendix B) used to
complete unvisited children's Q-values instead of a flat `0.0`; `discount` now scales the one-step
backup into `value_sum`. `use_mixed_value` defaults to `True`, so this is a real behavior change
under default settings, not just when explicitly toggled. New deterministic tests in
`tests/mcts/test_gumbel_integration.py::TestSequentialHalvingConfigKnobs` construct a
budget-starved root and prove each knob's effect exactly (a toggle flips `best_action`; a
`discount < 1.0` scales `value_sum` by the exact factor), rather than re-asserting the fields
exist.**]
`PDEGameConfig.error_metric` (`h1` silently yields l2), `BaseScenarioConfig.requires_gpu` (set by 4
scenarios, never checked — a `requires_gpu=True` scenario runs to completion on CPU and reports
PASS), `BasisSelectionConfig.rbf_kernel` (3 of 4 options unimplemented), `PDEGameConfig.success_metrics`,
`StrangTrainerConfig.n_particles`, `dt_min`/`dt_max`.

#### P2 — Lint rules worth enabling (~60 real fixes, no suppressions)

`TRY400` (18 `.error()`-in-`except` sites dropping tracebacks), `G201`, `DTZ` (18 naive
`datetime.now()` written into persisted artifacts and checkpoint metadata), `T20` (13 real —
`ScenarioRunner._print_summary` prints from a *library* class, invisible to structlog and to
`capture_logs`; the other 91 are legitimate CLI entry points needing per-file-ignores), `S301`
(pickle over worker IPC), `S324` (md5 → `usedforsecurity=False`). Deliberately **not** recommended:
`G004` (all 9 hits build dynamic structlog *event names* — idiomatic), `NPY002` (legacy RNG is
deliberate, see above), `TRY003`/`EM101`/`EM102` (810 findings, pure style; this repo's long
contextual exception messages are a feature).

#### P2 — Next-tier hardcoded values

Ordered by blast radius, not count: the LBB regularization margin multiplier `* 10` in
`src/modeling/galerkin_operator.py` (the same concept is already named **twice** in
`src/modeling/stability.py` — three copies, one literal, and it shapes the training gradient);
`min(batch_size * 10, replay_buffer_size // 10)` deciding when training starts; the triplicated
policy-CE floor `clamp(min=-100.0)`; Cole-Hopf `n_terms = 50` and its `1e-10` denominator floor
(the L2 reference for SBIR rows); `board_size = ... else 8` fallback in self-play (a silent wrong
shape); the FNO projection head fixed at 128 while every sibling dimension is a parameter.

#### P2 — Zero-logging packages

`src/pde/stochastic/` (all 11 modules, ~1.9k LOC — including a parallel-in-time trainer whose
calibrated gates fail silently) and `src/mcts/search.py`/`node.py`/`evaluator.py` (~1.2k LOC — the
core search engine, where the F0 defect lived, has no logger at all). Highest-value single line:
recording `search_mode`/`invert_backup` at `MCTS.__init__` — the entire `lshape_amr_compare`
headline moved from 0.86 to 0.96 on that one boolean and there is no runtime record of which mode
ran. Note `src/pde/stochastic/` is guarded by an import-isolation allowlist; extend it deliberately
in the same commit.

#### P3 — Reusability / BC (evidence recorded, no action taken)

- **Two parallel threshold schemas with different semantics**: `poc/config.py::MetricThreshold`
  (no tolerance on `<=`/`>=`) vs `templates/config.py::MetricDefinition` (adds `1e-9`). The
  `spec-new` skill declares `MetricThreshold` "the single source of truth", yet `src/pde/config.py`
  builds `success_metrics` from the other one. Migrating a `<=` threshold between them silently
  flips pass/fail at the boundary.
- **`FNetMixingLayer` declared twice** with identical bodies; one is labelled "alias for backward
  compatibility" but is a full re-declaration. The O(N) complexity headline runs the copy that no
  test covers.
- **No deprecation convention**: only 2 of ~19 shims warn; `pyproject.toml` has no `filterwarnings`
  at all, so `DeprecationWarning`s are neither errored nor tracked. `distributed/config.py` carries
  a `.. deprecated::` docstring directive with no `warnings.warn` behind it. Owner decision #6 sets
  a 2-release window; nothing implements it.
- **`compute_hash()` re-implemented 6×** without the canonical volatile-field exclusion — no live
  bug today (none of those configs has a timestamp field yet), but the first one added makes a
  reproducibility hash non-reproducible.
- **If-elif dispatch where the repo's own registry pattern fits**: `load_config_from_dict`'s 6
  copy-pasted lazy-import branches with a silent `BaseScenarioConfig` fallback (a typo'd scenario
  name parses instead of raising); basis-kind dispatch at 3 separate sites; 7 byte-identical
  backend-factory tails; `create_sampler` silently defaulting an unknown name to random.

### 7.7.1 Adversarial review of the Phase-1 diff — dispositions (2026-08-15)

A full adversarial pass re-ran every numeric claim in the Phase-1 diff to ground. **All of them
reproduced exactly** (noyron_basis 98%, Noyron HX 99%, SBIR P40 94%, llm_prior pair 81%; every
constant swap value-identical; `DEFAULT_BOARD_SIZES` never leaked by reference), and the
highest-risk item — whether `Trainer.training_config` could ever lack the new fields — was traced
through every construction path and cleared. Fixed in follow-up; remaining items below.

**Fixed** (PR #123): fork-PR concurrency collision; `ci-success` failing on superseded runs; the
v1.1.0 migration's orphan-dict no-op; the `coverage` job's 30-minute cap (measured 24m08s on a real
runner = 80% utilisation, one slow runner from red → raised to 45); four stale "abstract"
doc sites left by the `BaseTrainer` demotion; `evaluate()`'s docstring claiming a `step()` wiring
that does not exist; the 14th `[9, 13, 19]` site in `config/schemas.py` (the original sweep covered
`src/` and `dashboard/` but not `config/`); the `CoverageWarning` class name (it is
`CoverageWarning`, slug `module-not-imported`, not `CovReportWarning`); a stale native-gate count in
the charter guard's own docstring.

**Open, with the evidence needed to act:**

- **Gate duplication.** The Noyron/SBIR/llm_prior gate steps re-run files the job's main
  `pytest tests/` sweep already covers (none of those tests carry `@pytest.mark.slow`), so each is
  measured twice. Raising the timeout bought headroom; collapsing the duplication — `--ignore` the
  gated files from the sweep, or split the two Noyron gates into their own job — is the real fix.
- **`BaseTrainer.evaluate()` has zero call sites.** `step()` drives `generate_data` and
  `compute_loss` only. Demoting it from `@abstractmethod` removed it from
  `scripts/audit_abstractions`' view without resolving it — the F1 pattern one level down. Either
  delete it plus both subclass stubs, or wire an evaluation cadence into `step()`. Its docstring now
  states the truth in the meantime.
- **`GUMBEL_LOG_PRIOR_FLOOR` is documented as an algorithmic knob but shipped as a module
  constant.** By the project's own rule ("every knob a typed field; numerical-stability literals may
  be constants") a value whose docstring says "retuning it changes action selection" belongs on
  `GumbelMCTSConfig`. Promoting it is an API change and wants its own decision.
- **`config.schemas.TrainingConfig` ↔ `BaseTrainerConfig` still declare the same two scheduler
  knobs with different defaults.** Phase 1 collapsed 2→1 on the base-trainer side; the remaining
  split is rooted in `Trainer.__init__` not calling `super().__init__()`. Value-preserving and
  cross-documented, but it is a divergence waiting to be re-discovered.
- ~~**The `.pytest_cache` removal rationale is imprecise.**~~ **RETRACTED 2026-08-15 — this
  correction was itself wrong, and is kept as a worked example of the failure mode this audit
  exists to catch.** The reasoning was: "pytest creates `.pytest_cache/`, therefore the step would
  have found files, therefore 'never found files' is false." The premise is true and the conclusion
  does not follow: `actions/upload-artifact@v4` excludes hidden files by default, and
  `.pytest_cache/` is a dotted path — which is exactly why CI logged "No files were found". A local
  `ls` proving the directory exists was never evidence about the *action's* behaviour. The original
  claim stands; the deletion was correct for the stated reason.

### 7.7.2 SQE pass — findings and dispositions (2026-08-15)

A dedicated test-authoring pass added **91 tests across 9 files** for the Phase-1 surfaces and
measured what the diff had actually left covered. Three findings were defects, not gaps.

**Fixed here:**

- **`COVERAGE_CORE=pytrace` was missing from the documented gate commands.** The failure mode is
  silent *under-measurement*, not an error: the identical `src/training` gate reports **89.53%
  (PASS) with pytrace and 82.45% (FAIL) without**, with `base_trainer.py` at 46% and
  `checkpoint_migration.py` lines marked unexecuted while passing tests assert those exact values.
  Anyone following the Regression Surface table locally got a spurious red that looks like a
  coverage regression they caused. A warning now heads the table. (This also explains a confusing
  82.85% reading during this work — a run whose tree was being mutated underneath it.)
- **`FNetEvaluator._process_policy` returned an all-NaN policy for empty `legal_actions`.** Every
  entry is `-inf`, so the softmax shift `masked.max()` is `-inf` and `(-inf) - (-inf)` is NaN — it
  does not raise, it propagates into MCTS selection as silently corrupt priors. The
  `lm_studio/evaluator.py` implementation that this method's own docstring calls a mirror
  **already guarded it** (`np.max(masked[np.isfinite(masked)], initial=0.0)`), so the two differed
  in exactly the degenerate case that matters. Now aligned; degrades to an all-zero policy. The
  ordinary path is unchanged (verified: still sums to 1.0, illegal actions still 0.0).
- The newly-named `_SOFTMAX_NORMALIZER_FLOOR` guards a divide-by-zero that is in fact
  **unreachable** — `exp(x − max x)` always contains a 1, so the denominator is ≥ 1. It is kept
  (harmless, and the mirror carries it) but it was never the protection it appeared to be; the real
  degenerate case was the NaN above.

**Open, recorded:**

- **`config/` is under no coverage gate at all.** CI measures `--cov=src` and `--cov=dashboard`
  only, so the two new `TrainingConfig` scheduler fields — and every other `config/schemas.py`
  line — are structurally unmeasurable. Adding `--cov=config` (or a native-runner `--include`) is
  a CI decision.
- **`DEFAULT_BOARD_SIZES` is still a mutable `list`.** All 14 sites now copy, and tests prove it,
  but `Final[tuple[int, ...]]` would make the aliasing hazard structurally impossible rather than
  merely tested-against. Deferred because tests compare against list literals.
- **The llm_prior gate ratcheted 85 → 79.** Honest (the old file-path spec enforced nothing, and
  the surface had decayed to 81% unobserved), but a threshold reduction is an owner's call, not a
  side effect of a repair.
- Pre-existing and untouched: `src/training/operator_trainer.py` at 24%, `src/pde/operators.py`
  at 64%.

**Notable about the tests themselves:** the AST guards for the two Gumbel constants were
mutation-checked out-of-tree — swapping the constant names flips every extracted site, and
reintroducing one bare `1e-8` drops the sum-site count from 3 to 2, so both mutants die. Two tests
failed while being written and were *fixed rather than weakened*: one assumed
`visit_counts.sum() == n_simulations` (it isn't), and one built a second scheduler on an optimizer
that already had one (`initial_lr` reuse plus cosine's recursive update) — it now reads the
production `trainer.scheduler`.
