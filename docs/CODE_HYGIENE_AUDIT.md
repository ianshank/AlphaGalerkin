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

## 5. Prioritized backlog

Documented, not implemented. Ordered by suggested sequencing.

| # | Item | Effort | Risk | Notes / guards |
|---|---|---|---|---|
| B1 | Break the `poc`↔`research` cycle: promote `src/poc/device.py` → `src/device.py` (re-export from the old path), migrate `baselines.py`/`comparison.py` and opportunistically the 122 in-function imports (55 files) | M | Low | Gate paths in `ci.yml`/CLAUDE.md cite `src.poc.device` at 100% coverage — update in lockstep |
| B2 | `CompareScenarioBase` + Config→Params + name-lock-validator unification across the three `*_compare` triplets | L | Med | The three per-scenario 85%-branch regression surfaces, run together |
| B3 | Registry consolidation onto `src/templates/registry.py` (`ScenarioRegistry` last — most consumers) | L | Med | Charter capability guard (subprocess `ScenarioRegistry` read); PDE end-to-end; MCTS evaluator protocol |
| B4 | God-module splits at the seams in §3.1 — one module per PR, import-compatible `__init__` re-exports mandatory | XL | High | Per-module coverage gates (pde 75, training 85, research 85, mcts 90 branch); mypy's per-module override block names old paths and must move in lockstep |
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
| B17 | **Resolve the one known dead abstraction, `PDEGame.get_result`** (`src/pde/game.py:457`): declared `@abstractmethod`, documented as lifecycle step 4, implemented by every concrete game — and never called (the `get_result` call sites in `src/training/evaluation.py` / `src/engines/match.py` are the unrelated 1-arg `GameInterface.get_result`). Either wire it into the terminal path or delete it from the contract. It is why `src/pde` is run without `--fail-on-missing` | S-M | Med | `python -m scripts.audit_abstractions src/pde --fail-on-missing` should exit 0 afterwards; changing the `PDEGame` contract means re-running the PDE end-to-end + reward-reachability + clone-isolation surfaces |
| B18 | **Wire the abstraction audit into CI.** `grep audit_abstractions .github/` returns nothing — the F0/F1 screen documented in the Regression Surface is a manual-only step, so a newly-introduced dead abstraction in `src/mcts` or `src/refinement` would not be caught by any automated gate. Add a lint-job step once B17 lands (or add it now gating only `src/mcts src/refinement`) | S | Low | The audit script's own tests (`tests/scripts/test_audit_abstractions.py`) |
| B19 | **Document the 8 optional extras.** `fem`, `viz`, `dev`, `test-extras`, `jax`, `jax-gpu`, `picogk`, `lm-studio`, `docs` are undocumented in `README.md`, `CONTRIBUTING.md`, and `docs/getting-started.md`, all of which show only `pip install -e ".[dev]"`. Separately, `dashboard*` ships in the wheel (`pyproject.toml` `include`) but imports matplotlib **and** gradio at module level with no extra covering it — `pip install alphagalerkin && python -m dashboard.app` fails. Consider a `dashboard` extra | S | Low | None (packaging metadata + prose) |
| B20 | **Add the missing per-module coverage gates.** `src/poc/cli.py`, `src/poc/visualization/*`, the 3 classic scenarios, `src/constants.py` and `src/seeding.py` are covered only by the global 85% gate. Note `CLAUDE.md` documents a `noyron_basis` gate (97%/100%) that **is not wired in `ci.yml`** — the charter guard passes because it asserts documented thresholds ⊆ CI values, not that the gate step exists | S-M | Low | `tests/docs/test_charter_alignment.py`; add the step or drop the claim. **[2026-08-14: CLOSED-WIDER — the phantom-gate class was 3 gates, not 1 (`noyron_basis`, Noyron HX per-module, SBIR P40 per-module: zero `noyron`/`geometry_picogk`/`gpu_profiler` mentions existed in `ci.yml`), plus a fourth, *degraded* gate: under coverage 7.x, file-path `--cov=path.py` specs are silently dropped, so the llm_prior file-level gates enforced nothing. All four wired/repaired in native-runner form; see §7. The `poc/cli`, `visualization`, classic-scenario, `constants`/`seeding` gates remain open (Phase 2c, gate at measured−2).]** |

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
  `--cov=path/to/module.py` specs (only a `CovReportWarning`); the llm_prior
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
