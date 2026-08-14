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
   generator's template strings, zero bare `except:`, 155 modules on
   structlog vs. 4 legitimate `logging.basicConfig` bootstraps, and 64
   well-scoped `except ImportError` optional-dependency gates. The debt is
   concentrated in **enforcement gaps** (gates that exist but don't run),
   **duplicated boilerplate** (the same lifecycle/config/registry pattern
   reimplemented per-scenario instead of shared), and **layering drift**
   (a `poc` ↔ `research` import cycle papered over by ~101 in-function
   imports).
3. **Reproducibility hazard**: a per-seed derivation constant
   (`_SEED_PRIME_STRIDE`) exists under the identical private name in six
   modules with two different values (1009 in four, 7919 in two). This PR
   deduplicates the *arithmetic* without touching the *values*, because
   unifying the values would change derived seeds and invalidate results
   already committed to `config/baselines/*.json`. The value decision is
   deliberately deferred (§5 B9).

This PR lands 8 low-risk quick wins (§4) and documents 16 larger backlog
items (§5) that were evaluated and intentionally not attempted here —
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
vs. `ls -d src/*/ | wc -l`, 2026-08) despite the root `AGENT.md` describing
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
| `test(poc)` | New `tests/poc/conftest.py` — additive snapshot/restore fixture around `ScenarioRegistry` | `pytest tests/poc`; the order-stress trio (`test_complexity_scenario.py test_registry.py test_cli_commands.py`); `pytest tests/docs/ tests/regression/test_related_work_guard.py tests/hf_space/` |
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
| B1 | Break the `poc`↔`research` cycle: promote `src/poc/device.py` → `src/device.py` (re-export from the old path), migrate `baselines.py`/`comparison.py` and opportunistically the ~101 in-function imports | M | Low | Gate paths in `ci.yml`/CLAUDE.md cite `src.poc.device` at 100% coverage — update in lockstep |
| B2 | `CompareScenarioBase` + Config→Params + name-lock-validator unification across the three `*_compare` triplets | L | Med | The three per-scenario 85%-branch regression surfaces, run together |
| B3 | Registry consolidation onto `src/templates/registry.py` (`ScenarioRegistry` last — most consumers) | L | Med | Charter capability guard (subprocess `ScenarioRegistry` read); PDE end-to-end; MCTS evaluator protocol |
| B4 | God-module splits at the seams in §3.1 — one module per PR, import-compatible `__init__` re-exports mandatory | XL | High | Per-module coverage gates (pde 75, training 85, research 85, mcts 90 branch); mypy's per-module override block names old paths and must move in lockstep |
| B5 | `BaseScenarioConfig`/`BaseModuleConfig` + `poc`/`templates` logging unification | M | Med | Verify `compute_hash()` stability across the merge (it feeds artifact/log identity) |
| B6 | Make the mypy gate deterministic, then flip `continue-on-error` off | L | Low | Decide a `warn_unused_ignores` policy first — the flakiness is torch-version-dependent, not volume-dependent (§1, §3.4) |
| B7 | CI composite action for the ×10 checkout/setup/install preamble + a shared `--ignore`/`--deselect` args file (currently duplicated verbatim between `test-fast` and `coverage`) + `COVERAGE_CORE` centralization | M | Low-Med | Preserve step names — the charter's Quality-Gate guard parses `ci.yml` by `- name:` |
| B8 | Extend the drift guard in §3.4 to CLAUDE.md's Regression Surface table, not just the charter's | M | Med | New test under `tests/docs/`; don't conflict with the existing charter meta-guards |
| B9 | Seed-stride value unification (1009 vs 7919) | S per scenario | Med | Requires re-recording `config/baselines/*.json` via the existing `record-baseline`/`diff` CLI, one scenario at a time |
| B10 | Dead-package decisions: ~14k LOC (15% of `src/`) across `prototyping`, `tournament`, `analysis`, `curriculum`, `deployment`, `demos` have zero inbound `src/` references and are held alive only by their own tests | M each | Med | A cut needs the charter's scope register amended first; a keep-and-wire needs its own PR (e.g. `trainer.py`'s `_run_checkpoint_tournament` re-implements tournament logic instead of importing `src/tournament`) |
| B11 | YAML dedup (the `lm_studio` block copy-pasted across 6 scenario configs; `llm_prior_*` triplets differing by 4/75 lines) plus a decision on the flagged-not-deleted hydra configs (`train_5hr.yaml`, `train_experiment.yaml`, `config/presets/*`, `darcy_poc.yaml`, `transfer_ablation.yaml`) with the `hf_space/config/` mirror in view | S | Low | Demo-YAML validation tests per scenario surface |
| B12 | AGENT.md authoring for the 14 uncovered packages, starting with `src/research` (§3.7) | M | None | `check_doc_links.py` |
| B13 | CLAUDE.md/CHANGELOG restructure — CHANGELOG's `[Unreleased]` section is the large majority of the file; cut a release. CLAUDE.md's milestone log is append-only by its own header, so this is about the drift guard (B8), not moving content | M | Med | `tests/docs/` suite, docs CI workflow |
| B14 | `hf_space/` single-sourcing | — | — | Out of scope for this effort; owned by `tests/hf_space/test_mirror_guard.py`'s tracked follow-up |
| B15 | `scripts/run_{lshape_amr,transfer_baseline_compare,stochastic_galerkin_compare}.py` duplicate CLI/config-loading/baseline-diff boilerplate that `src/poc/cli.py` and `src/templates/cli.py` already provide but no script imports | M | Med | Each script has a dedicated `tests/scripts/` file; shrinks after the dedup |
| B16 | Prune the `tests/poc/` local registry-clear fixtures now that a save/restore wrapper exists (needs per-file analysis of the `sys.modules`-purging ones, §3.6); unify the drifted helix-geometry test fixture; a deprecation-timeline policy for the repo's ~19 back-compat shims; a config-driven `device` field (fail-loud default) for the 3 classic PoC scenarios | S-M | Low-Med | Respective module test suites |

## 6. Rejected or deferred, with reasons

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
