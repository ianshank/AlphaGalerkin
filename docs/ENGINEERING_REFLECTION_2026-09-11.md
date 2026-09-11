# Engineering Reflection & Optimisation Plan — 2026-09-11

> **Status:** proposed plan, not a delivery record. Nothing in this document has
> been implemented by the change that adds it. Every number below was measured on
> default-branch tip `ba03b43` (merge of PR #150) on 2026-09-11; the exact
> command for each is in [Appendix A](#appendix-a--reproduction-commands) so a
> reader can re-measure rather than trust.
>
> **Scope authority:** the project charter
> (`openspec/specs/project-charter/spec.md`) is supreme. This plan proposes work;
> anything that changes scope, a capability, a gate threshold policy, or a
> numeric claim goes through `openspec-change` first. Frozen tracks in
> `config/focus.yaml` (`codec`, `interactive-surfaces`) are respected: work on
> them is scheduled as *separate* changesets, never bundled with core work.
>
> **Relationship to earlier audits:** this is the third hygiene document. The
> [`CODE_HYGIENE_AUDIT.md`](CODE_HYGIENE_AUDIT.md) backlog (B1–B40) and the
> [`CODE_HYGIENE_REVIEW_2026-08-19.md`](CODE_HYGIENE_REVIEW_2026-08-19.md)
> findings are **not** re-litigated here; open items from both are pulled into
> the workstreams below by their existing IDs so there is one live queue, not
> three.

---

## 1. Executive summary

The repository is in materially better shape than the August audits found it:
CI is green on the default branch (16 jobs, `CI Success` gating 10 of them),
`ruff check` + `ruff format --check` are clean across 1,007 files, 43+ per-module
coverage gates are enforced, and a set of hermetic meta-guards now catches the
"check that cannot fail" defect class that produced seven incidents this year.

What remains is a different kind of debt — not *missing* enforcement but
**unmeasured shape**: the code that is green is still concentrated in a handful of
oversized modules, carries a large body of unreachable code, keeps ~370 lazy
in-function imports to dodge layering cycles, and duplicates its CLI, export and
device-resolution helpers per subsystem. None of that is gated, so none of it
ratchets.

The headline numbers, all measured today:

| Dimension | Measured | Enforced today? |
|---|---|---|
| CI on default branch | green, ~19.5 min wall-clock; `Per-Module Coverage Gates` job alone 14 min (44 serial steps) | yes (`ci-success`) |
| `mypy --strict` | **7 errors / 3 files** | **no** (`continue-on-error`) |
| Complexity (`C901`, `PLR091x`) | **162** findings; 17 functions > 50 statements, largest 85 | **no** (rules not selected) |
| Magic-value comparisons (`PLR2004`) | **159** in `src/` | **no** |
| Files > 900 lines in `src/` | **6** (`training/trainer.py` 1,246; `games/chess.py` 1,242; `training/checkpoint.py` 994; `video_compression/codec/codec.py` 976; `research/lshape_amr_compare.py` 942; `training/base_trainer.py` 900) | **no** |
| Dead-code candidates (vulture ≥ 60 %) | **893** (410 unused methods, 55 functions, 39 classes) | **no** |
| Packages with zero inbound `src/` imports | **5** (`prototyping`, `tournament`, `analysis`, `curriculum`, `deployment` — 12,403 LOC) | — (B10, owner decision) |
| Lazy in-function `src.*` imports | **368** across 139 files | **no** |
| Hardcoded `"cuda"` string literals in `src/` | **121** | **no** |
| `print()` calls in `src/` | **131** | **no** |
| `hf_space/src` mirror | 58,315 LOC; **98 of 155** mirrored files diverge from `src/` | frozen (B14) |
| Parked coverage gaps | `templates` 72.5 %, `math_kernel` 61.5 %, `backend` 56 %, `deployment` 27.9 %, `integrations/eval_harness` ungated (B37) | partially |
| Open PRs | 4 dependabot (incl. `actions/checkout` 4→7, `setup-python` 5→7, `gitleaks-action` 2→3), 4 stale feature PRs (Apr–Aug) | — |
| Dependency lockfile | **none** (`>=` floors only) | — |

The plan is nine workstreams sequenced into four phases over roughly one
quarter, each landing as its own reviewable PR under the existing
`pr-preflight` / `regression-surface` discipline. The single most important
design rule carried forward from this year's incidents: **measure, then gate,
then ratchet — never widen a threshold to get green.**

---

## 2. Reflection — what the last two cycles actually taught

Before adding to the backlog, the team looked at what kept going wrong. Five
patterns account for almost every incident recorded in `CODE_HYGIENE_AUDIT.md`,
`CODE_HYGIENE_REVIEW_2026-08-19.md`, and CLAUDE.md's Next Steps table.

1. **The invisibility defect.** Seven times a check existed, ran, and could not
   fail the build (`test-integration`/`test-jax`/`test-chess` outside
   `ci-success.needs`; `tests/e2e/` ignored by every lane but one; `src/backend`
   omitted *and* ungated; `tests/demos/` run by nothing; the Dockerfile built by
   nothing; file-path `--cov=x.py` specs silently dropped; `--cov` targets
   swallowed by `omit`). Every one was found by a human reading YAML. The
   fix that actually worked was **guards that read CI as data**
   (`tests/docs/test_coverage_gate_integrity.py`, `test_e2e_visibility.py`,
   `test_ci_exclusion_ledger.py`, `test_claude_coverage_gates.py`). *Lesson:*
   any new enforcement in this plan ships with a guard that proves it can fail
   (`harden-a-guard` skill; mutation-kill recorded).

2. **"Verified" claims that were false when written.** The tracer pin (B36),
   the "no Dockerfile" row, the eval-harness exemption text (B37), the
   `src/*/constants.py` shims, the fabricated notebook figure. *Lesson:* a
   number or a "verified" in prose is a liability unless a test or an artifact
   pins it. This document therefore cites a command for every figure and makes
   no claim it did not run.

3. **Global singleton state × collection order.** Four defects on one branch
   (`ScenarioRegistry.clear()`, `sys.modules` purges, cached structlog
   loggers, `RefinementSubstrateRegistry`). *Lesson:* registries need a
   `clear`/`ensure` contract (ADR 0005 exists) and tests need subprocess
   isolation when they read process-global state; that is a design smell to
   remove (WS4), not only a test pattern to remember.

4. **Accident-of-environment tests.** Network-fetching "unit" tests (B35), a
   CWD-dependent path-traversal test, and load-sensitive wall-clock ratio
   assertions in the blocking lane. *Lesson:* hermeticity is a property to
   assert (no egress, no CWD, no wall-clock in blocking lanes), not a hope.

5. **Doc drift outruns code drift.** CLAUDE.md is 898 lines of append-only
   log plus a Regression Surface table that has needed three "count was
   hand-maintained and drifted" corrections; CHANGELOG has **two**
   `[Unreleased]` headers (lines 3 and 130) and 1,864 lines; `docs/` carries
   six planning documents, and `mkdocs.yml` excludes 21 entries from the
   site build (22 with this document).
   *Lesson:* prose that must mirror machine state should be generated from it
   or machine-checked; the rest should be shorter.

What went right, and must be preserved: the per-module coverage gates with
`floor(measured) − 2 capped at 85`, the retracted-claims and charter-alignment
guards, the god-file-split recipe (B21, proven on `operators.py`,
`mesh_refinement.py`, `baselines.py`), the scope-containment gate
(`scripts/check_focus.py`), and the habit of recording a retraction *on the
page* rather than deleting it.

---

## 3. Principles for this cycle

- **Backwards compatible by default.** Every module move ships an
  import-compatible re-export and a public-API freeze test (the B21 recipe).
  Deprecations get a `DeprecationWarning`, a CHANGELOG line, and a two-minor-
  release window (owner decision #6 in `CODE_HYGIENE_AUDIT.md` §7.6).
- **One concern per PR.** A split is not a refactor; a gate is not a fix. This
  is what made the B4 slices reviewable and what `check_focus.py` enforces.
- **Measure → gate → ratchet.** A new rule lands with a per-file baseline
  (allowlist) equal to today's violations and a guard that the baseline only
  shrinks. No threshold is widened to get green.
- **Prefer deletion to abstraction.** B17 (delete `get_result`) beat wiring
  it; B36 (delete the pin) beat centralising it. Dead code is removed, not
  re-exported.
- **Frozen tracks stay frozen.** `video_compression`, `dashboard/`, `hf_space/`
  changes are separate PRs behind the `focus-override` label or after the
  freeze lifts.
- **Docs cite commands, not adjectives.**

---

## 4. Workstreams

Each workstream lists: evidence, actions (with the persona that owns them —
`build-engineer`, `sqe`, `reviewer`, `pde-solver`, `mcts-engineer`,
`integration-engineer`, or product/owner), the guard that makes it stick, and
effort (S < 1 day, M 1–3 days, L 1–2 weeks, XL > 2 weeks).

### WS0 — CI: from "green" to "green, honest and fast"

**Evidence.** CI is green, but three things are softer than they look:

- `mypy --strict` reports 7 errors in 3 files and the step is
  `continue-on-error`. Two are *outside* the frozen codec track and one was
  introduced by PR #150 itself (`src/poc/scenarios/stochastic_galerkin_compare.py:63`,
  `arg-type` against the new `SupportsMetricsMapping | SupportsMetricsMethod`
  Protocol) — a soft gate let a regression merge the same week the Protocol
  landed. The third is an `unused-ignore` in `src/training/base_trainer.py:45`.
  The remaining five are in `src/video_compression/codec/codec.py` (frozen;
  already docketed in the 2026-08-19 review as "need a Protocol/Union over the
  entropy-model variants").
- Wall-clock: the last default-branch run took ~19.5 min. The
  `Per-Module Coverage Gates` job is **14 min of 44 serial steps**, each
  re-collecting the suite; the critical path is `lint → test-fast → test-e2e`
  (~8 min for e2e because the chess memory workaround runs two processes
  serially). Twelve jobs each spend ~75–85 s on `Install dependencies`
  (only 3 use `cache: pip`).
- Seven wall-clock ratio assertions still run in the blocking lane and one has
  already failed under load (CLAUDE.md Next Steps).
- No lockfile: every job resolves `>=` floors fresh, which is exactly the
  "flakiness is torch-version-dependent" reason given for keeping mypy soft.
- The fast lane is not hermetic. Re-run locally in an egress-blocked sandbox
  with the exact CI selection (`make test-fast` invocation, Appendix A):
  **10,028 passed, 293 skipped, 2 failed in 6m03s** — the two failures are
  `tests/video_compression/unit/test_loss.py::TestCompressionLoss::{test_perceptual_loss_enabled,test_total_combines_all_losses}`,
  each with `OSError: Tunnel connection failed: 403 Forbidden` from the VGG16
  ImageNet weight download. This is B35 reproduced verbatim: CI is green only
  because GitHub runners have egress.

**Actions.**

| # | Action | Owner | Effort |
|---|---|---|---|
| 0.1 | Fix the two non-frozen mypy errors (`base_trainer.py:45` unused ignore; `stochastic_galerkin_compare.py:63` — make `MultiSeedStochasticComparison` satisfy the Protocol or narrow `_record_metrics`'s parameter). Add `src.video_compression.codec.codec` to a *documented* mypy override block with the B-item it is waiting on, so `mypy --strict` exits 0. | pde-solver / reviewer | S |
| 0.2 | Flip `Run MyPy type checking` from `continue-on-error: true` to hard, **after** 0.3 lands (the stated reason for softness is dependency drift). Guard: `tests/docs/test_ci_exclusion_ledger.py`-style assertion that no `continue-on-error` exists in `lint` except the `src/backend` audit step. | build-engineer | S |
| 0.3 | Introduce a lockfile for CI (`uv lock` / `pip-compile` from `pyproject.toml`, one per Python minor in the matrix), consumed by every `Install dependencies` step; dependabot keeps bumping the floors and the lockfile together. This is what makes 0.2 non-flaky and makes coverage numbers reproducible across runners. | build-engineer | M |
| 0.4 | B7: a composite action (`.github/actions/setup-python-env`) for the 12× checkout/setup/cache/install preamble; `cache: pip` everywhere, keyed on the lockfile. Preserve every `- name:` string — the charter's Quality-Gate guard parses them. | build-engineer | M |
| 0.5 | Parallelise `coverage-gates` as a matrix over gate groups (e.g. 4 shards of ~11 steps) — the 14-min job becomes ~4 min and the run's wall-clock drops below 12 min. Guard: `test_documented_gates_are_enforced_in_ci` must still find every gate (it parses step names, not job names — verify before sharding). | build-engineer | M |
| 0.6 | Move the seven wall-clock ratio assertions (`tests/benchmarks/`, `tests/integration/test_model.py:172`, `tests/e2e/test_sbir_demo.py:167,180`, `tests/demos/test_benchmark_demo.py`) to *complexity-scaling* assertions (fit the exponent over N, assert the slope) and keep one median-based timing check in a dedicated non-contended `benchmark` job. This is the option CLAUDE.md marks "real work, and what the O(N log N) claim actually needs". | sqe | M |
| 0.7 | B35: mock `_get_vgg` in the two ImageNet-downloading "unit" tests and add a `network` marker deselected from every blocking lane; add a hermeticity guard that fails any test in the fast lane that opens a socket (pytest-socket, or a `conftest.py` autouse that patches `socket.socket`). | sqe | S |
| 0.8 | PR hygiene: merge the four dependabot PRs (#103, #135, #136, #138/#139) in one sitting; close or rebase the four stale feature PRs (#48 Apr, #57 Apr — B28 already superseded its gate change, #118 Aug, #128 Aug) with a one-line disposition each. The monthly stale-PR Routine (B23) reports; a human closes. | owner | S |
| 0.9 | Root `conftest.py` / `pytest.ini`: `-p no:cacheprovider` is already used ad hoc; standardise `--strict-markers --strict-config -ra` and `filterwarnings = error::DeprecationWarning` for `src.*` so our own shims cannot rot silently (see WS3.6). | sqe | S |

**Done when:** `ci-success` needs every hard job and `lint` has no soft step;
a default-branch run completes in < 12 min; the fast lane cannot open a socket;
a lockfile is committed and dependabot updates it.

### WS1 — God-file reduction

**Evidence.** Six `src/` modules exceed 900 lines and 29 exceed 650. Three
B4 splits have already landed via the B21 recipe (`operators.py`,
`mesh_refinement.py`, `baselines.py`) with public-API freeze tests and
byte-identical mypy error sets, so the method is proven. Nothing stops a file
from growing back: there is no size guard.

The 17 functions over 50 statements (the largest: `research/baselines/amr.py:284`
at 85, `training/trainer.py:771` at 78 and `:155` at 75,
`experiments/train_physics.py:241` at 72, `research/baselines/navier_stokes.py:44`
at 70) are the seams the splits should cut along — a "god file" is usually two
or three god *functions* with helpers attached.

**Actions.**

| # | Action | Owner | Effort |
|---|---|---|---|
| 1.1 | Land a **module-size budget guard** first: `tests/docs/test_module_size_budget.py` reads `config/module_size_budget.yaml` (ceiling 600 lines for new modules; an explicit allowlist of today's 29 over-ceiling files with their current line counts). A listed file may shrink or stay; growing past its recorded count fails; an unlisted file over the ceiling fails; an allowlist entry for a file now under the ceiling fails as stale. Mutation-kill all three. | sqe / build-engineer | S |
| 1.2 | Split `src/training/trainer.py` (1,246) → `training/trainer/` (`loop.py` for `train_step`/`train`, `checkpointing.py`, `tournament.py` for `_run_checkpoint_tournament` — which per B10 re-implements `src/tournament`; decide wire-or-delete there, `hooks.py`). Public-API freeze test; mypy override glob; CLAUDE.md training gate unchanged. | sqe (trainer is training infra, not solver) | L |
| 1.3 | Split `src/training/checkpoint.py` (994) → `checkpoint/{io,migration,manager,safety}.py`; the `SAFE_CHECKPOINT_GLOBALS` window and `_SAFE_GLOBALS_LOCK` move *together* and `tests/security/test_checkpoint_safety.py` stays byte-identical in intent. | reviewer (security-sensitive) | M |
| 1.4 | Split `src/games/chess.py` (1,242) → `games/chess/{board,moves,encoding,rules}.py`. Owner-flagged as "parked" in B4; the chess E2E memory leak (CLAUDE.md Next Steps) is a strong reason to make the module navigable before root-causing it. | mcts-engineer | L |
| 1.5 | Split `src/research/lshape_amr_compare.py` (942) → `research/lshape_amr/{harness,arms,metrics,export}.py`; fold its `export_csv`/`export_plot` into WS3.2's shared exporter at the same time. | pde-solver | M |
| 1.6 | `src/training/base_trainer.py` (900), `src/modeling/model.py` (781), `src/pde/games/basis_selection.py` (800), `src/agents/config.py` (750): reduce below 600 by extracting the god functions named above (not by re-packaging) — each is one PR. | respective owner | M each |
| 1.7 | Add the same budget to `tests/` (ceiling 1,000; today 8 test files exceed it, largest `tests/pde/test_operators.py` at 1,717) with a **separate** allowlist — test files split by class are trivially safe and make failures attributable. | sqe | S then M |

**Done when:** no `src/` module exceeds 900 lines, the allowlist has shrunk by
≥ 10 entries, and the budget guard is mutation-killed.

### WS2 — Complexity, hardcoded values, observability

**Evidence.** `ruff` runs a strong style set (`E F W I N D UP ANN B C4 SIM`)
but **no** complexity or magic-number rule; opting them in reports 162 + 159
findings. `PLR2004` clusters where the domain logic lives (`games` 29,
`research` 23, `poc` 23, `pde` 21, `video_compression` 20 — the last frozen).
Beyond ruff's reach: 121 `"cuda"` string literals (four different device
resolvers exist: `src/device.py::resolve_device`, the `src/poc/device.py`
shim, `video_compression/perf/device.py`, `video_compression/zoo/cli_helpers.py`),
34 hardcoded `outputs/…`/`checkpoints/…`/`results/…` path literals, 4 URL/port
literals, and 131 `print()` calls in library code while the project standard is
structlog. The 2026-08-19 review's "flagged, not fixed" list (RBF centres on a
`[0,1]` assumption, `cost = 1.0` decoupled from `cost_per_dof`, `level < 2`
hp switchover, `0.1` obstacle floor) is still open.

**Actions.**

| # | Action | Owner | Effort |
|---|---|---|---|
| 2.1 | Enable `C901`, `PLR0911/12/13/15` and `PLR2004` in `[tool.ruff.lint]` with a generated per-file baseline in `per-file-ignores` **equal to today's violations** (`ruff check --add-noqa` is *not* used — inline `noqa` hides the count; a per-file entry is visible and greppable). Thresholds via `[tool.ruff.lint.mccabe] max-complexity` / `[tool.ruff.lint.pylint] max-args/max-statements/max-branches` set at the current maxima so day one is green, then ratcheted with each WS1 split. Guard: a test that the baseline only shrinks (parse `pyproject.toml`, compare to a committed snapshot). | build-engineer | S |
| 2.2 | `PLR2004` set `allow-magic-value-types = ["int", "str"]` for `src/` *only where the literal is an index or a token*; every remaining float goes to `src/constants.py` (already canonical) or a Pydantic field via the `surface-hardcoded-value` skill — value-identity checked. Close the 2026-08-19 "flagged" list in the same sweep (`pde-solver` owns the four PDE-game items). | pde-solver / mcts-engineer | M |
| 2.3 | Device literals: one resolver. Promote `src/device.py` as canonical (B1 — `src/poc/device.py` is already a 19-line shim), route the two codec resolvers through it *when the codec freeze lifts* (separate PR), and replace bare `"cuda"` in `src/` with `resolve_device(...)` calls. Guard: `tests/regression/test_device_literals.py` counts `"cuda"` literals under `src/` outside `src/device.py` and asserts the count only shrinks from a recorded baseline. | integration-engineer | M |
| 2.4 | Path literals → typed `OutputPathsConfig` (Pydantic) with an `ALPHAGALERKIN_OUTPUT_ROOT` env override; scripts stop writing `outputs/<name>` inline. | build-engineer | M |
| 2.5 | `print()` → structlog in library code (`src/`), leaving `scripts/` CLIs and docstring examples alone; add `T201` to ruff for `src/**` with the same shrinking baseline. | sqe | S |
| 2.6 | Exception-handling audit: the `src/deployment/validate.py` / `verify_invariance.py` cases were adjudicated as propagating, but 76 `except ImportError` optional-dependency guards exist with no shared helper; introduce `src/core/optional.py::require_extra("fem")` returning a typed sentinel and a uniform `MissingExtraError` message, then migrate the guards. | integration-engineer | M |

**Done when:** complexity/magic-number rules are selected in CI with a
committed baseline that has shrunk ≥ 25 %; one device resolver serves
`src/` outside the frozen track; no `print()` in `src/` outside allowlisted
CLIs.

### WS3 — Dead, redundant and duplicated code

**Evidence.** `vulture --min-confidence 80` finds 14 hits, all unused
variables (mostly `exc_tb` in `__exit__` signatures — noise). At 60 %
confidence it finds **893** candidates: 410 unused methods, 311 variables,
55 functions, 46 attributes, 39 classes, 32 properties, concentrated in
`video_compression` (112, frozen), `research` 80, `poc` 78, `games` 72,
`distributed` 62, `training` 59, `pde` 57, `agents` 55. Five packages have
zero inbound `src/` importers (B10). Known duplicates: `export_csv`/`export_plot`
defined 4× with identical shape across the `*_compare` harnesses; `resolve_device`
3×; nine `--allow-unsafe-pickle` argparse blocks; three near-identical
`SAFE_*_GLOBALS` allowlists; 22 argparse `main()`s in `scripts/` that do not
use `src/templates/cli.py` (B15); the `lm_studio` YAML block copy-pasted across
six scenario configs (B11); and the `hf_space/src` mirror where 98 of 155 files
have diverged from `src/` (B14). ~30 back-compat shim sites carry no
`DeprecationWarning` at all (the grep for `warnings.warn(...Deprecat...)` in
`src/` returns **0**), so none can ever be retired on evidence.

**Actions.**

| # | Action | Owner | Effort |
|---|---|---|---|
| 3.1 | Wire `vulture` into the `lint` job **report-only** with a committed `vulture_whitelist.py`, then triage the 60 %-confidence list package by package: delete confirmed-dead code (with the abstraction audit re-run), whitelist the dynamically-reached (registry-decorated, Protocol members, `__exit__` args). Ratchet: the whitelist may only shrink; flip to blocking when the un-whitelisted count is 0. | sqe + package owner | M then ongoing |
| 3.2 | Consolidate the four `export_csv`/`export_plot` pairs onto `src/poc/scenarios/_compare_common.py` (B2 already created the base) with a `Protocol` for the row shape; keep the four names as thin re-exports for one release. | pde-solver | S |
| 3.3 | One `add_unsafe_pickle_argument(parser)` helper in `src/training/checkpoint` (post-WS1.3) replacing the nine argparse blocks; one `SAFE_GLOBALS` builder parameterised by module, with the reflection test that already guards `SAFE_CODEC_GLOBALS` extended to all three. | reviewer | S |
| 3.4 | B15: migrate the three `run_*_compare.py` scripts and the SBIR runners onto `src/templates/cli.py` (which is 0 %-covered *because* nothing uses it — WS5.1 measures it once this lands). | build-engineer | M |
| 3.5 | B10 owner decision (register #1 in `CODE_HYGIENE_AUDIT.md` §7.6), executed as one `openspec-change` amending the scope register: recommended default — **delete** `prototyping` (3,200 LOC) and `analysis` (2,729); **wire** `tournament` (2,628) into `trainer.py`'s `_run_checkpoint_tournament` (WS1.2) or delete; **keep** `curriculum` (1,696) only if WS4 gives it a consumer, else delete; `deployment` (2,150) keep-and-deprecate (CI-exercised by the ONNX suites); `demos` keep (dashboard dependency). Every deletion goes through `tests/support/cut_modules.py::CUT_MODULES` so the charter guard learns it. | owner | M each |
| 3.6 | Deprecation policy made executable: every shim in the ~30-site list gets `warnings.warn(..., DeprecationWarning, stacklevel=2)`, a `deprecated_since` / `remove_in` pair recorded in `docs/migration/deprecations.md`, and WS0.9's `filterwarnings = error` for `src.*` so our own code cannot call a shim. A guard asserts every `DeprecationWarning` in `src/` has a ledger row and every ledger row past `remove_in` has been removed. | reviewer | M |
| 3.7 | B11: extract the shared `lm_studio` YAML block into `config/scenarios/_shared/lm_studio.yaml` loaded via a documented `extends:` key in `load_config_from_dict`; the `llm_prior_*` triplets become 4-line overrides. | integration-engineer | S |
| 3.8 | B14 decision: `hf_space/` is a deploy bundle, not a second source tree. Recommended: generate it (`deploy_space.py` already exists) from `src/` at release time and stop tracking `hf_space/src/**`; until then `tests/hf_space/test_mirror_guard.py` reports the divergence count (98) and must not grow. Frozen track — separate PR after the freeze lifts. | owner | L |

**Done when:** vulture runs in CI with a shrinking whitelist; the four
duplicate families above are single-sourced; B10 is decided and executed; every
shim warns and has a removal date.

### WS4 — Enterprise organisation and layering

**Evidence.** `src/` has 28 packages at one flat level with no tiering; the
`poc ↔ research` cycle (B1) is dodged by **368 lazy in-function `src.*`
imports across 139 files** (a lazy import is a layering violation with a
comment). Six registries exist (`ScenarioRegistry`, `GameRegistry`,
`PDEOperatorRegistry`, `RefinementSubstrateRegistry`, `SOLVER_REGISTRY`,
`BackendProfile` registry, plus `src/templates/registry.py` and
`src/core/registry.py` — two *base* registries), each with its own
`clear()`/`ensure` semantics, which is the direct cause of the four
collection-order defects in §2. Configuration is split between **136**
Pydantic config classes and **186** `@dataclass` sites; two base config
classes (`BaseScenarioConfig`, `BaseModuleConfig`) and two logging helpers
(`poc/logging.py`, `templates/logging.py`) coexist (B5). Of the 25 scripts in
`scripts/`, 22 are hand-rolled argparse and 2 use Hydra; the one typed CLI
helper (`src/templates/cli.py`) is used by `src/agents/cli.py` alone.

**Actions.**

| # | Action | Owner | Effort |
|---|---|---|---|
| 4.1 | Declare tiers in `ARCHITECTURE.md` and enforce them as import contracts (extending `tests/regression/test_import_contracts.py`, which already has the AST helpers): **L0 core** (`core`, `constants`, `seeding`, `device`, `templates`, `math_kernel`), **L1 domain** (`pde`, `refinement`, `mcts`, `games`, `modeling`, `physics`), **L2 engines** (`training`, `research`, `agents`, `distributed`, `integrations`), **L3 apps** (`poc`, `experiments`, `tools`, `demos`, `deployment`, `scripts/`, `dashboard/`). Rule: imports point downward only. Today's violations become the contract's exemption list with a reason each, and the list may only shrink. | reviewer | M |
| 4.2 | B1: promote `src/device.py` fully (it exists; 17 sites still import the `poc.device` shim) and break the `poc ↔ research` cycle so the lazy imports in those two packages can become module-level. Guard: a test counting lazy `src.*` imports under `src/` against a recorded baseline that only shrinks. | reviewer | M |
| 4.3 | B3: one registry base (`src/core/registry.py` — the newer, Protocol-typed one; retire `src/templates/registry.py` with a shim), with a single `snapshot()/restore()` contract per ADR 0005 and a root `conftest.py` fixture that uses it — replacing the nine heterogeneous `tests/poc/` registry-clear fixtures (B16). `ScenarioRegistry` migrates last. | mcts-engineer (owns `src/core/protocols.py`) | L |
| 4.4 | B5: one config base (`BaseModuleConfig`) and one logging module (`src/core/logging.py`), each with `compute_hash()` stability asserted before/after; `poc/logging.py` and `templates/logging.py` become shims. Migrate `@dataclass` configs that are user-facing (loaded from YAML/CLI) to Pydantic; leave pure value objects as dataclasses — write the rule down in `ARCHITECTURE.md`. | build-engineer | L |
| 4.5 | CLI: register console entry points in `pyproject.toml` (`[project.scripts] alphagalerkin-poc = "src.poc.cli:main"` etc.) so `scripts/` shrinks to thin wrappers over `src/templates/cli.py` (WS3.4); document the surviving `python -m` forms in one table. | build-engineer | M |
| 4.6 | Package naming: `src` is imported as a top-level package name (`from src.pde import …`), which collides with any other project on `sys.path` and is the root cause of the `hf_space/src` shadowing that forces `tests/dashboard/conftest.py` to mock. Spike the cost of renaming to `alphagalerkin/` with a `src` shim (`sys.modules` alias) — **decision only this cycle**, the rename is a v1.0 item. | owner | S (spike) |

**Done when:** tier contracts are enforced with a shrinking exemption list;
lazy-import count is below 200; one registry base, one config base, one
logging module; console entry points exist.

### WS5 — Coverage: close the parked gaps, measure the guards

**Evidence.** The 85 % global gate and 43+ per-module gates hold. Parked
packages (`CODE_HYGIENE_AUDIT.md` §7.4): `src/templates` 72.5 % (`cli.py` at
0 % — live code with no direct test), `src/math_kernel` 61.5 % (the `Jax*`
classes are exercised by nothing, including the `test-jax` job), `src/backend`
56 % (`logging.py` and `rng.py` at 0 %), `src/deployment` 27.9 % (626
statements, 416 missed — the largest under-tested surface in `src/`),
`src/integrations/eval_harness` 914 LOC gated by nothing (B37 — the extra is
installed in `test-extras` and the tests are not selected). Guards are
mutation-tested by hand and the kill is recorded in prose; nothing re-runs
those mutations.

**Actions.**

| # | Action | Owner | Effort |
|---|---|---|---|
| 5.1 | `src/templates/cli.py`: tests land with WS3.4 (first real consumer); gate `src/templates` at `floor(measured) − 2` via the `add-coverage-gate` skill. | sqe | S |
| 5.2 | `src/math_kernel`: extend `test-jax` to select the `Jax*` classes (a `jax` marker exists); gate at measured − 2 in `test-jax`, using the inline-coveragerc technique because the package is not in `omit` but the JAX arcs need the extra. | sqe | M |
| 5.3 | B37: add the `eval_harness` per-module gate to `test-extras` (the extra is already installed there) and select `tests/integrations/eval_harness/`. Remove the disclosed-gap exemption from `test_coverage_gate_integrity.py` the same PR. | integration-engineer | S |
| 5.4 | `src/backend`: test `logging.py`/`rng.py` (0 %), raise the gate from 54 to measured − 2; decide JAX-backend fate with WS3.5. | sqe | M |
| 5.5 | `src/deployment`: raise from the 25 tripwire to ≥ 60 with real ONNX-mocked tests for `export_onnx.py`/`quantize.py`/`runtime.py` (or deprecate per WS3.5 and gate what remains). | sqe | L |
| 5.6 | Guard mutation testing made repeatable: a `make mutate-guards` target running `mutmut` (or a small in-repo mutator) over `tests/docs/` and `tests/regression/` guard *subjects* with the planted defects recorded in each guard's docstring; report-only, monthly via the existing Routine. | sqe | M |
| 5.7 | Ratchet policy (owner decision #4): +2 per quarter on every gate below 85 until it reaches 85; a guard compares each gate to the recorded ratchet schedule so a missed quarter is visible. | owner | S |

**Done when:** no `src/` package is ungated; every gate below 85 has a
scheduled ratchet; guard mutations are re-runnable.

### WS6 — Hardening

**Evidence.** Checkpoint deserialisation is closed (`weights_only=True`
chokepoint, allowlist window, containment-before-existence). Still open from
the 2026-08-19 review: `src/distributed/worker.py:429,464` unpickles
peer-rank bytes from `all_gather`; SIGINT/SIGTERM handling for the training
stack is undesigned (`torch.multiprocessing.Pool` in `self_play.py`);
`.gitleaks.toml` allowlists all of `tests/`, `docs/`, `config/baselines/`;
`tests/security/test_config_injection.py::test_path_traversal_in_config`
passes by CWD accident; no SAST beyond gitleaks; no dependency vulnerability
scan; no SBOM; `requires-python >= 3.10` with unbounded upper floors.

**Actions.**

| # | Action | Owner | Effort |
|---|---|---|---|
| 6.1 | Replace `pickle` on the `all_gather` path with a schema'd payload (`msgpack` or a fixed `torch.save(weights_only=True)`-compatible dict of tensors); regression test that a marker payload cannot execute. | mcts-engineer (owns self-play) | M |
| 6.2 | Graceful shutdown design spike for training: signal handler sets a flag checked per step *and* terminates the self-play pool; emergency checkpoint on the way out (the P0 emergency-checkpoint path already exists). | sqe | M |
| 6.3 | Narrow `.gitleaks.toml` to `tests/fixtures/**` and `config/baselines/**` only, in its own PR (the audit already explains why splitting from wiring matters). | build-engineer | S |
| 6.4 | Add `pip-audit` (or `uv audit`) against the WS0.3 lockfile and `bandit -ll` on `src/` as `secrets`-job siblings, report-only for two weeks, then blocking; generate a CycloneDX SBOM as a release artifact. | build-engineer | S |
| 6.5 | Rename `test_path_traversal_in_config` to what it checks (`yaml.safe_load` cannot construct objects) and drop the traversal framing; add a *real* containment test at the one place a user path is resolved (`CheckpointManager.load` already has one). | sqe | S |
| 6.6 | Hermeticity guards (from WS0.7): no socket in the fast lane; no CWD dependence (`monkeypatch.chdir(tmp_path)` autouse in `tests/security/`). | sqe | S |

**Done when:** no `pickle.loads` on untrusted bytes anywhere in `src/`;
SAST + dependency audit block the build; the fast lane is provably offline.

### WS7 — Documentation and governance debt

**Evidence.** `CHANGELOG.md` is 1,864 lines with **two** `[Unreleased]`
headers (lines 3 and 130) and no release since `0.4.0-dev` (2026-08-16).
`CLAUDE.md` mixes four things — project context, an append-only milestone log,
the Regression Surface table (the largest single block, ~50 rows of prose-
wrapped commands), and command cookbooks — and its counts have drifted three
times. `docs/` holds six planning documents (`IMPLEMENTATION_PLAN`,
`NEXT_STEPS_PLAN`, `ROI_IMPLEMENTATION_PLAN`, `PLAN_2026-04-27`,
`NEXT_STEPS_REVIEW_2026-08-18`, `GALERKIN_FUSION_HEAD_PLAN`), one marked
historical, and 21 `exclude_docs` entries in `mkdocs.yml`. Ten Markdown files
sit at repo root.

**Actions.**

| # | Action | Owner | Effort |
|---|---|---|---|
| 7.1 | B13: cut `0.4.0` from the current `[Unreleased]` (merge the duplicate header first — a guard that `CHANGELOG.md` has exactly one `[Unreleased]` and that versions are monotonic is a 10-line test). Adopt `RELEASING.md`'s cadence (it exists) as monthly. | owner | S |
| 7.2 | Split CLAUDE.md's Regression Surface table into `docs/regression-surface.md` (generated or guard-mirrored from `ci.yml` step names + commands — `tests/docs/test_claude_coverage_gates.py` already parses the table, so point it at the new file). CLAUDE.md keeps a one-paragraph pointer. This is a governance change (CLAUDE.md's header says it *owns* the table) → `openspec-change`. | owner + build-engineer | M |
| 7.3 | Consolidate planning docs: move the five superseded plans under `docs/archive/plans/` (links fixed by `check_doc_links.py`), leaving `CODE_HYGIENE_AUDIT.md` as the live backlog and this document as the live plan; add a `docs/plans/README.md` index with a "live / historical" column. | owner | S |
| 7.4 | Machine-check the counts that drifted: the "15 skills / 6 subagents / 5 commands" and test-count floors already are; add the same treatment to the `src/` package count in `ARCHITECTURE.md` and the `Regression Surface` row count. | sqe | S |
| 7.5 | `check_doc_links.py` inline-span resolution (CLAUDE.md Next Steps): implement with the proposed ~15-entry allowlist convention, report-only first. | build-engineer | M |

**Done when:** one `[Unreleased]`; a tagged release; CLAUDE.md under 300
lines with the table living beside its guard; every planning doc labelled
live or historical.

### WS8 — Process and agentic workflow

**Evidence.** B22 recorded a real working-tree collision between concurrent
subagents. The harness has 15 skills and 6 subagents validated as executable
configuration (`tests/claude/`), which is good; what is missing is a default
isolation rule and a definition of done that the skills share.

**Actions.**

| # | Action | Owner | Effort |
|---|---|---|---|
| 8.1 | B22: `isolation: "worktree"` is the documented default for any multi-agent dispatch with Bash+git; add the rule to `AGENT.md` and each `.claude/agents/*.md` that runs concurrently. | owner | S |
| 8.2 | One "definition of done" checklist in `CONTRIBUTING.md`, referenced by `pr-preflight`, `wire-a-ci-job`, `add-coverage-gate` and `god-file-split` instead of each restating it: lint/format/mypy clean, regression surface green, guard mutation-killed, CHANGELOG line, CLAUDE.md row only for milestones, no threshold widened. | owner | S |
| 8.3 | `CODEOWNERS` already exists; align its paths with the WS4.1 tiers so reviews route by layer. | owner | S |

---

## 5. Sequencing

Phases are ordered so that each one makes the next cheaper and safer. Nothing
in Phase N+1 starts until Phase N's guard is green on the default branch.

| Phase | Weeks | Contents | Why this order |
|---|---|---|---|
| **P0 — Honest & fast CI** | 1–2 | WS0.1–0.5, 0.7–0.9; WS6.3–6.4 (report-only); WS7.1 | Everything after this lands under a *hard* mypy gate, a lockfile and a < 12-min loop. Cheapest, highest leverage, no owner decisions needed except merging dependabot. |
| **P1 — Baselines & guards** | 2–3 | WS1.1, WS1.7; WS2.1, 2.5; WS3.1 (report-only), 3.6; WS4.1, 4.2 guard; WS5.6 | Establish every ratchet *before* the refactors so progress is measurable and regressions are visible. All additive; no production behaviour changes. |
| **P2 — Splits, dedup, deletions** | 3–7 | WS1.2–1.6; WS2.2–2.4, 2.6; WS3.2–3.5, 3.7; WS5.1, 5.3, 5.4; WS6.1, 6.5–6.6; owner decisions B10 / WS4.6 spike | One module or one duplicate family per PR. B10 executed once decided. |
| **P3 — Structure & governance** | 7–12 | WS4.3–4.5; WS5.2, 5.5, 5.7; WS6.2; WS7.2–7.5; WS8; frozen-track PRs (WS0.1 codec override, WS2.3 codec resolvers, WS3.8) if the freeze lifts | Registry/config unification touches everything, so it goes last, on top of a codebase that is smaller, tiered and gated. |

Dependency notes: WS0.2 (hard mypy) depends on WS0.3 (lockfile). WS1.2
(trainer split) should precede WS3.5's `tournament` decision because the split
isolates `_run_checkpoint_tournament`. WS3.4 (scripts onto `templates/cli.py`)
precedes WS5.1 (gate `templates`). WS4.3 (one registry) precedes the removal of
the subprocess-isolation pattern in `tests/docs/test_charter_alignment.py`.

---

## 6. Targets (baseline → end of cycle)

| KPI | Baseline (2026-09-11) | Target | Guard |
|---|---|---|---|
| CI wall-clock, default branch | ~19.5 min | < 12 min | Actions run duration (report) |
| Soft steps in `lint` | 2 (`mypy`, backend audit) | 1 (backend audit only) | new `test_lint_has_no_soft_gate` |
| `mypy --strict` errors | 7 | 0 (codec via documented override until unfrozen) | hard CI step |
| `src/` modules > 900 lines | 6 | 0 | `test_module_size_budget` |
| `src/` modules > 600 lines | 29 | ≤ 18 | same, allowlist size |
| Complexity findings (`C901`+`PLR091x`) | 162 | ≤ 120 | ruff baseline snapshot test |
| `PLR2004` in `src/` | 159 | ≤ 80 | same |
| Lazy `src.*` imports in `src/` | 368 | < 200 | `test_lazy_import_budget` |
| `"cuda"` literals in `src/` (excl. `device.py`, frozen track) | ~100 | ≤ 20 | `test_device_literals` |
| `print()` in `src/` | 131 | ≤ 10 (allowlisted CLIs) | ruff `T201` baseline |
| vulture 60 % candidates not whitelisted | 893 | 0 (whitelist ≤ 300, shrinking) | vulture CI step |
| Packages with zero inbound importers | 5 | 0 (deleted or wired) | charter scope guard |
| Ungated `src/` packages | 1 (`eval_harness`) | 0 | `test_coverage_gate_integrity` |
| `--cov-fail-under` gates below 85 in `ci.yml` | 13 | each on a ratchet schedule | new ratchet guard |
| Shims without `DeprecationWarning` | ~30 | 0 | deprecation-ledger guard |
| `[Unreleased]` headers in CHANGELOG | 2 | 1 | new changelog guard |
| Open dependabot PRs older than 14 days | 4 | 0 | monthly Routine |

---

## 7. Owner decisions required

These block specific items and are recorded here rather than assumed.

| # | Decision | Blocks | Recommended default |
|---|---|---|---|
| D1 | B10 fate of `prototyping` / `analysis` / `tournament` / `curriculum` / `deployment` | WS3.5, WS5.4–5.5 | delete / delete / wire-into-trainer / delete-unless-consumed / deprecate-keep |
| D2 | Lockfile tool (`uv` vs `pip-tools`) | WS0.3 | `uv` (single tool for lock + audit + install; matches the SessionStart hook's `pip install -e` only if `uv pip` is used there too) |
| D3 | Hard mypy gate (owner decision #3, already recommended "yes after pinning torch") | WS0.2 | yes, once WS0.3 lands |
| D4 | `hf_space/` single-sourcing (B14) — generate vs. keep tracking | WS3.8 | generate at release; stop tracking `hf_space/src/**` |
| D5 | Rename top-level `src` package → `alphagalerkin` (WS4.6) | v1.0 | spike now, execute at 1.0 |
| D6 | CLAUDE.md ownership of the Regression Surface table (WS7.2) | WS7.2 | move table, keep pointer, update charter |
| D7 | Coverage ratchet schedule (+2/quarter) | WS5.7 | adopt |
| D8 | Release cadence | WS7.1 | monthly, cut 0.4.0 now |

---

## 8. What this plan deliberately does not do

- It does not re-argue retracted or corrected claims; the charter and its
  guards own those.
- It does not touch frozen tracks in the same changeset as core work; the
  codec mypy errors, codec device resolvers and `hf_space/` are scheduled as
  their own PRs.
- It does not raise any coverage gate by editing a number; every raise cites a
  measurement.
- It does not propose new product scope. Every workstream is shape, safety,
  speed or clarity of what already exists.

---

## Appendix A — Reproduction commands

All run from the repository root on `ba03b43`, Python 3.12, `pip install -e '.[dev]'`.

```bash
# CI state
#   GitHub Actions run 34476775571 (CI, push, ba03b43): 16 jobs, started 12:26:57Z,
#   ci-success completed 12:46:17Z; Per-Module Coverage Gates 12:29:37Z–12:43:28Z.

# Fast lane, exactly as CI's Unit Tests (Fast) selects it (Makefile CI_TEST_EXCLUDES)
python -m pytest tests/ -m "not slow and not e2e and not gpu_required" \
  --ignore=tests/e2e/ --ignore=tests/integration/ --ignore=tests/demos/ \
  --ignore=tests/training/test_extended_config.py --ignore=tests/notebooks/ \
  --deselect=tests/games/test_chess.py::TestChessEdgeCases::test_invalid_move_notation \
  --deselect=tests/games/test_chess.py::TestChessEdgeCases::test_illegal_move_notation \
  --deselect=tests/training/test_self_play.py::TestParallelSelfPlayWorker::test_generate_games_sequential_fallback_on_error \
  -q --no-header -p no:cacheprovider
#   sandboxed (no egress): 2 failed, 10028 passed, 293 skipped, 36 deselected in 363 s

# Lint / format / type (CI flag set)
ruff check src/ tests/ dashboard/ scripts/ config/ conftest.py deploy_space.py
ruff format --check src/ tests/ dashboard/ scripts/ config/ conftest.py deploy_space.py
python -m mypy src/ --strict --ignore-missing-imports          # 7 errors / 3 files

# Complexity and magic values (rules NOT selected in pyproject.toml today)
ruff check src/ --select C901,PLR0912,PLR0913,PLR0915,PLR0911 --statistics   # 162
ruff check src/ --select PLR2004 --statistics                                # 159
ruff check src/ --select PLR0915 --output-format concise                     # 17 functions > 50 stmts

# Module sizes
find src -name '*.py' -exec wc -l {} + | sort -rn | head -30
find tests -name '*.py' -exec wc -l {} + | sort -rn | head -15

# Dead code
pip install vulture
vulture src/ --min-confidence 80 | wc -l    # 14
vulture src/ --min-confidence 60 | wc -l    # 893
vulture src/ --min-confidence 60 | sed -E 's/.*: (unused [a-z]+).*/\1/' | sort | uniq -c

# Inbound importers of B10 candidates (0 for five packages)
for p in prototyping tournament analysis curriculum deployment demos distributed; do
  grep -rlE "from src\.$p|import src\.$p" src/ dashboard/ scripts/ --include=*.py | grep -v "^src/$p/" | wc -l
done

# Lazy in-function imports
grep -rnE "^\s{4,}from src\.|^\s{4,}import src\." src/ --include=*.py | wc -l           # 368
grep -rnE "^\s{4,}from src\.|^\s{4,}import src\." src/ --include=*.py | cut -d: -f1 | sort -u | wc -l  # 139

# Hardcoded values
grep -rnE "\"cuda\"|'cuda'" src/ --include=*.py | wc -l                                  # 121
grep -rnE "\"(outputs|checkpoints|results|data)/[a-z_]*\"" src/ --include=*.py | wc -l    # 34
grep -rn "^\s*print(" src/ --include=*.py | wc -l                                        # 131
grep -rn "except ImportError" src/ --include=*.py | wc -l                                # 76

# Duplication
grep -rn "^def export_csv\|^def export_plot" src/        # 4 pairs
grep -rn "^def resolve_device" src/                      # 3
grep -rln "allow-unsafe-pickle" scripts/ src/ | wc -l    # 9
grep -l "argparse" scripts/*.py | wc -l                  # 22
# hf_space mirror divergence: identical=48 diverged=98 only-in-mirror=9
for f in $(cd hf_space/src && find . -name '*.py'); do
  [ -f "src/$f" ] && { cmp -s "hf_space/src/$f" "src/$f" && echo same || echo diff; } || echo mirror-only
done | sort | uniq -c

# Shims / deprecations
grep -rniE "back-?compat|backwards.compat" src/ --include=*.py | cut -d: -f1 | sort -u | wc -l   # ~25 files
grep -rnE "warnings.warn\(.*Deprecat" src/ --include=*.py | wc -l                                # 0

# Config style
grep -rE "^class \w+\((BaseModel|BaseModuleConfig|BaseScenarioConfig)" src/ --include=*.py | wc -l   # 136
grep -rc "@dataclass" src/ --include=*.py | awk -F: '{s+=$2} END{print s}'                          # 186

# Docs
grep -n "^## \[Unreleased\]" CHANGELOG.md     # lines 3 and 130
wc -l CHANGELOG.md CLAUDE.md
```

## Appendix B — Cross-reference to the existing backlog

| Backlog ID | Status today | Picked up in |
|---|---|---|
| B1 device promotion / cycle | open | WS2.3, WS4.2 |
| B3 registry consolidation | open | WS4.3 |
| B4 god-file splits | partial (3 of 8 done) | WS1 |
| B5 config/logging unification | open | WS4.4 |
| B6 mypy hard gate | open | WS0.1–0.3 |
| B7 composite action / args file | half (ledger) | WS0.4 |
| B9 seed-stride unification | open | not scheduled (needs baseline re-record; owner call) |
| B10 dead packages | open (owner decision) | WS3.5 / D1 |
| B11 YAML dedup | open | WS3.7 |
| B13 CHANGELOG / release | open | WS7.1 |
| B14 `hf_space` single-source | open (frozen) | WS3.8 / D4 |
| B15 script CLI boilerplate | open | WS3.4, WS4.5 |
| B16 fixture prune / shim policy | open | WS4.3, WS3.6 |
| B21 god-file-split skill | done (skill exists) | used by WS1 |
| B22 subagent worktree isolation | open | WS8.1 |
| B35 network-fetching unit tests | open | WS0.7 |
| B37 eval-harness ungated | open | WS5.3 |
| 2026-08-19 "flagged, not fixed" hardcoded values | open | WS2.2 |
| 2026-08-19 `pickle.loads` on `all_gather` | open | WS6.1 |
| 2026-08-19 SIGINT handling | deferred | WS6.2 |
| CLAUDE.md Next Steps: wall-clock ratio assertions | open | WS0.6 |
| CLAUDE.md Next Steps: `.gitleaks.toml` allowlist | open | WS6.3 |
| CLAUDE.md Next Steps: `test_path_traversal_in_config` | open | WS6.5 |
| CLAUDE.md Next Steps: `check_doc_links.py` inline spans | open | WS7.5 |
| CLAUDE.md Next Steps: second workflow parser | open | fold into WS7.2 (one parser in `tests/support/workflows.py`) |
| CLAUDE.md Next Steps: chess E2E memory leak | open | after WS1.4 (module made navigable first) |
