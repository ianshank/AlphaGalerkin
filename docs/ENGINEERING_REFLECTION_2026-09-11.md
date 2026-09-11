# Engineering Reflection & Optimisation Plan — 2026-09-11 (rev 2, peer-reviewed)

> **Status:** proposed plan, not a delivery record. Nothing here is implemented by
> the change that adds it. Every number was measured on default-branch tip
> `ba03b43` (merge of PR #150); the command for each is in
> [Appendix A](#appendix-a--reproduction-commands), or the figure is labelled
> *report-only*. Every figure **corrected in rev 2** after a five-lens peer
> review plus two Copilot reviews of PR #151 is listed with its old and new
> value in [§1](#1-what-the-peer-review-changed).
>
> **Scope authority:** the project charter
> (`openspec/specs/project-charter/spec.md`) is supreme. Items tagged
> `openspec-change` touch a charter Requirement or an accepted-deviation row and
> go through that process first; items tagged `frozen` touch a path in
> `config/focus.yaml` and land as their own PR under the 20-line incidental
> budget or the `focus-override` label.
>
> **Relationship to earlier audits:** `docs/CODE_HYGIENE_AUDIT.md` (backlog
> B1–B40) and `docs/CODE_HYGIENE_REVIEW_2026-08-19.md` are not re-litigated; their
> open items are pulled in by ID ([Appendix B](#appendix-b--backlog-cross-reference))
> so there is one live queue.

---

## 0. One page

**Thesis.** The repository is green and well-guarded but its *shape* is
unmeasured: oversized modules, unreachable code, lazy imports that hide layering,
and duplicated helpers, none of it gated, so none of it ratchets. Meanwhile three
things that most directly protect the honest-benchmark story are softer than
they look: the only job that replays a headline benchmark is a soft gate, the
type checker is a soft gate that let a regression merge this week, and the fast
lane passes only because GitHub runners have network egress.

**Minimum viable cycle — 3 weeks, 14 PR-sized tickets, one concern each.**
A ticket that lands a guard is done when that guard is green on the default
branch and its planted mutation is recorded as killed. A governance or
administrative ticket (R-01; the ledger branch of R-07; R-14) has no mutation
target: it is done when the acceptance column's state is observable on GitHub
or in the tree.

| ID | Week | Ticket | Size | Guard / acceptance | Route |
|---|---|---|---|---|---|
| R-01 | 1 | PR hygiene: merge #136, #138; rebase-merge #103, #135; **close #139** (violates the repo's own dependabot ignore rule); disposition #48, #57, #118 (release or keep ADR 0003), #128 (unblocks the soft ONNX step) | S | zero dependabot PRs > 14 days | direct |
| R-02 | 1 | Hermetic fast lane: add `pytest-socket` to the `dev` extra and register a `network` marker in `pyproject.toml` (`--strict-markers` is on); run the fast-lane steps with `--disable-socket --allow-hosts=127.0.0.1,localhost --allow-unix-socket`; the two VGG-download tests mocked (own PR, ≤ 20 lines) | S | a planted socket-opening test goes red | direct + `frozen` (test file) |
| R-03 | 1 | `mypy --strict` to zero: fix `src/training/base_trainer.py:45` `[unused-ignore]` and `src/poc/scenarios/stochastic_galerkin_compare.py:63` `[arg-type]`; documented override block for `src/video_compression/codec/codec.py` naming its three real `Tensor \| None` defects, with `remove_when: codec freeze lifts` recorded in the WS3.6 ledger and the defects listed in `CHANGELOG.md` `[Unreleased]` so the KPI reads "zero **unsuppressed** errors", not "type-correct" | S | `python -m mypy src/ --strict --ignore-missing-imports` exits 0; the override block names exactly one module and a removal condition | direct (`pyproject.toml` is not frozen) |
| R-04 | 1 | Lockfile: universal `uv.lock` (**remove the `.gitignore:132` entry that ignores it** — the file's own comment at :203 says the project standardises on `uv.lock`); composite action that syncs it and puts `.venv/bin` on `GITHUB_PATH`; `uv lock --check` in lint; Dockerfile `COPY uv.lock`; hook fallback; dependabot `uv` block. The arena sidecar records only six distributions (`src/research/run_manifest.py`), so: pin those six to the sidecar's versions, let `uv lock` resolve the rest of the ~206-package graph, and record the lockfile hash in future sidecars. The CPU torch wheel is selected by the composite action's `--index` for CI/lint only, **not** baked into the universal lock — CUDA hosts keep the default index | M | `uv lock --check` fails on an un-relocked floor bump; guard that `.github/actions/**` runs no `pytest`/`--cov` | direct (ADR for tool choice, D2) |
| R-05 | 1 | Artifact-freeze manifest: sha256 of `results/*` and `config/baselines/*`, checked by a `tests/docs/` test; edits only via `claims-ledger` | S | alter one CSV byte → red | direct |
| R-06 | 2 | Hard mypy gate in CI **and** drop `\|\| true` from `Makefile` `mypy`; fix the now-false `ci.yml:175-177` comment; retire the pre-commit mypy hook or make it the CI invocation | S | `test_lint_has_no_soft_gate` (allowlist: backend audit, ONNX until #128); `MYPY=false make mypy` exits non-zero | `openspec-change` (deviation row :328) |
| R-07 | 2 | Transfer tripwire: three lockfile-pinned re-runs of `transfer-baseline-regression`; if stable, flip to hard in `ci-success`; else ledger it with a reopen criterion | S–M | `ci-success` script has no soft branch, or the ledger row exists | direct / ledger |
| R-08 | 2 | Release: merge the duplicate `[Unreleased]`, changelog guard (one header, monotonic versions), cut `0.4.0`, fix `RELEASING.md` "0.1.0" and `SECURITY.md` "pinned" claims, re-lock | S | guard fails on a second `[Unreleased]` | direct (D8) |
| R-09 | 2 | Scope gate: re-scope `config/focus.yaml`, add `focus` **and** `secrets` to `ci-success.needs` (both green; the `secrets` comment saying it never ran green is stale) | S | new `test_ci_success_hard_gates` (on `tests/support/workflows.py::job_needs`/`hard_gate_jobs`): both jobs in `ci-success.needs` **and** in its `exit 1` block. Event-aware fan-in is part of the ticket: `focus` runs only on `pull_request` and is skipped by the `focus-override` label, while `ci-success` runs on push under `!cancelled()`, so a bare `needs` entry would read `skipped` and fail every post-merge run — the check must accept `skipped` exactly when `github.event_name != 'pull_request'` or the label is present, and the guard asserts that conditional. `test_e2e_visibility` checks only E2E-selecting jobs and cannot see these two | `openspec-change` (deviation row :334) |
| R-10 | 3 | Module-size budget guard (dict allowlist with a *reason* column, ceiling 600; 44 entries today) | S | three planted mutations killed | direct |
| R-11 | 3 | Shape baseline: `scripts/measure_shape.py` emits `config/shape_baseline.yaml` (complexity, magic values, lazy imports, device-resolution sites, prints, orphan modules) **plus the `git rev-parse HEAD:src` tree hash it was generated from**; one `tests/docs/test_shape_baseline.py` asserts actual ≤ recorded, and when actual < recorded requires the recorded tree hash to differ from the current one (a legitimate improvement regenerates the file; an edited number without a source change is rejected) | S–M | raise any recorded count → red; lower a count by hand with `src/` unchanged → red; lower it and regenerate after a real change → green | direct |
| R-12 | 3 | CI critical path, three steps: (a) test jobs `needs` a ruff-only prerequisite instead of `test-fast` (interim ~16 min); (b) 4-shard `coverage-gates` in-file with a shard-assignment guard (~12 min); (c) mypy/audit split out of `lint` (~10.5 min) — the new job joins `ci-success.needs` **and** its `exit 1` block in the same PR, or the R-06 hard gate silently becomes advisory | M | default-branch run **< 12 min** after (b) — the §7 KPI; every gate step names an existing non-empty shard; `test_ci_success_hard_gates` lists the typecheck job | direct |
| R-13 | 3 | `eval_harness` gate in `test-extras` (inline coveragerc, because the package is in `omit`); register an `eval_harness_required` marker and extend the root `conftest.py` hook — which today special-cases only `fem_required` — so the marker hard-fails under `ALPHAGALERKIN_REQUIRE_EXTRAS=1`; replace the 8 module-level `pytest.importorskip` calls with the marker; drop the disclosed-gap exemption | S | `test_coverage_gate_integrity` passes with the exemption removed | direct |
| R-14 | 3 | Worktree-isolation rule for concurrent subagents in `AGENT.md` and the concurrent `.claude/agents/*.md` | S | `tests/claude/` path guard | direct |

**Decisions due** (full register in [§8](#8-owner-decisions)): D2 lockfile tool
(week 1), D3 hard mypy (week 2), D9 focus re-scope (week 2), D10 transfer
tripwire (week 2), D11 artifact freeze (week 1).

---

## 1. What the peer review changed

Five reviewers (adversarial fact-check, CI feasibility, test/coverage,
architecture, product/sequencing) and the Copilot review of PR #151 verified rev 1
against the tree. Rev 1 stated "the exact command for each number is in
Appendix A"; the review found that promise partly false — the same
"verified-but-wrong" pattern §3 warns about. Everything below is now re-measured.

**Measurement corrections.**

| Rev 1 said | Rev 2 (measured) | Consequence |
|---|---|---|
| only 3 of 12 install steps use `cache: pip` | **all 12** do; install still costs 56–84 s because the PyPI torch wheel is the CUDA build | a composite action saves YAML, not seconds; the lever is a CPU torch index + lockfile-keyed venv cache |
| `Per-Module Coverage Gates` (14 min) drives wall-clock | it finishes **2 min 43 s before** `test-e2e`; the critical path is `lint 2.5 → test-fast 8.6 → test-e2e 7.9 = 19.3 min` | sharding alone changes wall-clock by zero; rewire `needs:` first |
| `CI Success` gates 10 jobs | 10 in `needs`, **9 hard + 1 soft** (`transfer-baseline-regression`); `focus`, `secrets`, `test-slow` outside `needs`; a third soft step exists (ONNX suites, `ci.yml:1451`) | R-07, R-09 |
| 0 shims carry a `DeprecationWarning` | **3** do (`mcts/search.py`, `games/interface.py`, `games/chess.py`; multi-line calls the grep missed) of ~28 shim sites | KPI reworded |
| 368 lazy in-function imports / 139 files | **154 / 65** by AST (131 were `TYPE_CHECKING`, 75 were docstrings); 9 genuine cycles, 10 registry ordering, 31 optional/heavy, 15 torch/scipy, **89 with no reason** | target ≤ 60, not < 200 |
| `poc ↔ research` cycle dodged by lazy imports | no module-level cycle; the real SCCs are `{data, training}` and `{pde, experiments, research}` | B1 is a 19-site shim replacement; the cycles that matter are different |
| 121 `"cuda"` literals to fix | 121 total; 46 frozen; of the 75 others **10 are real resolution sites**, ~29 are legitimate `device.type == "cuda"` checks, 5 typed defaults, ~31 prose | guard counts resolution expressions, 10 → 0 |
| 6 files > 900 lines; 29 > 600 | **5** > 900 (`base_trainer.py` is exactly 900); **31** > 650; **44** > 600 (4 frozen codec, 4 demos); 8 test files > 1,000 | R-10 allowlist is 44 rows |
| 9 `--allow-unsafe-pickle` argparse blocks | **5** (2 frozen); the grep counted `.pyc` files | WS3.3 smaller |
| `export_csv`/`export_plot` 4× "identical shape" | signatures and bodies differ; shared part is a 15-line open/header/rows skeleton + a matplotlib guard | dedup is a helper in `src/research/`, not a Protocol in `poc` |
| 4 dependabot PRs | **5** (#103, #135, #136, #138, #139); #139 contradicts `.github/dependabot.yml`'s langfuse-major ignore rule | close #139 |
| 2026-08-19 "flagged, not fixed" hardcoded values still open | **all four fixed** in that review's own Round 2; what is still parked is the modeling LBB `* 10` and FNO `128` (CHANGELOG line 17) | WS2.2 retargeted |
| B4 splits: 3 of 8 done | **5 of 7** (`operators`, `trainer` C2 facade, `baselines`, `mesh_refinement`, `losses/physics`); remaining `chess.py`, `agents/config.py` | WS1 re-planned |
| `_run_checkpoint_tournament` re-implements `src/tournament` | it is a 9-line delegator to `trainer_eval.py`; `src/tournament/AGENT.md` records "no keep-and-wire" | WS1.2 `tournament.py` dropped |
| `src/core/registry.py` is the newer, Protocol-typed base | it has **0 production consumers** and imports `BaseRegistry` from `templates/registry.py` — the module it would replace | WS4.3 direction reversed |
| 6 registries + 2 bases | ~16 registry-like objects; 9 built on `templates.BaseRegistry`; `ScenarioRegistry`/`GameRegistry` hand-rolled; 4 bare dicts | — |
| ungated `src/` packages: 1 | **3** (`templates`, `math_kernel`, `eval_harness`); `backend` measured by nothing under the global gate | KPI corrected |
| `templates/cli.py` at 0 %, no consumer | 0 % under `tests/templates/`; **59 %** under `tests/agents/test_cli.py` (its consumer is `src/agents/cli.py`) | selection artefact |
| `-p no:cacheprovider` "already used ad hoc" | appears nowhere in the repo | dropped |
| `mkdocs.yml` excludes 21 entries (22 with this doc) | 20 → 21 | — |
| `hf_space/src` 58,315 LOC | 58,315 is all of `hf_space/`; `hf_space/src` is 55,038; 98 of 155 mirrored files diverge | — |
| 4 URL/port literals | 8 by a definitional grep | Appendix A |
| 13 gates below 85 | **12** (one hit was inside a comment) | — |

**Design reversals.**

- **WS0.5 sharding is not a speed item.** Rewire `test-e2e`/`test-integration`/
  `test-extras`/`test-slow` from `needs: test-fast` to a ruff-only prerequisite
  (16.3 min), then shard (12.2 min), then split mypy out of `lint` (~10.5 min).
- **WS2.1 baseline cannot live in `per-file-ignores`** (Copilot, sqe): a
  per-file suppression hides *new* findings in a listed file. The baseline is a
  `{(file, rule): count}` table compared against `ruff --output-format json`
  in-test, with a staleness clause. Folded into R-11.
- **WS2.2 must not set `allow-magic-value-types = ["int"]`**: 136 of 159
  findings are ints and they *are* the domain values; that is "widen to get
  green" by configuration.
- **WS3.1 vulture at 60 % is the wrong first tool**: on a 20-candidate sample
  (review-report measurement, not reproduced in Appendix A) 15–25 % were truly
  dead and ~45 % were public API reached only by tests (a product decision).
  Start with the **orphan-module guard** — reproducibly, 24 `src/` modules
  (7,215 LOC) have no importer in `src/`, `scripts/` or `dashboard/`; **21
  (6,274 LOC)** after excluding the three `python -m` CLI entry modules; **2**
  (`backend/logging.py`, `backend/rng.py`, 373 LOC) are imported by nothing at
  all, tests included (Appendix A snippet, both variants) — and vulture at 80 %
  report-only.
- **WS3.6 `filterwarnings = error::DeprecationWarning`** is the wrong mechanism
  for "shims cannot rot": with `stacklevel=2` the warning is attributed to the
  caller, so a `src.*` filter fires only when `src` calls a shim, never for
  tests; and three existing tests call shims without `pytest.warns`. Replaced
  by a ledger-driven test that imports each ledgered symbol under
  `catch_warnings` and asserts the warning.
- **WS4.3 direction reversed** and gated on **D13**: `docs/CODE_HYGIENE_AUDIT.md`
  §6 measured a `snapshot/restore` fixture as a net regression and ADR 0005 chose
  `clear`/`ensure`; rev 1 proposed the rejected design.
- **WS1 is four extractions, not eight package conversions**: `chess.py` is a
  cohesive rules class (extract only the ~230-line move-encoding seam);
  `agents/config.py` is a schema module (length = field count);
  `lshape_amr_compare.py` sits under a charter retirement condition (row :332)
  — investing in a module scheduled for deletion is waste.
- **WS3.5 B10 must cite the charter**: deviation row :335 already records that
  the four packages stay in scope pending a dedicated change; each `AGENT.md`
  says "do not delete in a hygiene PR"; `deployment` is CLI-addressable and
  stays. Deletion = one `openspec-change` + a frozen `hf_space/` scrub.
- **WS4.6 `src` rename is dropped**: `alphagalerkin` is already the console
  script, an entry-point group and `src/alphagalerkin/`; ~4,700 occurrences in
  ~1,850 files; the shim would recreate the `hf_space/src` shadowing it means to
  end. Not before B14.
- **New items the review added**: artifact-freeze manifest (R-05, D11), hash-pin
  protocol for scenarios with committed `.run.json` (§4), transfer tripwire
  hard-or-ledgered (R-07, D10), `focus`/`secrets` into `ci-success` (R-09, D9),
  stray top-level test files mirror guard, duplicated conftest fixtures, chess
  E2E leak measured (D14), `test_path_traversal_in_config` actually *errors*
  from a different CWD (`AttributeError`, a real `load_config` gap),
  `Trainer.__init__` does not call `super().__init__()`, `src/training/evaluation.py:321`
  still calls `torch.load` outside the audited chokepoint.

**Effort.** Rev 1 scheduled ~124 engineer-days into 12 weeks for a repository
where `CODEOWNERS` routes every path to one human reviewer. Agents parallelise
execution, not review. Rev 2 is a 3-week minimum viable cycle plus an ordered
stretch list.

---

## 2. State of the codebase (measured 2026-09-11)

| Dimension | Measured | Gated today? |
|---|---|---|
| CI, default branch | green; 16 jobs; 19.3 min; critical path `lint → test-fast → test-e2e`; `coverage-gates` 13.8 min off-path (44 serial steps); all 12 installs cached, 56–84 s each | 9 hard + 1 soft in `ci-success`; `focus`/`secrets`/`test-slow` outside; ONNX step soft |
| Fast lane, egress-blocked sandbox | 10,028 passed, 293 skipped, **2 failed** (VGG16 download, B35). *Report-only:* during the adversarial review's concurrent measurements the same lane also failed `test_fnet_vs_attention[cpu]` and `test_galerkin_attention_scaling[cpu]`, the load-sensitive ratio assertions CLAUDE.md already records | green only with egress |
| `mypy --strict` | 7 errors / 3 files; 5 in frozen `codec.py`, 1 unused-ignore, 1 `arg-type` introduced by PR #150; `Makefile` masks with `\|\| true`; pre-commit mypy hook is `stages: [manual]` (never runs) | **no** |
| Complexity (`C901`, `PLR091x`) | 162; 17 functions > 50 statements (max 85) | no |
| `PLR2004` | 159 = 136 int + 23 float; 20 frozen | no |
| Module size | > 900: 5 (+1 at exactly 900); > 600: 44; tests > 1,000: 8 | no |
| Dead / orphan code | vulture 80 %: 14 (6 `__exit__` args, 8 real); 60 %: 893 (15–25 % real on a sample, report-only); **21 orphan library modules** (6,274 LOC; e.g. `mcts/gumbel.py` 712, `research/scaling_runner.py` 467, `data/dataset.py` 315); `core.registry.Registry` has no consumer outside `src/core/__init__.py`'s re-export; `backend/logging.py`, `backend/rng.py` imported by nothing, tests included | no |
| Zero-inbound packages | `prototyping`, `analysis`, `curriculum`, `tournament` (charter row :335), `deployment` (CLI-addressable) | charter records disposition |
| Lazy imports (AST, non-`TYPE_CHECKING`) | 154 in 65 files; 89 without a reason; module-level SCCs `{data,training}`, `{pde,experiments,research}` | no |
| Device resolution | 6 named `resolve_device`/`_resolve_device` defs (12 device-resolving functions by a broader survey); 10 ad-hoc `"cuda" if is_available()` sites outside `src/device.py`; 19 consumers of the `poc.device` identity shim | no |
| `print()` in `src/` | 116 (ruff `T201`; 131 by raw grep incl. docstring examples); 99 in modules with a `__main__`; ~32 in library code | no |
| `except ImportError` | 76: 25 guard hard deps (dead), 11 guard first-party imports (hide breakage), ~33 genuine optional | no |
| Duplication | 4 `export_csv`/`export_plot` pairs (skeleton only); 5 unsafe-pickle argparse blocks (2 frozen); 3 `SAFE_*_GLOBALS`; 22 of 25 scripts hand-rolled argparse; `lm_studio` YAML: 8 blocks / 3 contents; `poc/logging.py` vs `templates/logging.py` duplicate `log_timing`/`log_call`/`DebugContext` | no |
| Config surface | 62 `BaseModel` (46 config-like, 16 records), 63 `BaseModuleConfig`, 11 `BaseScenarioConfig`; ~2 user-facing `@dataclass` configs | — |
| Registries | ~16; 9 on `templates.BaseRegistry`; `ScenarioRegistry`/`GameRegistry` hand-rolled; 20 test files call `*.clear()` | ADR 0005 |
| Shims | ~28 back-compat sites; 3 warn; no ledger; no removal dates | no |
| Coverage, parked | `templates` 72.4, `math_kernel` 61.5 (not in `omit`), `backend` 56.5, `deployment` 27.9 (its ONNX suites are soft), `eval_harness` 11 pass / 8 silent skips; 12 gates below 85 | partial |
| `hf_space/` | 58,315 LOC; `hf_space/src` 55,038; 98 of 155 mirrored files diverge | frozen; B14 |
| Docs / release | `CHANGELOG.md` 1,864 lines, two `[Unreleased]`; `CLAUDE.md` 898 lines; 7 planning docs; **no git tag ever**; `RELEASING.md` says 0.1.0 (pyproject 0.4.0-dev); `SECURITY.md` says dependencies are "pinned in `pyproject.toml`" — only `ruff==0.15.8`, a mypy/pydantic range and the eval-harness git SHA are pinned; the resolved graph is not; ADR 0003 reserved for PR #118 | partial |
| PRs | 5 dependabot (#139 must close), 4 stale (#48, #57, #118, #128) | monthly Routine reports |
| Dependencies | no lockfile; `>=` floors; `uv lock --dry-run` resolves 206 packages incl. both jax extras and the git dep in 10 s | no |

---

## 3. Reflection — five lessons, kept short

1. **A check that cannot fail is a report.** Seven invisibility incidents this
   year; every one found by a human reading YAML; every fix that stuck was a
   guard that reads CI as data. Rev 1 of this plan added two more (a soft
   benchmark gate, three jobs outside `needs`) to the list *by omission*.
2. **"Verified" in prose is a liability.** The tracer pin, the Dockerfile row,
   the eval-harness exemption, the fabricated figure — and eighteen numbers in
   rev 1 of this document. Numbers live in `config/shape_baseline.yaml` behind a
   test (R-11), not in prose.
3. **Global singleton state × collection order** produced four defects on one
   branch. ADR 0005's `clear`/`ensure` contract is the recorded answer; do not
   re-propose the rejected one (D13).
4. **Accident-of-environment tests.** Network in unit tests (B35), CWD-dependent
   security tests, wall-clock ratios in blocking lanes. Hermeticity is asserted
   (R-02), not hoped.
5. **Doc drift outruns code drift.** Two `[Unreleased]` headers, a stale release
   version, a stale pin claim, an 898-line CLAUDE.md. Machine-check the counts;
   shorten the rest.

---

## 4. Principles for this cycle

- **Backwards compatible by default**: import-compatible re-exports plus a
  public-API freeze test (B21 recipe); deprecations warn, are ledgered with a
  `remove_in`, and get two minor releases.
- **One concern per PR, ≤ 300 diff lines** where possible; review time is the
  constraint, not execution time.
- **Measure → gate → ratchet; never widen a threshold to get green** (including
  by configuration, e.g. `allow-magic-value-types`).
- **Prefer deletion to abstraction**: `core.Registry`, `backend/logging.py`,
  the 25 dead `except ImportError` guards go before anything is unified.
- **Hash-pin protocol**: any change to a scenario config that has a committed
  `results/*.run.json` (`config_hash`, `packages`) is resolved only by re-run +
  re-record via `run-provenance` and `claims-ledger`, never by editing the pin
  or the sidecar. The artifact-freeze manifest (R-05) makes every PR's rollback
  "revert; manifest proves artifacts untouched".
- **Frozen paths are their own PR** under the incidental budget or
  `focus-override`; recording a frozen file in a *read-only* baseline
  (R-10, R-11) is not a frozen-track change.
- **Docs cite commands.** Every figure in this document is reproducible from
  Appendix A or is labelled as a report.

---

## 5. Workstreams

Columns: **Mission** = protects benchmark honesty / speeds the core loop /
removes a defect class that has bitten / hygiene. **Route** = `direct`,
`openspec-change`, `ADR`, `frozen`. **When** = MVC (R-xx), stretch (order in
§6), deferred, cut.

### WS0 — CI: honest, hermetic, fast

| ID | Item | Mission | Size | Guard | Route | When |
|---|---|---|---|---|---|---|
| 0.1 | mypy to zero incl. documented codec override | defect class | S | mypy exit 0 | direct | R-03 |
| 0.2 | Hard mypy in CI + Makefile + retire manual pre-commit hook; fix `ci.yml:175` comment | defect class | S | `test_lint_has_no_soft_gate` | `openspec-change` (:328) | R-06 |
| 0.3 | Universal `uv.lock`, composite sync action, CPU torch index, `uv lock --check`, Dockerfile/hook/dependabot in lockstep; `phase2-zoo-validation.yml` left alone (frozen) | honesty (reproducible `packages`) | M | lock-check; no-pytest-in-actions guard | direct + ADR (D2) | R-04 |
| 0.4 | Critical-path rewiring: ruff-only prerequisite; test jobs `needs` it; mypy/audit as its own job (`if: != schedule`) | speed | S | interim: run < 16 min on its own; the R-12 KPI (< 12 min) needs 0.5 as well | direct | R-12 |
| 0.5 | 4-shard `coverage-gates` in-file matrix (steps stay in `ci.yml` — the charter and B8 guards read `run:` bodies); shard-assignment guard | speed | M | every gate step names an existing shard | direct | R-12 |
| 0.6a | `benchmark` marker deselected from `test-fast`, `coverage`, Makefile (fourth copy → ledger it); absolute floors in `test_sbir_demo.py:167,180` and `test_mcts_perf.py:71` scaled | defect class | S | `test_ci_exclusion_ledger` | direct | stretch 1 |
| 0.6b | Rewrite the 7 ratio assertions as complexity-scaling fits; dedicated non-contended `benchmark` job | honesty (the O(N) claim) | M | slope assertion with a vacuity check | direct | stretch 2 |
| 0.7 | `pytest-socket` + `network` marker + VGG mocks | defect class | S | planted socket test red | direct + `frozen` | R-02 |
| 0.8 | PR hygiene | — | S | — | owner | R-01 |
| 0.9 | Full fast-lane run under `-W error::DeprecationWarning` (1,118-test sample clean); then a *scoped* filter after the `pytest.warns` audit | hygiene | S | — | direct | stretch 6 |
| 0.10 | Transfer tripwire hard-or-ledgered | **honesty** | S–M | no soft branch or ledger row | direct (D10) | R-07 |
| 0.11 | `focus` + `secrets` into `ci-success` with event-aware skip handling (see R-09); `focus.yaml` re-scope (keep both tracks; `frozen_tracks` min 1) | honesty | S | `test_ci_success_hard_gates` (new) | `openspec-change` (:334) | R-09 |
| 0.12 | ONNX suites hard once #128 lands (`dynamo=False` pin) | defect class | S | soft-gate allowlist shrinks | direct | stretch 4 |

### WS1 — Module size

| ID | Item | Mission | Size | Guard | Route | When |
|---|---|---|---|---|---|---|
| 1.1 | Size-budget guard: dict allowlist **with reason** (44 rows at 600; e.g. `chess.py` "rules class", `agents/config.py` "schema module"); rules: unlisted ≤ ceiling, listed not grown, listed still needed, keys exist, scan non-empty | hygiene | S | 3 mutations | direct | R-10 |
| 1.2 | `training/trainer.py`: extract the six `_create_*` factories + `__init__` wiring to `trainer_setup.py` (pattern of `trainer_eval.py`); periodic hooks of `train()` to methods. `Trainer` deliberately does **not** call `super().__init__()` (documented at `base_trainer.py:13-16`, it drives its own loop) — preserve that contract; the extraction must not add a second initialisation path | hygiene | M | public-API freeze; 12 test files import it | direct | deferred (when touched) |
| 1.3 | `training/checkpoint.py` → `checkpoint/{safety,manager,model_io}.py` (`migration` already separate); patch targets in `tests/security/test_checkpoint_safety.py:462,507` move to `checkpoint.safety.torch.load`; permanent re-exports (frozen scripts import it); route `evaluation.py:321` through the chokepoint in the same PR | defect class (security) | M | `test_checkpoint_safety` intent unchanged | direct | stretch 9 |
| 1.4 | `games/chess.py`: extract move encoding (~230 lines) only; 16 private test refs updated | hygiene | S | chess pipeline gate 80 | direct | deferred |
| 1.5 | `pde/games/basis_selection.py`: extract `basis_library.py` (`BasisFunction` + candidate generation) — where the v2.2 geometry-aware library must live anyway | **core loop** | M | F1/F3 rows green | direct | stretch 8 |
| 1.6 | `base_trainer.py`: optimizer/scheduler factories → `optim_factory.py`, AMP block → `amp.py` (~670 after); `modeling/model.py`: heads/blocks split (cohesive; low priority) | hygiene | M | freeze tests | direct | deferred |
| 1.7 | Tests size budget (ceiling 1,000; 8 rows) | hygiene | S | same guard, second dict | direct | with R-10 |
| ~~1.8~~ | `lshape_amr_compare.py` split | — | — | — | — | **cut** (charter :332 retirement) |

### WS2 — Complexity, hardcoded values, observability

| ID | Item | Mission | Size | Guard | Route | When |
|---|---|---|---|---|---|---|
| 2.1 | Select `C901`, `PLR0911/12/13/15`, `PLR2004`, `T201` with a `{(file, rule): count}` baseline in `config/shape_baseline.yaml` (R-11); thresholds at today's maxima; no `per-file-ignores`, no `allow-magic-value-types` for int | hygiene | S | shrink-only + staleness | direct | R-11 |
| 2.2 | The 23 float magic values → `src/constants.py` / typed fields (`surface-hardcoded-value`); the parked modeling LBB `* 10` and FNO `128`; ints stay in the baseline and shrink with splits. **Hash-pin protocol applies** to any scenario config touched | core loop | M | value-identity assertion | direct | stretch 7 |
| 2.3 | One device resolver: replace the 19 `poc.device` shim imports with `src.device`; route the 10 ad-hoc resolution sites and `backend/torch_backend._resolve_device` through it; `ScenarioResult` receives the resolved device instead of the `poc/registry.py:246` fallback | honesty (`device` field truth) | M | R-11 counts resolution expressions 10 → 0 | direct | stretch 5 |
| 2.4 | Path literals: ~21 inline `outputs/…` → `OutputPathsConfig` + env override; the 13 typed defaults stay | hygiene | M | R-11 | direct | deferred |
| 2.5 | ~32 library `print()` → structlog; "CLI module" = has `__main__` | hygiene | S | `T201` baseline | direct | stretch 10 |
| 2.6 | `except ImportError`: delete the 25 hard-dep guards; replace the 11 first-party guards with explicit checks; `src/core/optional.py::require_extra` for the ~33 genuine ones (precedent `_require_skfem`) | defect class | M | R-11 counts guards | direct | stretch 11 |

### WS3 — Dead, duplicated, deprecated

| ID | Item | Mission | Size | Guard | Route | When |
|---|---|---|---|---|---|---|
| 3.1 | Orphan-module guard (21 rows, allowlist with reason; CLI entry modules and `[project.scripts]`/entry-point targets are not orphans) + vulture 80 % report-only; delete `backend/logging.py`, `backend/rng.py` after triage (the only two modules imported by nothing, tests included); `core.registry.Registry` is re-exported by `src/core/__init__.py` (`__all__`), so its removal is a documented public-API change with a one-release `DeprecationWarning` shim (WS3.6 ledger), not a silent delete | hygiene | S+M | orphan list shrink-only | direct | stretch 3 |
| 3.2 | `write_csv` + matplotlib-guard helper in **`src/research/`** (not `poc`, which would create the `research → poc` edge WS4 forbids); four exporters call it; **byte-identical golden** on committed CSV rows | honesty | S | golden test | direct | stretch 12 |
| 3.3 | One `add_unsafe_pickle_argument(parser)`; one `SAFE_GLOBALS` builder for the 2 non-frozen allowlists; reflection test extended | defect class | S | reflection test | direct (+`frozen` for codec) | stretch 13 |
| 3.4 | Scripts onto `src/templates/cli.py` (three `run_*_compare.py` first); gate `templates` via a native `--include` over `tests/templates/` + `tests/agents/test_cli.py` | hygiene | M | `add-coverage-gate` | direct | stretch 14 |
| 3.5 | B10: one `openspec-change` amending charter row :335 and the scope register; recommended **delete** `prototyping`, `analysis`, `tournament`, `curriculum` (production curriculum is `src/training/curriculum.py`); **keep** `deployment` (CLI-addressable). Execution = `CUT_MODULES` entry + `hf_space/src/<pkg>` scrub (frozen) + gates/rows removal | hygiene | M each | `test_mirror_stays_scrubbed`; charter guards | `openspec-change` + `frozen` | D1; at most two deletions this cycle |
| 3.6 | Deprecation ledger `docs/migration/deprecations.md` (symbol, `deprecated_since`, `remove_in`); guard imports each row under `catch_warnings` and asserts the warning; `remove_in ≤ version` → fail; the ~25 non-warning shims get warnings | hygiene | M | ledger guard | direct | stretch 6 |
| 3.7 | `lm_studio` YAML: check whether the 12-key block equals `LMStudioConfig` defaults after `apply_backend_defaults`; if so **delete** the blocks rather than build `extends:` | hygiene | S | demo-YAML tests | direct | stretch 15 |
| 3.8 | `hf_space/` single-sourcing (B14): generate from `src/` at release; mirror divergence (98) may not grow meanwhile | — | L | mirror guard | `frozen` + `openspec-change` (:326) | deferred (D4) |

### WS4 — Layering and organisation

| ID | Item | Mission | Size | Guard | Route | When |
|---|---|---|---|---|---|---|
| 4.1 | Tiers as an ADR + import contract: **L0** `core, constants, seeding, device, templates, math_kernel, backend`; **L1** `pde, refinement, mcts, games, modeling, physics, data, engines`; **L2** `training, research, agents, distributed, integrations, experiments`; **L3** `poc, tools, demos, deployment, alphagalerkin (facade), dashboard/, scripts/`; B10 packages tiered on decision. Day-one exemption list (12 sites): `pde → research` (5, substrates), `research → experiments` (3, `PhysicsOperator`/`cnn_baseline` are models, tier them L1 or move), `agents → poc` (2, `_centaur_common` should move below `poc`), `training → tools` (2, `SimpleGoGame` belongs in `games`) | core loop | M | contract in `test_import_contracts.py`; exemptions shrink-only | ADR | stretch 16 |
| 4.2 | Lazy-import baseline 154 → ≤ 60: hoist the 89 no-reason ones (intra-package first); break `{data,training}` by moving `Experience` to `src/data`; B1 = the 19 shim sites | hygiene | M | R-11 | direct | stretch 17 |
| 4.3 | Registries: retire `core.Registry` via the 3.1 shim; migrate `ScenarioRegistry`/`GameRegistry` onto `templates.BaseRegistry`; any lifecycle change follows **D13** (ADR 0005 `clear`/`ensure` unless new measurements beat audit §6) | defect class | L | subprocess capability guard | ADR | deferred (D13) |
| 4.4 | Config/logging: unify the two logging modules (real duplicate); migrate the ~2 CLI dataclass configs; **do not** rebase `BaseScenarioConfig` on `BaseModuleConfig` (`name` required, `created_at: datetime` widens the pickle allowlist and destabilises hashes) — share a slim hash mixin | hygiene | M | `compute_hash()` pins; hash-pin protocol | direct | deferred |
| 4.5 | Console entry points: extend the existing `[project.scripts] alphagalerkin` rather than add parallel names; `python -m src.poc.cli` is cited 27× and driven by E2E | hygiene | S | E2E `--help` journeys | direct | deferred |
| ~~4.6~~ | `src` → `alphagalerkin` rename | — | — | — | — | **cut** (name collision; depends on B14) |

### WS5 — Coverage

| ID | Item | Size | Guard | When |
|---|---|---|---|---|
| 5.1 | `templates` gate via native `--include` over both suites (with 3.4) | S | `add-coverage-gate` | stretch 14 |
| 5.2 | `math_kernel`: add `tests/math_kernel/` to `test-jax`, plain `--cov` (not in `omit`), gate at measured − 2 | M | — | stretch 18 |
| 5.3 | `eval_harness` gate + `eval_harness_required` marker | S | integrity guard | R-13 |
| 5.4 | `backend`: delete-first (`logging.py`, `rng.py`), then raise 54 → measured − 2 in the existing inline-coveragerc step (`src/backend/*` is in the global `omit`; a bare `--cov=src/backend` measures nothing — the false-pass class `test_coverage_gate_integrity` exists for); JAX fate with D1 | M | integrity guard | stretch 19 |
| 5.5 | `deployment` 25 → ≥ 60: blocked on 0.12 (soft ONNX step) | L | — | deferred |
| 5.6 | Planted-defect runner: `tests/docs/mutations/<guard>/<n>.patch` + expected-failing nodeid, applied in a `git worktree` (guards resolve `REPO_ROOT` from `__file__`); vacuity guard on an empty dir; mutmut only ever for `tests/support/` | M | runner itself mutation-tested | stretch 20 |
| 5.7 | Ratchet policy: gate := max(current, floor(measured) − 2) capped 85, re-measured quarterly by the monthly Routine (report-only, never calendar-red in `ci-success`) | S | `config/coverage_ratchet.yaml` | D7 |
| 5.8 | Test hygiene the review surfaced: stray top-level `tests/test_{data_generation,operator_training,physics_solvers}.py` mirror guard; 7 duplicated conftest fixture names; inventory of the 20 files calling `*Registry.clear()`; RSS ceiling on the `-k "chess"` E2E half (report first) | S each | — | stretch 21 |

### WS6 — Hardening

| ID | Item | Size | Route | When |
|---|---|---|---|---|
| 6.1 | `src/distributed/worker.py`: **all four** pickle sites — `_serialize_experiences` (:415 `dumps`), `_deserialize_experiences` (:429 `loads`), and `_all_gather_experiences` (:464 `loads`, :467 returns `dumps`) — move to a tensor-only dict via `torch.save` / `torch.load(weights_only=True)`; a marker-payload regression test per public path, so no deserializer or returned byte string is still pickle | M | direct | stretch 9 |
| 6.2 | Graceful-shutdown spike for training (flag per step + pool termination + emergency checkpoint) | M | direct | deferred |
| 6.3 | `.gitleaks.toml`: **after** #135 (v3) and a local full-history `gitleaks detect`, drop `tests/.*` and `docs/.*` (there is no `tests/fixtures/`; 432 test files and full history become in scope) | S ×2 PRs | direct | stretch 22 |
| 6.4 | `supply-chain` job: `pip-audit -r <(uv export --frozen)` (`uv audit` does not exist) + `bandit -ll`, report-only, outside `ci-success`; SBOM once a release workflow exists (R-08 creates none — disclosed) | S | direct | stretch 23 |
| 6.5 | Rename `test_path_traversal_in_config` to what it checks; add the `ValueError` on non-mapping YAML that its CWD accident revealed | S | direct | stretch 6 |
| 6.6 | `monkeypatch.chdir(tmp_path)` autouse in `tests/security/` | S | direct | with R-02 |

### WS7 — Docs and governance

| ID | Item | Size | Route | When |
|---|---|---|---|---|
| 7.1 | Release 0.4.0 + changelog guard + stale claims in `RELEASING.md`/`SECURITY.md` | S | direct | R-08 |
| 7.2 | Regression Surface table → `docs/regression-surface.md`: four guard files read `CLAUDE.md` by path (single constants each); 9 skills cite it; repo-root-relative links must be rewritten for `mkdocs --strict` or the file excluded | M | `openspec-change` (D6) | deferred |
| 7.3 | Planning docs → the existing `docs/archive/plans/`; index with live/historical; this file and the audit stay live | S | direct | stretch 24 |
| 7.4 | Machine-check the `src/` package count in `ARCHITECTURE.md` and the Regression Surface row count | S | direct | stretch 24 |
| 7.5 | `check_doc_links.py` inline-span resolution with the ~15-entry allowlist, report-only | M | direct | deferred |

### WS8 — Process

| ID | Item | Size | When |
|---|---|---|---|
| 8.1 | Worktree isolation rule (B22) | S | R-14 |
| 8.2 | Extend `.github/PULL_REQUEST_TEMPLATE.md`'s checklist (it already is the definition of done) with "guard mutation recorded" and "no threshold widened"; skills reference it | S | stretch 25 |
| ~~8.3~~ | CODEOWNERS by tier | — | **cut** (single owner; `CODEOWNERS` already says so honestly) |

---

## 6. Sequencing

**MVC (weeks 1–3)** is the R-01…R-14 table in §0, in that order. Nothing in
stretch starts before R-04 (lockfile) and R-05 (artifact freeze) are on the
default branch, because both are what make later refactors reversible and
their coverage numbers comparable.

**Stretch (weeks 4–8), in order:** 0.6a → 0.6b → 3.1 → 0.12 → 2.3 → {0.9, 3.6,
6.5} → 2.2 → 1.5 → {1.3, 6.1} → 2.5 → 2.6 → 3.2 → 3.3 → {3.4, 5.1} → 3.7 →
4.1 → 4.2 → 5.2 → 5.4 → 5.6 → 5.8 → 6.3 → 6.4 → {7.3, 7.4} → 8.2; D1
decided by week 4 with at most two deletions executed (`prototyping`,
`analysis`).

**Deferred (next cycle, with the unblocking decision):** 1.2, 1.4, 1.6 (touch-
triggered; the size guard holds the line); 2.4; 3.8 (D4); 4.3 (D13); 4.4; 4.5;
5.5 (after 0.12); 6.2; 7.2 (D6); 7.5; remaining D1 deletions.

**Cut:** 1.8 `lshape_amr_compare` split; 4.6 `src` rename; 8.3 CODEOWNERS
tiers.

Dependencies: R-06 after R-04 (determinism is the stated reason mypy is soft);
R-07 after R-04 (pinned re-runs); R-12 after R-04 (venv cache keyed on the
lock); 3.2 golden before any exporter edit; 4.1 ADR before 4.2 hoisting across
packages; 1.3 carries the `evaluation.py:321` chokepoint fix.

---

## 7. Targets

Guarded by the end of MVC (baseline → target, guard):

| KPI | Baseline | Target | Guard |
|---|---|---|---|
| Soft steps across all jobs | 3 (backend audit, mypy, ONNX) | 1 (backend audit) | `test_lint_has_no_soft_gate` (all jobs, allowlist) |
| Jobs outside `ci-success.needs` that run on PRs | 2 (`focus`, `secrets`) | 0 | `test_ci_success_hard_gates` (new) |
| Soft branches inside `ci-success` | 1 (transfer) | 0 or ledgered | script parse / ledger |
| `mypy --strict` errors | 7 | 0 | hard CI step |
| Fast lane hermetic | no | yes | planted socket test |
| Lockfile | none | `uv.lock` checked | `uv lock --check` |
| Artifacts frozen | no | manifest | `test_artifact_manifest` |
| `[Unreleased]` headers | 2 | 1; tag `v0.4.0` | changelog guard |
| Unlisted `src/` modules > 600 lines | — | 0 | `test_module_size_budget` |
| Ungated `src/` packages | 3 | 2 (`eval_harness` gated) | integrity guard + per-package gate test |
| Default-branch CI wall-clock | 19.3 min | < 12 min | report (Routine) |

Measured by `scripts/measure_shape.py`, gated as shrink-only from R-11 on:
complexity 162, `PLR2004` 159, lazy imports 154, ad-hoc device resolution 10,
library `print()` ~32, dead `ImportError` guards 25, orphan modules 21, shims
without warning ~25, mirror divergence 98.

Not gated this cycle and said so: vulture 60 % count, dependabot age, CI p95.

---

## 8. Owner decisions

| # | Decision | Recommendation | Route | Due |
|---|---|---|---|---|
| D1 | B10 fates, one row per package | delete `prototyping`, `analysis` now; `tournament`, `curriculum` after a one-day spike each; keep `deployment` | `openspec-change` (:335) + `frozen` scrub | week 4 |
| D2 | Lockfile tool | `uv` (universal lock, CPU index source, `uv export` for audit); pip fallback in the hook | ADR | week 1 |
| D3 | Hard mypy | yes, after D2 | `openspec-change` (:328) | week 2 |
| D4 | `hf_space/` single-sourcing | park; frozen; no consumer this cycle | — | — |
| D5 | `src` rename | **dropped**; revisit only after B14 with the `alphagalerkin` collision resolved | — | — |
| D6 | Regression Surface table ownership | defer to a docs-themed cycle; the guards work today | `openspec-change` | — |
| D7 | Coverage ratchet | gate := max(current, floor(measured) − 2) capped 85; quarterly re-measure by the Routine; never calendar-red | direct | week 3 |
| D8 | Release cadence | cut 0.4.0 now; then monthly or at each `openspec/changes/` archive | direct | week 2 |
| D9 | `focus`/`secrets` into `ci-success`; `focus.yaml` re-scope | yes; keep both tracks listed | `openspec-change` (:334) | week 2 |
| D10 | Transfer tripwire | characterise under the lockfile (3 runs); flip hard or ledger with reopen criterion | direct | week 2 |
| D11 | Artifact-freeze manifest | adopt; edits via `claims-ledger` only | direct | week 1 |
| D12 | B9 seed-stride unification | defer; it rewrites `config/baselines/*.json` | — | next cycle |
| D13 | Registry lifecycle | keep ADR 0005 `clear`/`ensure` unless new measurements beat audit §6's table | ADR | before any 4.3 work |
| D14 | Chess E2E leak | `tracemalloc`/RSS spike first (S), not a split | direct | stretch |

---

## 9. Disclosed gaps and non-goals

Already in the repo: `RELEASING.md`, `SECURITY.md`, `CODEOWNERS`,
`CONTRIBUTING.md` + PR template, `docs/adr/` (4 ADRs, 2 CI-enforced),
`dependabot.yml`, gitleaks, `docs/ci-exclusion-ledger.md`, nightly `test-slow`,
a monthly stale-PR Routine.

Still missing after this cycle, stated rather than implied:

- **CI SLOs**: no p50/p95 or flake-rate measurement; add to the monthly Routine.
- **Release automation**: no on-tag workflow, no signed artifacts; R-08 cuts a
  version by hand.
- **Dependency policy**: no upper bounds, license allowlist or update SLA; D2
  plus a `SECURITY.md` paragraph closes most of it.
- **Bus factor**: one owner; nothing repository-side fixes it.
- **Nightly on-call**: `test-slow` at 04:00 UTC has no failure notification.
- **ADR discipline**: tiers (4.1), lockfile (D2), registry (D13) are ADR-shaped
  and routed there; `ARCHITECTURE.md` prose is not a decision record.
- **Branch protection**: whether it requires only `CI Success` is unverifiable
  from the repo; state it in `CONTRIBUTING.md`.

This plan does not re-argue retracted claims, does not bundle frozen-track
work with core work, does not raise any gate by editing a number, and proposes
no product scope.

---

## Appendix A — Reproduction commands

All from the repository root on `ba03b43` after `pip install -e '.[dev]'`.
There is no lockfile yet, so two fresh installs can resolve different
versions; the set these numbers were measured with is Python 3.11.15,
torch 2.14.0+cu130, numpy 2.4.6, scipy 1.17.1, pydantic 2.9.2, coverage 7.16.0,
pytest 9.1.1, ruff 0.15.8 (the CI pin). `uv` and `vulture` are **not** in any
extra — `pip install uv==0.8.17 vulture==2.16` first. CI's `lint` job installs a
pinned minimal set on 3.11, so its mypy count can differ from a local `.[dev]`
env — which is the R-04 argument. Static counts (grep/AST/ruff) do not depend
on the package set; coverage percentages and the mypy count can. GitHub-side
figures (run timings, PR counts) come from the API commands below.

```bash
# CI timeline (GitHub API, run 34476775571 on ba03b43; started_at/completed_at per job)
curl -s https://api.github.com/repos/ianshank/AlphaGalerkin/actions/runs/34476775571/jobs?per_page=30 \
  | python -c "import json,sys;[print(j['name'],j['started_at'],j['completed_at']) for j in json.load(sys.stdin)['jobs']]"
#   16 jobs; lint 12:27:01-12:29:34; test-fast(3.11) 12:29:38-12:38:14;
#   test-e2e 12:38:18-12:46:11; ci-success 12:46:17  -> 19.3 min critical path
#   coverage-gates 12:29:37-12:43:28 (off path); install steps 56-84 s, all with cache: "pip"
# Open PRs (5 dependabot + 4 stale + this one at the time of writing)
curl -s "https://api.github.com/repos/ianshank/AlphaGalerkin/pulls?state=open&per_page=50" \
  | python -c "import json,sys;[print(p['number'],p['user']['login'],p['title'][:60]) for p in json.load(sys.stdin)]"
grep -c 'cache: "pip"' .github/workflows/ci.yml                     # 12
grep -n "continue-on-error" .github/workflows/ci.yml                # 171, 183, 1451
grep -nF 'needs: [' .github/workflows/ci.yml                        # ci-success list (10)

# Fast lane exactly as CI selects it (Makefile CI_TEST_EXCLUDES); egress-blocked sandbox
python -m pytest tests/ -m "not slow and not e2e and not gpu_required" \
  --ignore=tests/e2e/ --ignore=tests/integration/ --ignore=tests/demos/ \
  --ignore=tests/training/test_extended_config.py --ignore=tests/notebooks/ \
  --deselect=tests/games/test_chess.py::TestChessEdgeCases::test_invalid_move_notation \
  --deselect=tests/games/test_chess.py::TestChessEdgeCases::test_illegal_move_notation \
  --deselect=tests/training/test_self_play.py::TestParallelSelfPlayWorker::test_generate_games_sequential_fallback_on_error \
  -q --no-header
#   2 failed (VGG16 download, B35), 10028 passed, 293 skipped, 36 deselected, 363 s

# Lint / format / type
ruff check src/ tests/ dashboard/ scripts/ config/ conftest.py deploy_space.py
ruff format --check src/ tests/ dashboard/ scripts/ config/ conftest.py deploy_space.py
python -m mypy src/ --strict --ignore-missing-imports                # 7 errors / 3 files
grep -n "|| true" Makefile                                           # mypy target

# Complexity / magic values (not selected in pyproject.toml today)
ruff check src/ --select C901,PLR0912,PLR0913,PLR0915,PLR0911 --statistics   # 162
ruff check src/ --select PLR2004 --output-format json | python -c \
  "import json,sys;d=json.load(sys.stdin);print(len(d))"                     # 159 (136 int, 23 float by value)
ruff check src/ --select T201 --statistics                                   # 116 print() (131 by a raw grep that also counts docstring examples)

# Module sizes
find src -name '*.py' -exec wc -l {} + | awk '$1>900 && $2!="total"' | wc -l    # 5
find src -name '*.py' -exec wc -l {} + | awk '$1>=900 && $2!="total"' | wc -l   # 6
find src -name '*.py' -exec wc -l {} + | awk '$1>600 && $2!="total"' | wc -l    # 44
find tests -name '*.py' -exec wc -l {} + | awk '$1>1000 && $2!="total"' | wc -l # 8

# Dead / orphan code
vulture src/ --min-confidence 80 | wc -l      # 14
vulture src/ --min-confidence 60 | wc -l      # 893
# Orphan modules, three variants in one run:
#   production-orphan raw            24 modules / 7,215 LOC
#   production-orphan excl. CLI      21 / 6,274   (the R-11 / WS3.1 figure)
#   orphan incl. tests, excl. CLI     2 / 373     (backend/logging.py, backend/rng.py)
python - <<'EOF'
import ast, sys
from pathlib import Path
sys.path.insert(0, ".")
from tests.support.import_graph import imported_modules, module_name_for, python_files_under
root = Path(".").resolve()
CLI_ENTRY = {"src.agents.cli", "src.poc.cli", "src.tools.cli"}  # python -m / [project.scripts] targets
def importers_for(bases):
    out = {}
    for base in bases:
        for p in python_files_under(root / base):
            names = set(imported_modules(p, root))
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names.update(f"{node.module}.{a.name}" for a in node.names)
            out[module_name_for(p, root)] = names
    return out
def orphans(importers, exclude=frozenset()):
    res = []
    for p in python_files_under(root / "src"):
        if p.name in ("__init__.py", "__main__.py"):
            continue
        m = module_name_for(p, root)
        if m in exclude or any(m in names for who, names in importers.items() if who != m):
            continue
        res.append((m, sum(1 for _ in p.open())))
    return res
prod = importers_for(("src", "scripts", "dashboard"))
for label, found in (
    ("production-orphan raw", orphans(prod)),
    ("production-orphan excl. CLI", orphans(prod, CLI_ENTRY)),
    ("orphan incl. tests, excl. CLI", orphans(importers_for(("src", "scripts", "dashboard", "tests")), CLI_ENTRY)),
):
    print(f"{label}: {len(found)} modules / {sum(n for _, n in found)} LOC")
    for m, n in sorted(found):
        print(f"  {n:5d} {m}")
EOF

# Lazy in-function src.* imports by AST, excluding TYPE_CHECKING blocks -> 154 in 65 files
# (a naive grep '^\s{4,}from src\.' gives 368: +131 TYPE_CHECKING, +75 docstring lines)
python - <<'EOF'
import ast, pathlib
n, files = 0, set()
for p in pathlib.Path("src").rglob("*.py"):
    if "__pycache__" in p.parts:
        continue
    tree = ast.parse(p.read_text(encoding="utf-8"))
    tc = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            tc.update(id(s) for s in ast.walk(node))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)) and id(sub) not in tc:
                    mod = sub.module if isinstance(sub, ast.ImportFrom) else sub.names[0].name
                    if mod and (mod == "src" or mod.startswith("src.")):
                        n += 1
                        files.add(p)
print(n, "in-function src imports in", len(files), "files")
EOF
# The cycle / registry / optional / no-reason classification of those 154 is a review-report
# measurement (transitive module-level graph per site); the module-level SCCs are reproducible
# with tests/support/import_graph.imported_modules and any SCC routine.

# Registry-like objects (heuristic: class *Registry or UPPER_REGISTRY names) -> 10 by this grep;
# the ~16 in §2 adds the create_registry(...) products, counted by hand (report-only)
grep -rhoE "class \w+Registry\b|\b[A-Z_]+_REGISTRY\b" src/ --include=*.py | sort -u | wc -l
# Back-compat shim sites (heuristic: files mentioning back-compat) -> 28 files
grep -rliE "back-?compat|backwards.compat" src/ --include=*.py | wc -l
grep -rn "src.core.registry\|from src.core import" src/ dashboard/ scripts/ --include=*.py \
  | grep -v "^src/core/"                                              # none; src/core/__init__.py re-exports Registry in __all__

# URL / endpoint literals (string defaults, excluding comments)
grep -rnE "\"https?://[^\"]+\"" src/ --include=*.py | grep -v "^\s*#" | wc -l   # 8

# Device
grep -rnE "\"cuda\"|'cuda'" src/ --include=*.py | wc -l                                     # 121
grep -rnE "\"cuda\"|'cuda'" src/ --include=*.py | grep -vc "^src/video_compression/"        # 75
grep -rn "^def resolve_device\|def _resolve_device" src/ --include=*.py | wc -l             # 6 named resolvers (12 device-resolving functions by a broader AST survey)
grep -rlE "src\.poc\.device" src/ scripts/ dashboard/ --include=*.py \
  | grep -vE "^src/(device|poc/device)\.py|^src/video_compression/" | wc -l                 # 19 live shim consumers

# Duplication / guards
grep -rn "^def export_csv\|^def export_plot" src/ | wc -l                     # 8 (4 pairs)
grep -rn '"--allow-unsafe-pickle"' scripts/ src/ --include=*.py | wc -l      # 5 argparse sites (2 in frozen scripts)
grep -rn "except ImportError" src/ --include=*.py | wc -l                     # 76
grep -rn "DeprecationWarning" src/ --include=*.py | grep -v "^\s*#" | wc -l   # 5 lines / 3 sites
grep -rc "lm_studio:" config/scenarios/*.yaml config/agents/*.yaml | grep -v ":0"   # 8 blocks

# Config surface
grep -rE "^class \w+\(BaseModel" src/ --include=*.py | wc -l            # 62
grep -rE "^class \w+\(BaseModuleConfig" src/ --include=*.py | wc -l     # 63
grep -rE "^class \w+\(BaseScenarioConfig" src/ --include=*.py | wc -l   # 11

# hf_space mirror: identical=48 diverged=98 only-in-mirror=9
for f in $(cd hf_space/src && find . -name '*.py'); do
  [ -f "src/$f" ] && { cmp -s "hf_space/src/$f" "src/$f" && echo same || echo diff; } || echo mirror-only
done | sort | uniq -c

# Coverage (branch), parked packages
pytest tests/templates/ --cov=src/templates --cov-branch -q --no-header          # 72.39
pytest tests/math_kernel/ --cov=src/math_kernel --cov-branch -q --no-header      # 61.51
printf '[run]\nbranch = true\n' > .coveragerc.backend && \
  pytest tests/backend/ --cov=src/backend --cov-config=.coveragerc.backend -q --no-header   # 56.46
pytest tests/deployment/ --cov=src/deployment --cov-branch -q --no-header        # 27.91
pytest tests/integrations/eval_harness/ -q                                       # 11 passed, 8 skipped
grep -oE "fail-under=[0-9]+" .github/workflows/ci.yml | grep -v "=8[5-9]\|=9" | wc -l   # 13 hits; line 1582 is inside a comment -> 12 gates below 85

# Docs / release
grep -n "^## \[Unreleased\]" CHANGELOG.md        # 3, 130
wc -l CHANGELOG.md CLAUDE.md                     # 1864, 898
git tag | wc -l                                  # 0
grep -n "0\.1\.0" RELEASING.md; grep -n "pinned" SECURITY.md; grep -n "^version" pyproject.toml

# Hermeticity trial (plugin installed in the scratchpad only)
pytest tests/distributed tests/integrations tests/dashboard -p pytest_socket \
  --disable-socket --allow-unix-socket --allow-hosts=127.0.0.1,localhost -q   # 542 passed; 4 fail without --allow-hosts

# Lockfile feasibility
uv lock --dry-run --python 3.10     # 206 packages, ~10 s
```

## Appendix B — Backlog cross-reference

| Backlog ID | Status today | Picked up in |
|---|---|---|
| B1 device promotion | open; no module-level `poc↔research` cycle exists | 2.3, 4.2 |
| B2 `CompareScenarioBase` | done (PR #150) | relied on by 3.2 |
| B3 registry consolidation | open | 4.3 (D13) |
| B4 god-file splits | 5 of 7 done | WS1 (four extractions) |
| B5 config/logging unification | open | 4.4 (logging half only) |
| B6 mypy hard gate | open | R-03, R-06 |
| B7 composite action / args file | ledger half done | R-04; 0.6a adds a `-m` copy to the ledger |
| B9 seed-stride unification | open | D12 (deferred) |
| B10 dead packages | open; charter :335 records disposition | 3.5 / D1 |
| B11 YAML dedup | open | 3.7 (delete-first) |
| B13 CHANGELOG / release | open | R-08 |
| B14 `hf_space` single-source | open (frozen) | 3.8 / D4 |
| B15 script CLI boilerplate | open | 3.4 |
| B16 fixture prune / shim policy | open | 3.6, 5.8 |
| B21 god-file-split skill | done | used by WS1 |
| B22 subagent worktree isolation | open | R-14 |
| B35 network-fetching unit tests | open; reproduced | R-02 |
| B37 eval-harness ungated | open | R-13 |
| CHANGELOG parked LBB `* 10` / FNO `128` | open | 2.2 |
| 2026-08-19 `pickle.loads` on `all_gather` | open | 6.1 |
| 2026-08-19 SIGINT handling | deferred | 6.2 |
| 2026-08-19 `evaluation.py:321` `torch.load` outside chokepoint | open (`pyproject.toml:46` names it) | 1.3 |
| CLAUDE.md Next Steps: wall-clock ratio assertions | open; reproduced under load | 0.6a, 0.6b |
| CLAUDE.md Next Steps: `.gitleaks.toml` allowlist | open | 6.3 |
| CLAUDE.md Next Steps: `test_path_traversal_in_config` | open; stronger than documented | 6.5 |
| CLAUDE.md Next Steps: `check_doc_links.py` inline spans | open | 7.5 |
| CLAUDE.md Next Steps: second workflow parser | open | with 7.2 |
| CLAUDE.md Next Steps: chess E2E memory leak | open | D14, 5.8 |
| `docs/FOCUS.md`: promote `focus`, re-scope tracks | open (promised there) | R-09 / D9 |
