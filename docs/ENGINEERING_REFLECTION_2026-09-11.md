# Engineering Reflection & Optimisation Plan — 2026-09-11 (rev 4)

> **Status:** a plan **and** the implementation landing behind it, in the same pull
> request (#151). Rev 3 said "nothing here is implemented by the change that adds
> it"; that was true of the commit that added rev 3 and false of the PR as it now
> stands, which a reviewer rightly flagged. The
> [implementation ledger](#implementation-ledger) at the end of this document is
> the record of what has landed, by ticket and commit. Every *measurement* in
> [§1](#1-what-review-changed) and the KPI tables is the **pre-implementation
> baseline** taken on default-branch tip `ba03b43` (merge of PR #150); the command
> for each is in [Appendix A](#appendix-a--reproduction-commands), or the figure
> is labelled *report-only*. The one committed measurement artifact added by this
> PR, `config/shape_baseline.yaml`, records its own later provenance
> (`generated_from: efdf4c871ca9bf92889980a4c350ae7cee0fe2c8`) because the file
> did not exist on `ba03b43`. Where a landed ticket has changed a measured fact
> (the `lint`/`typecheck` split, the merge-gate membership, the install-step
> count), the table keeps the baseline value and the ledger states the new one —
> the numbers are not rewritten in place, so the before/after stays legible.
> Figures corrected by review are listed with old and new value in §1.
>
> **Scope authority:** the project charter
> (`openspec/specs/project-charter/spec.md`) is supreme. Items tagged
> `openspec-change` change a charter Requirement's text, a register row, or an
> accepted-deviation row and go through that process first. Items tagged `ADR`
> record an architectural decision under `docs/adr/` (ADRs are immutable; a
> change supersedes). Items tagged `frozen` touch a path listed in
> `config/focus.yaml`. The scope gate (`scripts/check_focus.py`) reports a
> violation **only** when one changeset makes a substantive frozen-track change
> (> 20 changed lines per track) **and** touches a `core_paths` entry
> (`src/{mcts,pde,refinement,research}/`); a frozen edit in a PR that touches no
> core path needs neither a separate PR nor the `focus-override` label.
>
> **Relationship to earlier audits:** `docs/CODE_HYGIENE_AUDIT.md` (backlog
> B1–B40) and `docs/CODE_HYGIENE_REVIEW_2026-08-19.md` are not re-litigated; their
> open items are pulled in by ID ([Appendix B](#appendix-b--backlog-cross-reference)).

---

## 0. One page

**Thesis.** The repository is green and well-guarded but its *shape* is
unmeasured: oversized modules, unreachable code, lazy imports that hide layering,
duplicated helpers — none of it gated, so none of it ratchets. Three things that
most directly protect the honest-benchmark story are softer than they look: the
only job that replays a headline benchmark is a soft gate, the type checker is a
soft gate that let a regression merge this week, and the fast lane passes only
because GitHub runners have network egress.

**Minimum viable cycle — 3 weeks, 15 PR-sized tickets (R-04 is two PRs).**
A ticket that lands a guard is done when the guard is green on the default
branch and its planted mutation is recorded as killed. A governance or
administrative ticket (R-01, the deviation branch of R-07, R-14) has no mutation
target: it is done when the acceptance column's state is observable on GitHub
or in the tree. Implementation cards for every ticket are in
[§5](#5-ticket-implementation-cards); the rows below are the summary.

| ID | Wk | Ticket | Size | Acceptance | Route |
|---|---|---|---|---|---|
| R-01 | 1 | PR hygiene: merge #136, #138; rebase-merge #103, #135; **close #139** (contradicts the repo's dependabot langfuse-major ignore rule); each of #48, #57, #118, #128 closed or labelled `keep:<reason>` | S | zero dependabot PRs > 14 days; no unlabelled stale PR | direct |
| R-02 | 1 | Hermetic fast lane: `pytest-socket` in the `dev` extra, `network` marker registered, socket flags on the two fast-lane CI steps and their Makefile mirrors (not `addopts`); the two VGG-download tests mocked; `monkeypatch.chdir(tmp_path)` autouse in `tests/security/` | S | new `tests/docs/test_fast_lane_is_hermetic.py`; a planted socket-opening test fails under the fast-lane command | direct (frozen test file, no core path) |
| R-03 | 1 | `mypy --strict` to zero **unsuppressed** errors: fix `src/training/base_trainer.py:45` `[unused-ignore]` and the `SupportsMetricsMapping` Protocol behind `src/poc/scenarios/stochastic_galerkin_compare.py:63` `[arg-type]`; a documented override block for `src/video_compression/codec/codec.py` (5 diagnostics from 3 real `Tensor \| None` defects) with `remove_when: codec freeze lifts` in its comment and a CHANGELOG line | S | mypy exits 0; the override names one module and a removal condition | direct |
| R-04a | 1 | Lockfile: universal `uv.lock` (un-ignore `.gitignore:132`), `cpu` extra with `[tool.uv] conflicts` + `[tool.uv.sources]` for the CPU torch wheel (a **lock-time** choice, not a sync flag), `uv lock --check` in `lint`, dependabot `uv` block, ADR 0006 | M | `uv lock --check` fails on an un-relocked floor bump; `uv.lock` tracked | direct + `ADR` (D2) |
| R-04b | 1–2 | Install surfaces onto the lock: composite action (`astral-sh/setup-uv`, `uv sync --frozen --extra cpu`, `.venv/bin` on `GITHUB_PATH`, venv cache keyed on the lock) across the 22 non-frozen install steps in four workflows, `docker/Dockerfile`, `.claude/hooks/session_start.sh` (pip fallback), `CONTRIBUTING.md`, `docs/getting-started.md` | M | new `tests/docs/test_composite_actions_are_inert.py` | direct |
| R-05 | 1 | Artifact-freeze manifest `results/MANIFEST.sha256` over `results/*.csv`, `results/*.run.json`, `config/baselines/*_ci.json` (PNGs presence-only); regenerated only by the `claims-ledger`/`run-provenance` skills; fix `ci.yml:1503` to upload the run's `outputs/transfer_ci/`, not the committed files | S | new `tests/docs/test_artifact_manifest.py`; one flipped byte → red | direct |
| R-06 | 2 | Hard mypy: drop `continue-on-error` (`ci.yml:183`) and rewrite the false comment (:175–182); drop `\|\| true` from `Makefile` `mypy`; retire the manual pre-commit mypy hook; `CONTRIBUTING.md`, CLAUDE.md prose; retire charter deviation row :328 | S | new `tests/docs/test_soft_gates.py`; `MYPY=false make mypy` exits 1 | `openspec-change` (:328) |
| R-07 | 2 | Transfer tripwire: three `workflow_dispatch` re-runs on the lockfile; if all within `config/baselines/transfer_ci.json`'s 40 % tolerance, flip the soft branch in `ci-success` to `exit 1` (and rewrite `test_e2e_visibility.py::test_a_reported_but_ungated_job_is_not_counted_as_blocking`, which asserts the job is *not* blocking); else record an accepted-deviation row in the form of :328 | S–M | no soft branch in `ci-success`, or a deviation row with a retirement condition | direct (hard) / `openspec-change` (soft) |
| R-08 | 2 | Release: merge the two `[Unreleased]` blocks (lines 3 and 130; a second preamble sits at 125–128), changelog guard, cut `0.4.0` per `RELEASING.md`, fix `RELEASING.md:6` and `SECURITY.md:38`, re-lock | S–M | new `tests/docs/test_changelog_headers.py`; tag `v0.4.0` exists | direct (D8), after R-04a |
| R-09 | 2 | `focus` and `secrets` into `ci-success.needs` and its `exit 1` block, with event-aware skip handling for `focus` (exact script in §5); rewrite charter row :334 and `docs/FOCUS.md:27-28`; `config/focus.yaml` itself is unchanged | S | new `tests/docs/test_ci_success_hard_gates.py` | direct (promotion) + `openspec-change` (:334 prose) |
| R-10 | 3 | Module-size budget guard: in-test dict path → (recorded lines, reason), ceiling 600 (44 rows), second dict for tests at 1,000 (8 rows) | S | new `tests/docs/test_module_size_budget.py`; three mutations | direct |
| R-11 | 3 | Shape baseline: `scripts/measure_shape.py` → `config/shape_baseline.yaml` (nine metrics + a content hash over `src/**/*.py`); `tests/docs/test_shape_baseline.py` asserts actual ≤ recorded and rejects a lowered count when the content hash is unchanged | M | raise a count → red; hand-lower with `src/` unchanged → red | direct |
| R-12 | 3 | CI critical path: (a) `test-slow`/`test-integration`/`test-e2e`/`test-extras` `needs: lint` instead of `test-fast`; (b) 4-shard `coverage-gates` in-file; (c) `typecheck` job split from `lint` (job id `lint` stays ruff-only) and added to `ci-success` | M | ≤ 15 min on a PR run (the `coverage` sweep alone is 9.7–12.4 min; < 12 needs the sweep de-duplicated — stretch 27); new `tests/docs/test_coverage_gate_shards.py` | direct |
| R-13 | 3 | `eval_harness`: `eval_harness_required` marker + root `conftest.py` hook; relocate the four module-scope `eval_harness` imports; report-only coverage run in `test-extras`, then gate at `floor(measured) − 2` if ≥ 75 else a disclosed tripwire; drop the integrity-guard exemption | M | `test_coverage_gate_integrity` green with the exemption removed; CI step lands before the charter row | direct (`add-coverage-gate` carve-out) |
| R-14 | 3 | Worktree-isolation rule in root `AGENT.md` and all six `.claude/agents/*.md` (all declare `Bash`) | S | new `tests/claude::TestConcurrencyRule` anchor-sentence test | direct |

**Decisions due inside the cycle** (register in [§9](#9-owner-decisions)):
D2 lockfile shape (week 1), D11 artifact freeze (week 1), D3 hard mypy (week 2),
D8 release (week 2), D9 what R-09 re-scopes (week 2), D10 tripwire hard or
deviation (week 2), D7 ratchet rule (week 3).

---

## 1. What review changed

Rev 1 was reviewed by five internal lenses (adversarial fact-check, CI
feasibility, test/coverage, architecture, product/sequencing); rev 2 by three
Copilot rounds; rev 3 by a consistency/executability audit, a per-ticket
dry-run against the tree, and a governance/security pass. Rev 1 promised "the
exact command for every number" and the reviews found that promise partly
false — the same "verified-but-wrong" pattern §3 warns about. Everything below
is re-measured.

**Measurement corrections (rev 1 → measured).**

| Rev 1 said | Measured | Consequence |
|---|---|---|
| 3 of 12 CI install steps cache pip | all 12 do; install still costs 56–84 s (CUDA torch wheel from PyPI) | the lever is a CPU-wheel lock + venv cache, not a composite action |
| 12 install steps | **23** across five workflows (12 `ci.yml`, 7 `regression-surface.yml`, 2 `sbir-demo-smoke.yml`, 1 `docs.yml`, 1 frozen `phase2`) plus Dockerfile, hook, CONTRIBUTING | R-04b scope. *Baseline; the R-12a `typecheck` split added a 13th `ci.yml` install (24 total) — see the ledger* |
| `coverage-gates` (14 min) drives wall-clock | finishes 2 min 43 s before `test-e2e`; critical path `lint 2.5 → test-fast 8.6 → test-e2e 7.9 = 19.3 min`; the `coverage` sweep alone is 9.7 min (push) to 12.4 min (PR) | rewire first; < 12 min needs the sweep, not sharding |
| `CI Success` gates 10 jobs | 9 hard + 1 soft (transfer); `focus`, `secrets`, `test-slow` outside `needs`; a third soft step (ONNX, `ci.yml:1451`) | R-07, R-09, 0.12. *Baseline; after R-09 + R-12a the gate is 12 hard + 1 soft with only `test-slow` outside `needs` — see the ledger* |
| 0 shims warn | 3 of ~28 shim sites emit `DeprecationWarning` | KPI reworded |
| 368 lazy imports / 139 files | **154 / 65** by AST; classification (9 cycle, 10 registry, 31 optional, 15 torch/scipy, 89 no reason) is report-only | target ≤ 60 |
| `poc ↔ research` cycle | none at module level; real SCCs `{data, training}`, `{pde, experiments, research}` | 4.2 |
| 121 `"cuda"` literals to fix | 121 total; 46 frozen; **10** real resolution sites among the other 75 | 2.3 counts resolution sites |
| 6 files > 900; 29 > 600 | **5** > 900 (`base_trainer.py` is exactly 900); **44** > 600; 8 test files > 1,000 | R-10 |
| 9 unsafe-pickle argparse blocks | **5** (2 frozen) | 3.3 |
| 4 dependabot PRs | **5**; #139 must close | R-01 |
| 2026-08-19 "flagged" values still open | all four fixed there; the parked ones are modeling LBB `* 10` and FNO `128` (CHANGELOG:17) | 2.2 |
| B4 splits 3 of 8 | **5 of 7** (`operators`, `trainer` C2 facade, `baselines`, `mesh_refinement`, `losses/physics`) | WS1 |
| `_run_checkpoint_tournament` re-implements `src/tournament` | a 9-line delegator to `trainer_eval.py`; `src/tournament/AGENT.md` records "no keep-and-wire" | 1.2 |
| `core/registry.py` is the newer base | 0 consumers outside `src/core/__init__.py`'s `__all__` re-export; it imports `BaseRegistry` from `templates/registry.py` | 3.1, 4.3 |
| ungated packages 1 | **3** (`templates`, `math_kernel`, `eval_harness`) | R-13, 5.1, 5.2 |
| `templates/cli.py` 0 %, no consumer | 59 % under `tests/agents/test_cli.py`; consumer `src/agents/cli.py` | 5.1 |
| 3 modules imported by nothing incl. tests | **2** (`backend/logging.py`, `backend/rng.py`) | 3.1 |
| `SECURITY.md` claims pins, none exist | pins exist for ruff, a mypy/pydantic range, the eval-harness git SHA; the resolved graph is not locked | R-08 wording |
| `evaluation.py:321` `torch.load` is a security gap | under the declared `torch>=2.6.0` floor it is `weights_only=True` by default (`pyproject.toml:40-51` says so); the defect is loader **consistency**: no `SAFE_CHECKPOINT_GLOBALS` window (a `CheckpointManager` checkpoint carrying a `datetime` fails to load there), no lock, no normalised error, no audited `allow_unsafe_pickle` | 1.3 mission reworded |

**Design reversals.**

- Sharding `coverage-gates` is not a speed item; rewire `needs:` first, and
  even then the `coverage` sweep caps the run at ~15 min on a PR day.
- A complexity baseline cannot live in `per-file-ignores` (hides new findings
  in a listed file) and must not use `allow-magic-value-types = ["int"]` (136 of
  159 findings are ints and they *are* the domain values). Baseline is a table
  compared in-test (R-11).
- vulture at 60 % is the wrong first tool (report-only sample: 15–25 % truly
  dead, ~45 % public API reached only by tests); the reproducible first pass is
  the orphan-module count (§2, Appendix A).
- `filterwarnings = error::DeprecationWarning` cannot guard shims (attributed to
  the caller); a ledger-driven test does (3.6).
- Registry unification reversed and gated on D13: ADR 0005 chose
  `clear`/`ensure`; the audit measured `snapshot/restore` as a net regression.
- WS1 is extractions, not package conversions: two scheduled (1.3, 1.5), three
  touch-triggered (1.2, 1.4, 1.6); the `lshape_amr_compare` split is cut
  (charter retirement row :332).
- The `src` package rename is cut (`alphagalerkin` is already the console script,
  an entry-point group and `src/alphagalerkin/`; ~4,700 occurrences; depends on
  B14).
- B10 deletions cite charter row :335 and each package's `AGENT.md`; `deployment`
  is CLI-addressable and stays.
- **Rev 3 route corrections:** the frozen-track rule (header) was wrong in
  rev 2; R-07's soft branch is an accepted-deviation row, not an exclusion-ledger
  entry (that ledger's guard requires its fence to equal the ignore lists); R-09
  is `direct` for the promotion (`ci-success.needs` is in no register) and
  `openspec-change` only for the :334 prose; 7.2/D6 is `direct` (the table's
  owner is `openspec/project.md`'s rank-3 row, not a Requirement); 3.5 is a
  wider change package (Non-Goal Exclusion text is date-bound to 2026-07-22,
  Scope Integrity, four Quality-Gate rows, :335) executed as **one** PR with the
  `hf_space/src/<pkg>` scrub, because `test_mirror_guard.py` iterates
  `CUT_MODULES`; 4.1 needs the ADR **and** `ARCHITECTURE.md` (it owns layering);
  a CPU torch index cannot be a sync-time flag under a lockfile (R-04a).
- **Rev 3 security corrections:** a tensor-only dict would drop `Experience`'s
  `board_size`, `target_value` and `metadata` (6.1 reworded); PNG hashes churn on
  every honest re-run because matplotlib embeds its version and rasterises per
  platform (R-05 hashes CSV/JSON only); `uv export` takes `--format
  requirements.txt` and `uv audit` does not exist (6.4); dropping `tests/.*` from
  the gitleaks allowlist is safe at HEAD (no key-shaped strings; the only
  `api_key=` literals are 1–8-char stand-ins) but CI's action scans the push
  range, so full-history coverage is a local precondition (6.3).

---

## 2. State of the codebase (measured 2026-09-11)

| Dimension | Measured | Gated today? |
|---|---|---|
| CI, default branch | green; 16 jobs; 19.3 min; critical path `lint → test-fast → test-e2e`; `coverage` sweep 9.7–12.4 min; `coverage-gates` 13.8 min off-path (44 serial steps); all 12 `ci.yml` installs cached, 56–84 s each | 9 hard + 1 soft in `ci-success`; `focus`/`secrets`/`test-slow` outside; ONNX step soft |
| Fast lane, egress-blocked sandbox | 10,028 passed, 293 skipped, **2 failed** (VGG16 download, B35). *Report-only:* under the adversarial review's concurrent load the same lane also failed `test_fnet_vs_attention[cpu]` and `test_galerkin_attention_scaling[cpu]` | green only with egress |
| `mypy --strict` | 7 errors / 3 files in a `.[dev]` env (5 in frozen `codec.py`, 1 unused-ignore, 1 `arg-type` introduced by PR #150); the `lint` job (now `typecheck`, R-12a) installs a different minimal set, so its count can differ; `Makefile` masks with `\|\| true`; pre-commit mypy hook is `stages: [manual]` | **no** |
| Complexity (`C901`, `PLR091x`) | 162; 17 functions > 50 statements (max 85) | no |
| `PLR2004` | 159 = 136 int + 23 float; 20 frozen | no |
| Module size | > 900: 5 (+1 at exactly 900); > 600: 44 (4 frozen codec, 4 demos); tests > 1,000: 8 (one is `tests/docs/test_e2e_visibility.py`) | no |
| Dead / orphan code | vulture 80 %: 14 (6 `__exit__` args, 8 real); 60 %: 893 (report-only sample 15–25 % real); orphan library modules **21** (6,274 LOC; e.g. `mcts/gumbel.py` 712, `research/scaling_runner.py` 467, `data/dataset.py` 315); `core.registry.Registry` unused outside its re-export; `backend/logging.py`, `backend/rng.py` imported by nothing, tests included | no |
| Zero-inbound packages | `prototyping`, `analysis`, `curriculum`, `tournament` (charter :335), `deployment` (CLI-addressable) | charter records disposition |
| Lazy imports (AST, non-`TYPE_CHECKING`) | 154 in 65 files; module-level SCCs `{data,training}`, `{pde,experiments,research}` | no |
| Device resolution | 6 named `resolve_device`/`_resolve_device` defs; 10 ad-hoc `"cuda" if is_available()` sites outside `src/device.py`; 19 consumers of the `poc.device` identity shim | no |
| `print()` in `src/` | 116 (ruff `T201`): 99 in modules with a `__main__`, **17** in library code | no |
| `except ImportError` | 76: 25 guard hard deps (dead), 11 guard first-party imports (hide breakage), ~40 genuine optional (9 of them frozen) | no |
| Duplication | 4 `export_csv`/`export_plot` pairs (shared skeleton only); 5 unsafe-pickle argparse blocks (2 frozen); 3 `SAFE_*_GLOBALS`; 22 of 25 scripts hand-rolled argparse; `lm_studio` YAML 8 blocks / 3 contents; `poc/logging.py` vs `templates/logging.py` duplicate `log_timing`/`log_call`/`DebugContext` | no |
| Config surface | 62 `BaseModel` (46 config-like, 16 records), 63 `BaseModuleConfig`, 11 `BaseScenarioConfig`; ~2 user-facing `@dataclass` configs | — |
| Registries | ~16 registry-like objects (report-only count; 10 by the Appendix A grep); 9 on `templates.BaseRegistry`; `ScenarioRegistry`/`GameRegistry` hand-rolled; 20 test files call `*.clear()` | ADR 0005 |
| Shims | ~28 back-compat sites; 3 warn; no ledger; no removal dates | no |
| Coverage, parked | `templates` 72.4, `math_kernel` 61.5 (not in `omit`), `backend` 56.5, `deployment` 27.9 (its ONNX suites are soft), `eval_harness` 11 pass / 8 silent module-level `importorskip`; 12 gates below 85 | partial |
| `hf_space/` | 58,315 LOC; `hf_space/src` 55,038; 98 of 146 mirrored files diverge, 9 mirror-only | frozen; B14 |
| Docs / release | `CHANGELOG.md` 1,864 lines, two `[Unreleased]` (3, 130) and two `## [0.3.0]` (842, 1648); `CLAUDE.md` 898 lines; 7 planning docs; **no git tag ever**; `RELEASING.md:6` says 0.1.0 (pyproject `0.4.0-dev`); `SECURITY.md:38` "pinned in pyproject.toml"; ADR 0003 reserved for PR #118 | partial |
| PRs | 5 dependabot (#139 must close), 4 stale (#48, #57, #118, #128) | monthly Routine reports |
| Dependencies | no lockfile; `uv lock --dry-run` resolves 206 packages in 10 s; `uv` not on runners or in any extra | no |

---

## 3. Reflection — five lessons

1. **A check that cannot fail is a report.** Seven invisibility incidents this
   year, all found by humans reading YAML; every durable fix was a guard that
   reads CI as data. Rev 1 of this plan omitted two more (a soft benchmark gate,
   three jobs outside `needs`).
2. **"Verified" in prose is a liability.** The tracer pin, the Dockerfile row,
   the eval-harness exemption, the fabricated figure — and the §1 table above.
   Numbers live in `config/shape_baseline.yaml` behind a test (R-11).
3. **Global singleton state × collection order** produced four defects on one
   branch. ADR 0005's contract is the recorded answer (D13).
4. **Accident-of-environment tests.** Network in unit tests (B35), CWD-dependent
   security tests, wall-clock ratios in blocking lanes. Hermeticity is asserted
   (R-02).
5. **Doc drift outruns code drift.** Two `[Unreleased]` headers, a stale release
   version, a stale pin claim, an 898-line CLAUDE.md. Machine-check counts;
   shorten the rest.

---

## 4. Principles for this cycle

- **Backwards compatible by default**: import-compatible re-exports plus a
  public-API freeze test (B21 recipe); deprecations warn, are ledgered with a
  `remove_in`, and get two minor releases.
- **One concern per PR, ≤ 300 diff lines** where possible; review time, not
  execution time, is the constraint (one human reviewer in `CODEOWNERS`).
- **Measure → gate → ratchet; never widen a threshold to get green**, including
  by configuration.
- **Prefer deletion to abstraction.**
- **Hash-pin protocol**: a change to any scenario config with a committed
  `results/*.run.json` is resolved only by re-run + re-record via
  `run-provenance` and `claims-ledger`; if a `charter:evidence` value moves,
  that is a numeric-claim correction → `openspec-change` + `claims-ledger`. The
  manifest (R-05) makes every PR's rollback "revert; manifest proves artifacts
  untouched".
- **Frozen paths never share a PR with core-path work** unless the frozen delta
  is ≤ 20 lines or the PR carries `focus-override`; a frozen edit in a PR that
  touches no core path is unconstrained; recording a frozen file in a read-only
  baseline (R-10, R-11) touches no frozen path at all.
- **Docs cite commands**, or say report-only.

---

## 5. Ticket implementation cards

Each card: files, guards that go red and must move in the same PR, the new
guard's assertions and planted mutations, blockers the summary omits.

**R-01.** Owner action on GitHub. Ordering: merge #136 (`setup-python` 5→7) and
#103 (`checkout` 4→7) **before** R-04b, or the composite-action PR conflicts on
every `uses:` line; #135 (`gitleaks-action` 2→3) must be green on ≥ 2 runs
before R-09 promotes `secrets`. If #118 closes, release ADR 0003 in
`docs/adr/README.md:31-33`. #128 is the ONNX `dynamo=False` fix that lets 0.12
go hard.

**R-02.** Files: `pyproject.toml` (`dev` += `pytest-socket`; markers += one
`"network: …",` line — the vocabulary guard's regex needs one per line);
`ci.yml` steps `Run fast unit tests` and `Run tests with coverage`; `Makefile`
`test-fast`/`coverage`. Flags on the steps, not `addopts` (which would silently
change `docker/Dockerfile`'s `CMD`, the manual LM-Studio/GPU smokes and every
Regression Surface command). Guards that go red: if the fast-lane `-m` gains
`and not network`, `tests/docs/test_marker_vocabulary.py::test_known_expressions_are_found_verbatim`
anchors the literal expression — update it in the same PR. VGG tests:
`tests/video_compression/unit/test_loss.py::TestCompressionLoss::{test_perceptual_loss_enabled,test_total_combines_all_losses}`
→ mock `src/video_compression/metrics/quality.py:392-406::_get_vgg` (frozen
path; no core path touched, so same PR). New guard asserts both steps and both
recipes carry `--disable-socket --allow-hosts=127.0.0.1,localhost --allow-unix-socket`,
`pytest-socket` is in `dev`, `network` is registered; mutations: drop the flag
from one recipe, drop the dependency. **Blocker:** the trial covered 542 tests
in three directories; run the full 10k fast lane under the flags before landing.

**R-03.** Files: `base_trainer.py:45`; `src/poc/scenarios/_compare_common.py`
(make `SupportsMetricsMapping.metrics` a read-only property member — the
comparison's `metrics` is a read-only `dict`); `pyproject.toml` override block
`disable_error_code = ["operator", "union-attr", "arg-type"]` for
`src.video_compression.codec.codec` (exactly the three codes at :392/:515/:518/
:520/:531) with the removal condition in its comment; `CHANGELOG.md`. No guard
parses the mypy overrides. **Blocker:** re-measure in the `typecheck` job's
environment (the mypy step moved there from `lint` in R-12a) after R-04b and
before R-06.

**R-04a.** `uv lock` is universal across 3.10–3.12. CPU torch: `cpu` extra,
`[tool.uv] conflicts = [[{extra = "cpu"}, {extra = "cu126"}]]`,
`[[tool.uv.index]] name = "pytorch-cpu" explicit = true`,
`[tool.uv.sources] torch = [{index = "pytorch-cpu", extra = "cpu"}, …]`; base
`dependencies` keep `torch>=2.6.0` from PyPI for CUDA hosts. Pin the six
distributions `src/research/run_manifest.py`'s collector (lines 211–222) records
to the arena sidecar's versions; the rest resolve. `uv lock --check` needs
network for the `langfuse-eval-harness` git dependency — keep it out of
`make lint`. Guards: `tests/claude::test_the_directory_is_not_gitignored` and
`tests/research/test_run_manifest.py::TestSidecarsAreCommittable` read
`.gitignore`. ADR 0006 ratifies `.gitignore:203`'s existing "we standardize on
uv.lock" against `:132`.

**R-04b.** Composite `.github/actions/setup-env/action.yml` (inputs:
`python-version`, `extras`); 22 install steps (`ci.yml` 12,
`regression-surface.yml` 7, `sbir-demo-smoke.yml` 2, `docs.yml` 1;
`phase2-zoo-validation.yml` frozen, left alone); `docker/Dockerfile`
(`COPY uv.lock` after it is tracked — `tests/docs/test_dockerfile_context.py`;
`--no-install-project` before `COPY src/`); hook (`tests/claude` `bash -n`);
docs. New guard: every `run:` under `.github/actions/**` contains no
`pytest`/`--cov`/`coverage`; every workflow install step `uses:` the composite
(frozen phase2 allowlisted, self-expiring); `uv.lock` tracked and not ignored.
Mandatory because `tests/support/workflows.py` globs only
`.github/workflows/*.yml`: a gate or `-m` moved into a composite escapes
`test_coverage_gate_integrity`, `test_marker_vocabulary` and
`test_e2e_visibility` entirely. Mutations: `pytest` in `action.yml`; re-ignore
`uv.lock`; a raw `pip install -e` in a non-allowlisted workflow.

**R-05.** Inventory: 13 tracked files under `results/` (incl.
`lambda_scheduling.{csv,png}`, pinned permanently by charter :331, and three
`.run.json`), 3 under `config/baselines/` (`poc_headline.example.json` is a
template — excluded). Format `sha256sum -c` compatible. Guard: entries ↔
`git ls-files results config/baselines`, hashes equal, ≥ 10 entries; mutations:
flip a byte in `transfer_baseline_compare.csv`, delete a line, add an unlisted
CSV. Writers: CI never writes `results/` (`transfer-baseline-regression` writes
`outputs/transfer_ci`; E2E writes `tmp_path`) but its upload step
(`ci.yml:1503`) uploads the committed files — fix in this ticket. Shipped YAMLs
default to `output_dir: results`, so the documented live-run commands dirty the
tree by design: the `claims-ledger` skill and CLAUDE.md live-run blocks must say
so and prefer `--output-dir outputs/…`. The skill edits cite
`results/MANIFEST.sha256`, which `tests/claude::test_cited_repo_paths_exist`
requires to exist — land the file first.

**R-06.** Files: `ci.yml:174-184`; `Makefile:117`; `.pre-commit-config.yaml`
mypy hook (retire it: `pass_filenames: true` in pre-commit's own venv with torch
as an additional dependency, `stages: [manual]`, so it never runs — also
`ci: skip: [mypy]`); `CONTRIBUTING.md:55`; `CLAUDE.md:352,406`; charter :328 via
`openspec/changes/<id>/{proposal,design,tasks}.md` + delta; audit B6 status.
Guard `test_no_soft_gate_steps`: every `continue-on-error: true` across all
workflows ∈ allowlist `{("typecheck", backend audit), ("test-extras", ONNX until
#128)}` (the backend audit moved from `lint` to `typecheck` in R-12a),
self-expiring; the `mypy` recipe contains no bare `true`. Mutations:
re-add the soft flag; re-add `|| true`; keep an allowlist row for a step that is
no longer soft.

**R-07.** Job facts: 4 baseline entries, `tolerance_pct: 40`, ~3 min,
`workflow_dispatch` exists. "Stable" = all three re-runs within tolerance on the
locked environment. The hard flip edits `ci.yml:1536-1541` and must rewrite the
live-file assertion in `test_e2e_visibility.py:712-719`. The soft branch is an
accepted-deviation row (retirement condition: three stable runs), not a
`docs/ci-exclusion-ledger.md` row.

**R-08.** CHANGELOG: two preambles (lines 1–3 and 125–130), two `[Unreleased]`
blocks with their own `### Added/Changed/Fixed` groups to merge; `## [0.3.0]`
appears twice (842 `2026-07-22`, 1648 `2026-04-01`) — the guard is
"non-increasing under PEP 440, unique except an allowlisted `{"0.3.0": reason}`",
self-expiring. Bump touches `pyproject.toml:7`, `README.md:277`
(`test_version_consistency`), `hf_space/src/__init__.py:8` (scanned; frozen; one
line, no core path), `docker/Dockerfile:1`, `RELEASING.md:6` (+ cadence sentence
+ `Development Status` classifier), `SECURITY.md:38` (say "locked in `uv.lock`"
— true only after R-04a), `uv.lock` (embeds the version — re-lock). No
`cz bump`; hand-cut per `RELEASING.md`. No on-tag workflow (disclosed §9).

**R-09.** Facts: `focus` `if:` is `github.event_name == 'pull_request' &&
!contains(labels, 'focus-override')`; `secrets` and `ci-success` share
`github.event_name != 'schedule'`, so `secrets` is never skipped when
`ci-success` runs and needs no skip handling; `focus` was `skipped` on the last
push run. Script (verified to register both jobs in
`tests/support/workflows.py::hard_gate_jobs`; a precomputed-variable form is
**not** recognised):

```bash
if [[ "${{ needs.focus.result }}" != "success" && "${{ needs.focus.result }}" != "skipped" ]]; then
  echo "FAIL: Scope containment failed (result=${{ needs.focus.result }})"; exit 1
fi
if [[ "${{ needs.focus.result }}" == "skipped" && "${{ github.event_name }}" == "pull_request" \
      && "${{ contains(github.event.pull_request.labels.*.name, 'focus-override') }}" != "true" ]]; then
  echo "FAIL: Scope containment skipped on a pull request without the focus-override label"; exit 1
fi
if [[ "${{ needs.secrets.result }}" != "success" ]]; then
  echo "FAIL: Secret scan failed (result=${{ needs.secrets.result }})"; exit 1
fi
```

Files (line numbers as of `ba03b43`, before the edits): `ci.yml:1512` `needs`,
the echo table, stale comments at :205-211, :289-293, :1580-1584;
`config/focus.yaml:4` (**said**, until R-09 corrected it, that the check was
"wired into CI's `lint` job"; it is its own job); `docs/FOCUS.md:27-28`
(`tests/scripts/test_check_focus.py` keeps it in step with the YAML); charter
:334 rewrite. "Re-scope" is nil in YAML (both tracks stay; `frozen_tracks` min
1) — D9 records that. Guard: (1) `{focus, secrets} ⊆ job_needs(ci-success)`;
(2) `⊆ hard_gate_jobs`; (3) a `[[…]]` block naming `needs.focus.result` contains
`skipped`, `github.event_name` and `focus-override`; (4) every job with a
PR-firing `if:` is in `needs ∩ hard`, allowlist `{test-slow: PR-invisible by
design}` self-expiring; R-12's `typecheck` joins the expected set. Mutations:
drop `focus` from `needs`; `exit 1` → `echo`; bare `!= success`; a new
`if: != schedule` job outside `needs`.

**R-10.** `tests/docs/test_module_size_budget.py`: `ALLOWLIST: dict[str,
tuple[int, str]]`, `CEILING = 600`, lines = `len(read_text().splitlines())`
(equals `wc -l` under `end-of-file-fixer`); assertions: unlisted ≤ 600, listed
≤ recorded, listed still > 600, keys exist, scan ≥ 300 files. Mutations:
+50 lines to `src/poc/cli.py`; a new 601-line unlisted module; trim a listed
file below 600 and keep its entry. Second dict `CEILING_TESTS = 1000` (8 rows —
`tests/docs/test_e2e_visibility.py` lists a guard, so each new clause there bumps
its own ceiling). Frozen codec rows are data, not edits.

**R-11.** Files: `scripts/measure_shape.py`, `config/shape_baseline.yaml`,
`tests/docs/test_shape_baseline.py`, `tests/scripts/test_measure_shape.py`,
optional `Bash(python -m scripts.measure_shape:*)` permission (must resolve —
`tests/claude`). Nine metrics: complexity, `PLR2004` (per `(file, rule)`),
`T201`, lazy imports, device-resolution sites, orphan modules, dead
`ImportError` guards, shims without warning, mirror divergence. Content hash =
sha256 over sorted `src/**/*.py` paths + bytes, computed identically by script
and test (`git rev-parse HEAD:src` is wrong: local improvements are uncommitted,
and PR checkouts are synthetic merge commits). Ruff counts via a pinned
`ruff==0.15.8` subprocess in-test (~2 s). Each Appendix A snippet becomes a
function in the script — that is the real size.

**R-12.** (a) `ci.yml:489,541,1323` (`test-integration`, `test-e2e`,
`test-extras`) `needs: test-fast` → `needs: lint`. **Not** `test-slow` (`:438`,
listed in rev 3 by a line-number sweep): its job-level `if:` reads
`needs.test-fast.result`, so rewiring it would leave that condition referring to
a job outside its `needs` and change whether the scheduled/default-branch run
fires; it keeps `needs: test-fast`. Keep job id `lint` for the ruff-only job (`test_e2e_visibility.py:708` asserts
`{"lint","test-fast"} ⊆ blocking`); new job `typecheck` (mypy + both audit
steps, `if: github.event_name != 'schedule'` — `test_nightly_schedule`) added
to `needs`, the `exit 1` block, and R-09's expected set; update CLAUDE.md's
abstraction-audit row ("`lint` job — B18") and `config/focus.yaml:4`.
(b) `strategy.matrix.shard: [1,2,3,4]`, per-step `if: matrix.shard == N`, steps
stay in `ci.yml` (the charter and B8 guards read `run:` bodies); guard: matrix
non-empty, every gate step names exactly one existing shard, every shard has
≥ 1 step; mutations: unsharded step, `shard == 5`, empty shard. Wall-clock: the
`coverage` sweep alone took 12:24 on the PR run — after (a)+(b)+(c) the path
is `lint + max(coverage, gates)` ≈ 15 min; < 12 min needs the sweep
de-duplicated (audit §7.7; stretch 27).

**R-13.** The 8 `importorskip("eval_harness")` sites are
`tests/integrations/eval_harness/test_{contract,dataset,eval_harness_smoke,import_guard,plugins,runner,scorers,sink}.py`;
**four** (`contract`, `plugins`, `scorers`, `sink`) also import `eval_harness.*`
at module scope, so swapping in a marker makes them collection errors without
the extra — move those imports into fixtures. A module-level `importorskip`
yields zero items, so the root hook can never hard-fail them: the current
`test-extras` run is a silent skip by construction. Root `conftest.py`: mirror
the `fem_required` block (lines 60–75) using `importlib.util.find_spec("eval_harness")`;
under `ALPHAGALERKIN_REQUIRE_EXTRAS=1` raise `pytest.UsageError`, else skip with
a counted reason. Gate step in `test-extras`, inline-coveragerc form (the only
one `_step_really_gates` accepts for an omitted path), then delete the
`_OMIT_WITHOUT_A_CI_GATE` entry; register the marker; CLAUDE.md row and charter
gates row after the CI step exists. Threshold unmeasurable here (git extra not
installed): report-only run first.

**R-14.** No agent file mentions concurrency today. Rule text: "each concurrent
subagent runs in its own `git worktree` (a sibling directory, not inside the
repo); never `git stash`, `git reset`, `git checkout -- <path>` or `git clean`
in a shared working tree". Home: root `AGENT.md` `## Concurrent subagents` and
each `.claude/agents/*.md` whose `tools` include `Bash` (all six). Guard: an
anchor-sentence test parametrised over agent files. Traps: do not add a skill
(`CLAUDE.md:332` inventory count); before R-04b the editable install points at
the primary checkout, so run `python -m …` from the worktree root;
`settings.json` already allows `Bash(git worktree:*)`.

---

## 6. Workstreams beyond the cycle

Columns: **Mission** = honesty / core loop / defect class / hygiene. **Route**
as in the header. **When** = stretch N (order in §7), deferred (with the
unblocking decision), or cut.

### WS0 — CI

| ID | Item | Mission | Size | Route | When |
|---|---|---|---|---|---|
| 0.6a | `benchmark` marker deselected from `test-fast`, `coverage`, Makefile (ledger the fourth `-m` copy) **and** a dedicated non-contended `benchmark` job landed in the same PR, or `tests/benchmarks/` is selected by nothing; scale the absolute floors in `test_sbir_demo.py:180` and `test_mcts_perf.py:71` | defect class | S | direct | stretch 1 |
| 0.6b | Rewrite the 7 ratio assertions as complexity-scaling fits with a vacuity check | honesty (the O(N) claim) | M | direct | stretch 2 |
| 0.9 | Full fast lane under `-W error::DeprecationWarning` (1,118-test sample clean); scoped filter after a `pytest.warns` audit (three tests call shims without it) | hygiene | S | direct | stretch 7 |
| 0.12 | ONNX suites hard once #128 lands | defect class | S | direct | stretch 4 |
| 0.14 | Coverage-sweep de-duplication (one sweep, N `coverage report --include … --fail-under`; the charter matcher needs `--cov=<target>`/`--cov-fail-under=N` in one step, so this is an `openspec-change` on the gates register) — the item that makes < 12 min reachable | speed | L | `openspec-change` | stretch 27 |

### WS1 — Module size

| ID | Item | Mission | Size | Route | When |
|---|---|---|---|---|---|
| 1.2 | `training/trainer.py`: six `_create_*` factories + `__init__` wiring → `trainer_setup.py`; periodic hooks of `train()` to methods; preserve the documented no-`super().__init__()` contract (`base_trainer.py:13-16`) | hygiene | M | direct | deferred (touch-triggered) |
| 1.3 | `training/checkpoint.py` → `checkpoint/{safety,manager,model_io}.py`; patch targets in `tests/security/test_checkpoint_safety.py:462,507` move; permanent re-exports (frozen scripts import it); route `evaluation.py:321` through the chokepoint | defect class (loader consistency, defence-in-depth) | M | direct | stretch 10 |
| 1.4 | `games/chess.py`: move-encoding seam only (~230 lines; 16 private test refs) | hygiene | S | direct | deferred |
| 1.5 | `pde/games/basis_selection.py` → `basis_library.py` (where the v2.2 geometry-aware library must live) | core loop | M | direct | stretch 9 |
| 1.6 | `base_trainer.py` optimizer/scheduler factories + AMP block; `modeling/model.py` heads/blocks | hygiene | M | direct | deferred |
| ~~1.8~~ | `lshape_amr_compare` split | — | — | — | cut (:332) |

### WS2 — Complexity, hardcoded values, observability

| ID | Item | Mission | Size | Route | When |
|---|---|---|---|---|---|
| 2.2 | 23 float magic values → `src/constants.py`/typed fields; the parked LBB `* 10` / FNO `128`; ints shrink via the baseline. Hash-pin protocol; if a registered value moves, `openspec-change` + `claims-ledger` | core loop | M | direct (conditional) | stretch 8 |
| 2.3 | One device resolver: 19 shim imports → `src.device`; 10 ad-hoc sites + `backend/torch_backend._resolve_device` through it; `ScenarioResult` gets the resolved device (`poc/registry.py:246`) | honesty | M | direct | stretch 6 |
| 2.4 | ~21 inline `outputs/…` literals → a new `OutputPathsConfig` + env override | hygiene | M | direct | deferred |
| 2.5 | 17 library `print()` → structlog; "CLI module" = has `__main__` | hygiene | S | direct | stretch 11 |
| 2.6 | Delete the 25 hard-dep `ImportError` guards; explicit checks for the 11 first-party ones; a new `src/core/optional.py::require_extra` for the genuine optional ones (frozen sites excluded) | defect class | M | direct | stretch 12 |

### WS3 — Dead, duplicated, deprecated

| ID | Item | Mission | Size | Route | When |
|---|---|---|---|---|---|
| 3.1 | Orphan-module guard (21 rows with reason; CLI entry modules excluded) + vulture 80 % report-only; delete `backend/logging.py`, `backend/rng.py`; `core.Registry` is re-exported by `src/core/__init__.py`, so it gets a one-release `DeprecationWarning` shim recorded in the override-comment/CHANGELOG form until 3.6's ledger exists | hygiene | S+M | direct | stretch 3 |
| 3.2 | `write_csv` + matplotlib-guard helper in **`src/research/`** (not `poc`); byte-identical golden on committed CSV rows | honesty | S | direct | stretch 13 |
| 3.3 | One `add_unsafe_pickle_argument(parser)`; one `SAFE_GLOBALS` builder for the two non-frozen allowlists (codec sites in the same PR are fine — no core path) | defect class | S | direct | stretch 14 |
| 3.4 | `scripts/run_{stochastic_galerkin_compare,transfer_baseline_compare}.py`, `run_lshape_amr.py`, `run_mcts_classical_amr_arena.py` onto `src/templates/cli.py`; gate `templates` via a native `--include` over `tests/templates/` + `tests/agents/test_cli.py` (native form → no charter row) | hygiene | M | direct | stretch 15 |
| 3.5 | B10: one change package MODIFYING Non-Goal Exclusion (date-bound text), Scope Integrity, Quality Gate Fidelity (four rows), :335; executed as **one** PR with the `CUT_MODULES` entries, the `hf_space/src/<pkg>` scrub, `ARCHITECTURE.md`, and the gate steps | hygiene | M each | `openspec-change` | D1; ≤ 2 deletions this cycle |
| 3.6 | Deprecation ledger `docs/migration/deprecations.md`; guard imports each row under `catch_warnings` and asserts the warning; `remove_in ≤ version` → fail; the ~25 silent shims get warnings | hygiene | M | direct | stretch 7 |
| 3.7 | `lm_studio` YAML: delete the blocks if the 12-key block equals `LMStudioConfig` defaults after `apply_backend_defaults` | hygiene | S | direct | stretch 16 |
| 3.8 | `hf_space/` single-sourcing (B14) | — | L | `frozen` + `openspec-change` (:326) | deferred (D4) |

### WS4 — Layering

| ID | Item | Mission | Size | Route | When |
|---|---|---|---|---|---|
| 4.1 | Tiers (L0 `core, constants, seeding, device, templates, math_kernel, backend`; L1 `pde, refinement, mcts, games, modeling, physics, data, engines`; L2 `training, research, agents, distributed, integrations, experiments`; L3 `poc, tools, demos, deployment, alphagalerkin, dashboard/, scripts/`) as an ADR **and** in `ARCHITECTURE.md` (it owns layering); import contract with the 12-site exemption list (`pde → research` 5, `research → experiments` 3, `agents → poc` 2, `training → tools` 2), shrink-only | core loop | M | `ADR` | stretch 17 |
| 4.2 | Lazy-import baseline 154 → ≤ 60; break `{data,training}` by moving `Experience` to `src/data`; B1 = the 19 shim sites | hygiene | M | direct | stretch 18 |
| 4.3 | Registries: retire `core.Registry` via the 3.1 shim; `ScenarioRegistry`/`GameRegistry` onto `templates.BaseRegistry`; lifecycle per D13 (a change supersedes ADR 0005) | defect class | L | `ADR` | deferred (D13) |
| 4.4 | Unify the two logging modules; migrate ~2 CLI dataclass configs; slim hash mixin, not a `BaseModuleConfig` rebase; a `compute_hash()` change invalidates every sidecar → hash-pin protocol | hygiene | M | direct | deferred |
| 4.5 | Extend `[project.scripts] alphagalerkin` rather than add parallel names | hygiene | S | direct | deferred |
| ~~4.6~~ | `src` rename | — | — | — | cut |

### WS5 — Coverage

| ID | Item | Size | Route | When |
|---|---|---|---|---|
| 5.1 | `templates` gate (with 3.4) | S | direct (native form) | stretch 15 |
| 5.2 | `math_kernel` 61.5 %: `tests/math_kernel/` into `test-jax`; below the skill's 75 % triage line → D15 decides gate-as-tripwire vs triage first | M | owner (D15) | stretch 19 |
| 5.4 | `backend`: delete-first, then ratchet 54 → `floor(measured) − 2` in the existing inline-coveragerc step; `jax_backend.py` fate rides #118 / ADR 0003 (D16) | M | direct (ratchet) | stretch 20 |
| 5.5 | `deployment`: tests for `export_onnx.py`/`quantize.py`/`runtime.py`, gate = `floor(measured) − 2` after they land (≥ 60 is the aspiration, not the gate); blocked on 0.12 | L | direct | deferred |
| 5.6 | Planted-defect runner: `tests/docs/mutations/<guard>/<n>.patch` + expected-failing nodeid, applied in a `git worktree`; vacuity guard; mutmut only for `tests/support/` | M | direct | stretch 21 |
| 5.7 | Ratchet rule per D7, recorded in `config/coverage_ratchet.yaml`, checked by the monthly Routine (never calendar-red in `ci-success`) | S | direct + amend audit §7.6 #4 and the skill | stretch 5 |
| 5.8 | Stray top-level tests mirror guard (`tests/test_{data_generation,operator_training,physics_solvers}.py`); 7 duplicated conftest fixture names; inventory of the 20 files calling `*Registry.clear()`; RSS ceiling on the chess E2E half (report first) | S each | direct | stretch 22 |

### WS6 — Hardening

| ID | Item | Size | Route | When |
|---|---|---|---|---|
| 6.1 | `src/distributed/worker.py` — all four pickle sites (:415 `dumps`, :429 `loads`, :464 `loads`, :467 `dumps`): `torch.save` of a primitives-and-tensors dict (`board_state`, `board_size`, `target_policy`, `target_value`, `metadata` with non-primitive values rejected at serialise time) into `BytesIO`, loaded with `weights_only=True` and rebuilt into `Experience`, or admit `Experience` via `extra_safe_globals` (the `SAFE_DISTRIBUTED_GLOBALS` pattern). The property is "no path uses the unrestricted unpickler" (the container is still a pickle stream by format); a marker payload must not execute on any of the four paths | M | direct | stretch 10 |
| 6.2 | Graceful-shutdown spike for training | M | direct | deferred |
| 6.3 | `.gitleaks.toml`: after #135 and a local full-history `gitleaks detect` with the narrowed config (CI's action scans only the push/PR range), drop `tests/.*` and `docs/.*` (no key-shaped strings under either at HEAD) | S ×2 | direct | stretch 23 |
| 6.4 | `supply-chain` job, report-only, outside `ci-success`: `security` extra with `pip-audit` + `bandit`; `uv export --frozen --no-emit-project --no-hashes -o req.txt` (default `requirements.txt` format; skip the git extra); `pip-audit -r req.txt --no-deps`; `bandit -ll -r src scripts`; SBOM once a release workflow exists | S | direct | stretch 24 |
| 6.5 | Rename `test_path_traversal_in_config`; add the `ValueError` on non-mapping YAML its CWD accident revealed | S | direct | stretch 7 |

### WS7 / WS8 — Docs, governance, process

| ID | Item | Size | Route | When |
|---|---|---|---|---|
| 7.2 | Regression Surface table → `docs/regression-surface.md`; the owner is `openspec/project.md`'s rank-3 row + CLAUDE.md's header (not a Requirement); four guard constants, nine skills, link rewrite for `mkdocs --strict` | M | direct (D6) | deferred |
| 7.3 | Planning docs → the existing `docs/archive/plans/`; live/historical index | S | direct | stretch 25 |
| 7.4 | Machine-check the `src/` package count in `ARCHITECTURE.md` and the Regression Surface row count | S | direct | stretch 25 |
| 7.5 | `check_doc_links.py` inline-span resolution, report-only | M | direct | deferred |
| 8.2 | Extend `.github/PULL_REQUEST_TEMPLATE.md` (already the definition of done) with "guard mutation recorded" and "no threshold widened" | S | direct | stretch 26 |
| ~~8.3~~ | CODEOWNERS tiers | — | — | cut |

---

## 7. Sequencing

**MVC (weeks 1–3):** R-01 → R-02 → R-03 → R-04a → R-05 → R-04b → R-06 →
R-07 → R-08 → R-09 → R-10 → R-11 → R-12 → R-13 → R-14. Dependencies: R-04b
after #136/#103 (R-01); R-06, R-07, R-08 and R-12 after R-04a; R-06 after the
R-03 re-measure in the new `lint` environment; R-09 after ≥ 2 green `secrets`
runs under gitleaks v3 (#135); R-13's charter row after its CI step; R-14 easier
after R-04b. Nothing in stretch starts before R-04a and R-05 are on the default
branch.

**Stretch (weeks 4–8), in order:** 1 0.6a → 2 0.6b → 3 3.1 → 4 0.12 → 5 5.7 →
6 2.3 → 7 {0.9, 3.6, 6.5} → 8 2.2 → 9 1.5 → 10 {1.3, 6.1} → 11 2.5 → 12 2.6 →
13 3.2 → 14 3.3 → 15 {3.4, 5.1} → 16 3.7 → 17 4.1 → 18 4.2 → 19 5.2 → 20 5.4 →
21 5.6 → 22 5.8 → 23 6.3 → 24 6.4 → 25 {7.3, 7.4} → 26 8.2 → 27 0.14; D1
decided by week 4 with at most two deletions executed (`prototyping`, `analysis`).

**Deferred (with the unblocking decision):** 1.2, 1.4, 1.6 (touch-triggered);
2.4; 3.8 (D4); 4.3 (D13); 4.4; 4.5; 5.5 (after 0.12); 6.2; 7.2 (D6); 7.5;
remaining D1 deletions.

**Cut:** 1.8, 4.6, 8.3.

---

## 8. Targets

Guarded by the end of the MVC:

| KPI | Baseline | Target | Guard |
|---|---|---|---|
| Soft `continue-on-error` steps, all workflows | 3 | 2 (backend audit; ONNX until #128 — 0.12) | `test_soft_gates` (new) |
| PR-firing jobs outside `ci-success.needs` | 2 (`focus`, `secrets`) | 0 | `test_ci_success_hard_gates` (new) |
| Soft branches inside `ci-success` | 1 (transfer) | 0, or an accepted-deviation row | script parse / charter guard |
| `mypy --strict` unsuppressed errors | 7 | 0 | hard CI step |
| Fast lane hermetic | no | yes | `test_fast_lane_is_hermetic` (new) |
| Lockfile | none | `uv.lock` checked | `uv lock --check` |
| Artifacts frozen | no | manifest | `test_artifact_manifest` (new) |
| `[Unreleased]` headers | 2 | 1; tag `v0.4.0` | `test_changelog_headers` (new) |
| Unlisted `src/` modules > 600 lines | — | 0 | `test_module_size_budget` (new) |
| Ungated `src/` packages | 3 | 2 | integrity guard + per-package gate test |
| Default-branch CI wall-clock | 19.3 min | ≤ 15 min (< 12 after 0.14) | report (Routine) |

Measured by `scripts/measure_shape.py` and gated shrink-only from R-11 on
(`config/shape_baseline.yaml` is the source of truth; the hand counts in rev 3
read 10 / 25 / ~25 for the last three because they counted conditional
expressions, guarded import statements, and an eyeballed shim list rather than
the script's units — bare `torch.device("cuda")` sites, `try` statements whose
every protected import is a hard dependency, and files): complexity 162,
`PLR2004` 159, library `T201` 17, lazy imports 154, ad-hoc device resolution
**11**, orphan modules 21, dead `ImportError` guards **26**, shims without
warning **26**, mirror divergence 98.

Not gated this cycle and said so: vulture 60 % count, dependabot age, CI p95.

---

## 9. Owner decisions

| # | Decision | Recommendation | Route | Due |
|---|---|---|---|---|
| D1 | B10 fates, one row per package; **supersedes** audit §7.6 #1 (`curriculum` spike, not delete; `deployment` keep, not deprecate) | delete `prototyping`, `analysis` now; `tournament`, `curriculum` after a one-day spike each; keep `deployment` | `openspec-change` (four Requirements + :335) as one PR with the `hf_space` scrub | week 4 |
| D2 | Lockfile shape | `uv`; `cpu` extra + `conflicts` + `sources`; ADR 0006 ratifies `.gitignore:203` | `ADR` | week 1 |
| D3 | Hard mypy | yes, after R-04a/b and the re-measure | `openspec-change` (:328) | week 2 |
| D4 | `hf_space/` single-sourcing | park (frozen; no consumer this cycle) | — | — |
| D5 | `src` rename | dropped | — | — |
| D6 | Regression Surface table ownership | defer to a docs-themed cycle; route is `direct` via `openspec/project.md` | direct | — |
| D7 | Coverage ratchet rule | either `min(current + 2, floor(measured) − 2)` capped 85 per quarter (the recorded +2/quarter rule) or supersede audit §7.6 #4 and the `add-coverage-gate` skill in the same PR; add the skill's < 75 % triage clause | direct | week 3 |
| D8 | Release cadence | cut 0.4.0 now; add "monthly or at each `openspec/changes/` archive" to `RELEASING.md` | direct | week 2 |
| D9 | What R-09 re-scopes | nothing in `config/focus.yaml` this cycle; R-09 promotes the jobs and rewrites row :334 + `docs/FOCUS.md` | direct + `openspec-change` (prose) | week 2 |
| D10 | Transfer tripwire | characterise (three locked re-runs); flip hard or record an accepted-deviation row | direct / `openspec-change` | week 2 |
| D11 | Artifact-freeze manifest | adopt; CSV/JSON hashed, PNG presence-only; `lambda_scheduling.*` pinned; `poc_headline.example.json` excluded | direct | week 1 |
| D12 | B9 seed-stride unification | defer (rewrites `config/baselines/*.json` → claims-ledger + manifest regen) | — | next cycle |
| D13 | Registry lifecycle | keep ADR 0005 unless new measurements beat audit §6's table; any change is a superseding ADR | `ADR` | before 4.3 |
| D14 | Chess E2E leak | `tracemalloc`/RSS spike first | direct | stretch |
| D15 | `math_kernel` at 61.5 %: gate as a disclosed tripwire or triage first | triage first (`add-coverage-gate` < 75 % rule) | direct | before 5.2 |
| D16 | `jax_backend.py` fate | rides PR #118 / reserved ADR 0003, not D1 | — | with R-01 |
| D17 | Tier assignments in 4.1 (nine packages were unassigned in rev 1) | as listed in 4.1 | `ADR` | before 4.1 |

---

## 10. Disclosed gaps and non-goals

Already in the repo: `RELEASING.md`, `SECURITY.md`, `CODEOWNERS`,
`CONTRIBUTING.md` + PR template, `docs/adr/` (4 ADRs, 2 CI-enforced),
`dependabot.yml`, gitleaks, `docs/ci-exclusion-ledger.md`, nightly `test-slow`,
a monthly stale-PR Routine.

Still missing after this cycle:

- **CI SLOs**: no p50/p95 or flake-rate measurement; add to the monthly Routine.
- **Release automation**: no on-tag workflow, no signed artifacts; R-08 cuts by
  hand; the SBOM in 6.4 waits on a release workflow.
- **Dependency policy**: no upper bounds, license allowlist or update SLA; D2
  plus a `SECURITY.md` paragraph closes most of it.
- **Bus factor**: one owner; nothing repository-side fixes it.
- **Nightly on-call**: `test-slow` at 04:00 UTC has no failure notification.
- **ADR discipline**: tiers (D17), lockfile (D2), registry (D13) are routed to
  ADRs; `ARCHITECTURE.md` prose is not a decision record.
- **Branch protection**: whether it requires only `CI Success` is unverifiable
  from the repo; state it in `CONTRIBUTING.md`.
- **This document is outside the retracted-claims guard's scan set**
  (`docs/business`, `docs/doe_genesis`, `README.md`, `CLAUDE.md`); a grep for
  every guarded phrase is clean, and adding it to `SCAN_FILES` is a one-line
  option.

This plan does not re-argue retracted claims, does not bundle frozen-track work
with core-path work, does not raise any gate by editing a number, and proposes
no product scope.

---

## Implementation ledger

Appended in rev 4. One row per ticket, in landing order; commit SHAs are on
PR #151's branch. "Guard" is the test file that turns the ticket into a check.

| Ticket | Landed | Commit(s) | Guard | Notes |
|---|---|---|---|---|
| R-02 hermetic fast lane | yes | `cafd998` (+ floor fix in `a9092f7`) | `tests/docs/test_fast_lane_is_hermetic.py` | first draft imported `tomllib` against the 3.10 floor; `test_python_floor_compatibility` caught it |
| R-03 mypy to zero unsuppressed | yes | `1a37665` | pyproject mypy override + CHANGELOG | 5 remaining diagnostics scoped to frozen `codec.py` by a `disable_error_code` override; remove when the freeze lifts |
| R-10 module size budget | yes | `b9fe859` (merge) | `tests/docs/test_module_size_budget.py` | 44 `src/` rows at ceiling 600, 8 test rows at 1000; 5/5 mutations |
| R-11 shape baseline | yes | `d80d8fe` (merge), `3eea67c` (regenerated at merged tip `efdf4c871ca9bf92889980a4c350ae7cee0fe2c8`) | `tests/docs/test_shape_baseline.py`, `tests/scripts/test_measure_shape.py` | 11 / 26 / 26 for the last three metrics (units, not drift — see §8); the committed YAML records post-landing provenance, while §1/§8 keep the pre-implementation baseline numbers |
| R-09 focus + secrets hard gates | yes | `a9092f7` | `tests/docs/test_ci_success_hard_gates.py` | charter row amended via `openspec/changes/focus-secrets-merge-gate/`; 6 mutations after review |
| R-12a lint / typecheck split | yes | `7d62012` | `tests/docs/test_ci_success_hard_gates.py` (`typecheck` in the expected set) | `test-slow` deliberately keeps `needs: test-fast`; `ci.yml` install steps now 13 |
| R-12b coverage-gates 4-way shard | yes | `7d62012` | `tests/docs/test_coverage_gate_shards.py` | gate set derived from coverage commands, not names; 5 mutations after review |
| R-13 eval_harness marker + tripwire gate | yes | `b5dc832` (merge `694188c`) | `tests/docs/test_eval_harness_gating.py` | `--cov-fail-under=1` is a tripwire; set `floor(measured)-2` from the first green `test-extras` run |
| R-08 CHANGELOG structure | yes | `d6550c1` (merge) | `tests/docs/test_changelog_headers.py` | 206/206 bullets preserved; no version bump (owner's call, D8) |
| R-04a lockfile foundation | **blocked** | `impl/r04a` @ `a9cd02d`, unmerged | `tests/docs/test_uv_lock.py` | `uv lock` needs egress to `download.pytorch.org`, denied by the implementing sandbox's policy; the branch carries the `cpu` extra, pins, ADR 0006 and the guard, and lands once the lock is generated on a host with egress |
| R-05 artifact manifest | in progress | `impl/r05` | `tests/docs/test_artifact_manifest.py` | — |
| R-14 worktree rule | yes | `1b5be92` (merge) | `tests/claude/test_worktree_rule.py` | 5/5 of six planted; the sixth survives by design (Bash ⇒ anchor, not the converse) |
| R-04b, R-06, R-07 | not started | — | — | R-04b/R-06 wait on the lock; R-07 needs three stable transfer runs after it |
| R-01 | owner | — | — | merge dependabot, close #139, disposition stale PRs |

## Appendix A — Reproduction commands

All from the repository root on `ba03b43` after `pip install -e '.[dev]'`,
except the committed `config/shape_baseline.yaml` artifact noted above, whose
`generated_from` is `efdf4c871ca9bf92889980a4c350ae7cee0fe2c8`.
There is no lockfile yet, so two fresh installs can resolve different versions;
the set these numbers were measured with is Python 3.11.15, torch 2.14.0+cu130,
numpy 2.4.6, scipy 1.17.1, pydantic 2.9.2, coverage 7.16.0, pytest 9.1.1,
ruff 0.15.8 (the CI pin). `uv` and `vulture` are **not** in any extra —
`pip install uv==0.8.17 vulture==2.16` first. CI's `lint` job installs a pinned
minimal set on 3.11, so its mypy count can differ. Static counts (grep/AST/ruff)
do not depend on the package set; coverage percentages and the mypy count can.
GitHub-side figures come from the API commands below.

```bash
# CI timeline (GitHub API; started_at/completed_at per job)
curl -s https://api.github.com/repos/ianshank/AlphaGalerkin/actions/runs/34476775571/jobs?per_page=30 \
  | python -c "import json,sys;[print(j['name'],j['started_at'],j['completed_at']) for j in json.load(sys.stdin)['jobs']]"
#   push run 34476775571 on ba03b43: 16 jobs; lint 12:27:01-12:29:34; test-fast(3.11) 12:29:38-12:38:14;
#   test-e2e 12:38:18-12:46:11; ci-success 12:46:17 -> 19.3 min; coverage 12:29:37-12:39:19 (9:42);
#   coverage-gates 12:29:37-12:43:28 (off path). PR run 34602317653: coverage 13:12:33-13:24:57 (12:24).
curl -s "https://api.github.com/repos/ianshank/AlphaGalerkin/pulls?state=open&per_page=50" \
  | python -c "import json,sys;[print(p['number'],p['user']['login'],p['title'][:60]) for p in json.load(sys.stdin)]"
grep -c 'cache: "pip"' .github/workflows/ci.yml                     # 12
grep -n "continue-on-error" .github/workflows/ci.yml                # 171, 183, 1451
grep -nF 'needs: [' .github/workflows/ci.yml                        # ci-success list (10)
grep -c "name: Install dependencies" .github/workflows/*.yml        # ci 12, regression-surface 7, sbir-demo 2, phase2 1 (frozen);
                                                                    # docs.yml's single install step (line 49) is named differently -> 23 total

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

# Complexity / magic values / prints (rules not selected in pyproject.toml today)
ruff check src/ --select C901,PLR0912,PLR0913,PLR0915,PLR0911 --statistics   # 162
ruff check src/ --select PLR2004 --output-format json | python -c \
  "import json,sys;d=json.load(sys.stdin);print(len(d))"                     # 159 (136 int, 23 float by value)
ruff check src/ --select T201 --output-format json | python -c "
import json,sys,collections; c=collections.Counter()
for r in json.load(sys.stdin): c['main' if '__main__' in open(r['filename']).read() else 'lib']+=1
print(c)"                                                                    # main 99, lib 17

# Module sizes
find src -name '*.py' -exec wc -l {} + | awk '$1>900 && $2!="total"' | wc -l    # 5
find src -name '*.py' -exec wc -l {} + | awk '$1>600 && $2!="total"' | wc -l    # 44
find tests -name '*.py' -exec wc -l {} + | awk '$1>1000 && $2!="total"' | wc -l # 8

# Dead / orphan code
vulture src/ --min-confidence 80 | wc -l      # 14
vulture src/ --min-confidence 60 | wc -l      # 893
# Orphan modules, three variants:
#   production-orphan raw            24 modules / 7,215 LOC
#   production-orphan excl. CLI      21 / 6,274   (the R-11 / 3.1 figure)
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
grep -rn "src.core.registry\|from src.core import" src/ dashboard/ scripts/ --include=*.py \
  | grep -v "^src/core/"                                              # none; src/core/__init__.py re-exports Registry

# Lazy in-function src.* imports by AST, excluding TYPE_CHECKING -> 154 in 65 files
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

# Device / prints / guards / duplication
grep -rnE "\"cuda\"|'cuda'" src/ --include=*.py | wc -l                                     # 121
grep -rnE "\"cuda\"|'cuda'" src/ --include=*.py | grep -vc "^src/video_compression/"        # 75
grep -rn "^def resolve_device\|def _resolve_device" src/ --include=*.py | wc -l             # 6 named resolvers
grep -rlE "src\.poc\.device" src/ scripts/ dashboard/ --include=*.py \
  | grep -vE "^src/(device|poc/device)\.py|^src/video_compression/" | wc -l                 # 19 shim consumers
grep -rn "except ImportError" src/ --include=*.py | wc -l                     # 76 (9 in src/video_compression)
grep -rn "^def export_csv\|^def export_plot" src/ | wc -l                     # 8 (4 pairs)
grep -rn '"--allow-unsafe-pickle"' scripts/ src/ --include=*.py | wc -l      # 5 argparse sites (2 frozen)
grep -rn "DeprecationWarning" src/ --include=*.py | grep -v "^\s*#" | wc -l   # 5 lines / 3 sites
grep -rc "lm_studio:" config/scenarios/*.yaml config/agents/*.yaml | grep -v ":0"   # 8 blocks
grep -rnE "\"https?://[^\"]+\"" src/ --include=*.py | grep -v "^\s*#" | wc -l   # 8 URL literals
grep -rhoE "class \w+Registry\b|\b[A-Z_]+_REGISTRY\b" src/ --include=*.py | sort -u | wc -l   # 10 (heuristic)
grep -rliE "back-?compat|backwards.compat" src/ --include=*.py | wc -l        # 28 shim files (heuristic)

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
grep -oE "fail-under=[0-9]+" .github/workflows/ci.yml | grep -v "=8[5-9]\|=9" | wc -l   # 13 hits; :1582 is a comment -> 12 gates

# Docs / release
grep -n "^## \[Unreleased\]\|^## \[0.3.0\]" CHANGELOG.md       # 3, 130; 842, 1648
wc -l CHANGELOG.md CLAUDE.md                     # 1864, 898
git tag | wc -l                                  # 0
grep -n "0\.1\.0" RELEASING.md; grep -n "pinned" SECURITY.md; grep -n "^version" pyproject.toml

# Hermeticity trial (plugin installed in the scratchpad only; 3 directories, 542 tests)
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
| B4 god-file splits | 5 of 7 done | WS1 (1.3, 1.5 scheduled; 1.2, 1.4, 1.6 touch-triggered) |
| B5 config/logging unification | open | 4.4 (logging half) |
| B6 mypy hard gate | open | R-03, R-06 |
| B7 composite action / args file | ledger half done | R-04b; 0.6a adds a `-m` copy to the ledger |
| B9 seed-stride unification | open | D12 (deferred) |
| B10 dead packages | open; charter :335 records disposition | 3.5 / D1 |
| B11 YAML dedup | open | 3.7 (delete-first) |
| B13 CHANGELOG / release | open | R-08 |
| B14 `hf_space` single-source | open (frozen) | 3.8 / D4 |
| B15 script CLI boilerplate | open | 3.4 |
| B16 fixture prune / shim policy | open | 3.6, 5.8 |
| B21 god-file-split skill | done | used by WS1 |
| B22 subagent worktree isolation | done (2026-09-11, R-14) | R-14 |
| B35 network-fetching unit tests | open; reproduced | R-02 |
| B37 eval-harness ungated | open | R-13 |
| CHANGELOG parked LBB `* 10` / FNO `128` | open | 2.2 |
| 2026-08-19 `pickle.loads` on `all_gather` | open | 6.1 |
| 2026-08-19 SIGINT handling | deferred | 6.2 |
| 2026-08-19 `evaluation.py:321` outside the chokepoint | open (`pyproject.toml:46` names it; consistency, not security) | 1.3 |
| CLAUDE.md Next Steps: wall-clock ratio assertions | open; reproduced under load | 0.6a, 0.6b |
| CLAUDE.md Next Steps: `.gitleaks.toml` allowlist | open | 6.3 |
| CLAUDE.md Next Steps: `test_path_traversal_in_config` | open; stronger than documented | 6.5 |
| CLAUDE.md Next Steps: `check_doc_links.py` inline spans | open | 7.5 |
| CLAUDE.md Next Steps: second workflow parser | open | R-09 adds a consumer of the shared parser; convergence deferred with 7.2 |
| CLAUDE.md Next Steps: chess E2E memory leak | open | D14, 5.8 |
| `docs/FOCUS.md`: promote `focus`, re-scope tracks | open (promised there) | R-09 / D9 |
