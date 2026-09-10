# Design: `hygiene-hardening-cycle`

## Technical Approach

Disclose the two owner decisions that the live charter currently mis-states,
then consume the hygiene ledger under a core-only allowlist. Code work is
sequenced after this delta is reviewable; this design records the decisions
a reviewer could disagree with, not the file-move recipe.

## Architecture Decisions

### AD1 — B10 packages stay; keep-reason is a deviation, not a cut

**Decision.** `src/prototyping/`, `src/analysis/`, `src/curriculum/`, and
`src/tournament/` remain in the scope register. The new deviation states they
are test-held and not production-wired, and that this is **not** a
2026-07-22-style cut. AGENT.md files in those packages carry the same sentence.

**Rationale.** Deleting ~14k LOC to “focus” already failed once
(`video_compression` cut 2026-07-22, restored 2026-07-23). Keeping them costs
CI time; a second cut-and-restore costs more. Wiring
`Trainer._run_checkpoint_tournament` onto `src/tournament` is explicitly out
of this cycle (that would be a keep-and-wire, a different decision).

**Retirement.** The row is removed when a dedicated change either (a) wires a
production caller in `src/` outside each package’s tests, or (b) cuts the
package through the charter’s Non-Goal Exclusion path with a CUT_MODULES
entry. Silent deletion is the failure mode.

### AD2 — FOCUS freeze-lift does not empty `frozen_tracks`

**Decision.** Amend the frozen-tracks deviation: the thesis freeze lifted on
the committed arena result (median ratio **0.9532**). `codec` and
`interactive-surfaces` remain paused until a follow-up edits
`config/focus.yaml`. Empty `frozen_tracks` is already rejected by the focus
config validator.

**Rationale.** `docs/FOCUS.md` already tells this story. Leaving the charter
row in the pre-arena tense is drift. Emptying the YAML in the same change as
core solver work would trip `scripts/check_focus.py` or, worse, disable the
split-attention gate while hygiene PRs still need it.

**Retirement.** Remove or rewrite the row when `config/focus.yaml` is
re-scoped (new frozen set, or a successor focus document). Promoting the
`focus` job into `ci-success.needs` is a separate CI-only PR.

### AD3 — `src/device.py` is a root module, not a Scope Integrity edit

**Decision.** Wave B1 adds `src/device.py` as a sibling of `src/seeding.py`.
No charter scope-register row. `ARCHITECTURE.md` prose (“three root-level
modules”) is updated in the B1 PR; that sentence is not machine-checked today.

**Rationale.** `tests/docs/test_architecture_map.py` enumerates
`src/*/__init__.py`. A package-map row for a plain module fails both
directions of the scope guard. Claiming a Requirement change for a file that
cannot appear in the register is governance theater.

### AD4 — Core-only allowlist; frozen tracks incidental

**Decision.** In-cycle paths: `src/mcts/`, `src/pde/`, `src/refinement/`,
`src/research/`, `src/training/`, `tests/docs/`, `.github/workflows/ci.yml`,
`Makefile`, `openspec/`, governance `docs/`, plus glue `src/poc/device.py` /
`src/poc/scenarios/*_compare*` and new `src/device.py`. Frozen-track edits
stay under the 20-line incidental budget. B1 must not edit
`src/video_compression/perf/device.py`.

**Rationale.** Owner scope is tighter than FOCUS. Parking modeling / backend /
eval_harness / dashboard leaves real debt; mixing it with god-file splits is
how this repo ships unattributable reds.

### AD5 — God-file splits freeze public `dir()` names, not raw `dir()`

**Decision.** Before converting a module to a package, freeze
`{n for n in dir(m) if not n.startswith("_")}`. `__all__` is one-directional
(public set ⊆ `__all__`). `del` submodule names in package `__init__.py`.
mypy overrides do not cascade — add `"<pkg>.*"`. Coverage `--include` after a
`.py` → package conversion must list both `*/pkg.py` (the `__init__.py`) and
`*/pkg/*`.

**Rationale.** PR #140 (`src/pde/operators`). Claiming raw `dir()` identity is
the overclaim peer review already killed (`__path__` / `__all__` appear on
packages). Measuring only the re-export shim is the B31 omit-collision class.

### AD6 — `Trainer` facade without `super().__init__()`

**Decision.** Extract evaluation / tournament / engine-eval / buffer-fill into
sibling modules. Do not call `BaseTrainer.__init__`. Do not reorder `_log`
binding to match BaseTrainer.

**Rationale.** `Trainer.__init__` documents that it does not call `super()`.
`_log` is bound *after* optimizer/scheduler. Calling `super()` is a behavior
change, not a split.

### AD7 — Compare helpers live in `_compare_common.py`, not `BaseScenario`

**Decision.** Shared teardown and name-lock for the four `*_compare` families
go in `src/poc/scenarios/_compare_common.py`. Name-lock is a function, not a
config field. Lazy `from src.research...` imports stay lazy.

**Rationale.** `BaseScenario.teardown` emptying CUDA cache would hit every PoC
scenario. Adding a config field would change `compute_hash()` and invalidate
committed baselines. Module-level research imports recreate the poc↔research
cycle the lazy imports exist to break.

## Non-goals (design-level)

- Research numbers, new arena CSV, trained-MCTS.
- mypy hard-gate, ONNX hard-gate, `focus`/`secrets` into `ci-success`.
- Seed-stride unification.
- B7 args-file extract, B37 eval_harness, B3 registry consolidation.
- Manufactured `HeatOperator.exact_solution`.
- Modeling LBB `* 10` / FNO `128` (parked with `src/modeling/`).
