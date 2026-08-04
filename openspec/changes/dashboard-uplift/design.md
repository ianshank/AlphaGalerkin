# Design: `dashboard-uplift`

## Technical Approach

The dashboard renders numbers from three sources: Pydantic defaults in `dashboard/config.py`,
live scenario runs, and hardcoded markdown strings. Only the middle one is trustworthy today.
WS1 makes the first traceable to committed artifacts and deletes the retracted framings from
the third; WS2 makes that property machine-checked so it cannot silently regress.

The guard is placed in `tests/docs/test_charter_alignment.py` rather than `tests/dashboard/`
because the charter's own meta-guard (`test_every_requirement_has_a_guard`) checks the
Requirement↔guard mapping **in both directions** — a Requirement whose guard lives elsewhere
would either fail that check or require weakening it. The AQA tests that need the real Pydantic
objects stay in `tests/dashboard/`, where gradio is already a test dependency.

WS3–WS6 are designed but not implemented. The sequencing matters: WS3 must precede WS4, because
a registry-driven scenario tab that imports `src.poc.scenarios` while the mirror shadows `src/`
would enumerate the drifted copies.

## Architecture Decisions

### AD1 — Committed figures are static Pydantic defaults plus an AQA agreement test

**Decision.** `TransferMilestone` carries the committed numbers as literal `Field` defaults. A
test asserts they agree with `config/baselines/transfer_ci.json` within that file's own
`tolerance_pct`.

**Rationale.** Reading the baseline JSON in a `default_factory` was considered and rejected.
Import-time file I/O makes the config non-hermetic — it becomes cwd-sensitive and acquires a
failure mode (missing file) at construction rather than at test time. The repository already has
an idiom for exactly this problem: spec↔config agreement tests, as in
`specs/transfer_baseline_compare.spec.md` AC6. Static defaults keep the config a pure data
declaration and move the drift check to where drift checks belong.

**Consequence.** A benchmark rerun that updates `transfer_ci.json` will fail the AQA test until
the dashboard default is updated too. That is the intended coupling.

### AD2 — The new Requirement carries no delimited register

**Decision.** *UI Claim Fidelity* is prose plus a guard, with no `<!-- charter:*:start -->`
region.

**Rationale.** *Novelty Claim Discipline* sets the precedent: not every Requirement needs a
machine-readable table. The figures this Requirement governs already live in the evidence
register; a second table would duplicate them and create a third place to edit when a benchmark
changes. `_REGIONS` in the guard module is unchanged, so the region meta-guard needs no update.

### AD3 — The guard loads `dashboard/config.py` standalone, never the package

**Decision.** The guard uses `importlib.util.spec_from_file_location` to execute
`dashboard/config.py` in isolation under a private module name.

**Rationale.** `dashboard/__init__.py` imports `dashboard.app`, which imports gradio and every
tab module. `tests/docs/` deliberately imports nothing heavy — its only subprocess is the
capability guard's registry read, which is bounded by an explicit timeout for that reason. A
plain `from dashboard.config import …` would pull the whole UI stack into the charter guard.
`dashboard/config.py` imports only `typing` and `pydantic`, so standalone execution is cheap and
total.

**Consequence.** The guard degrades honestly: if `dashboard/config.py` ever grows a heavy
import, the guard fails loudly rather than silently skipping.

### AD4 — Un-shadowing requires relocating modules, not reordering `sys.path`

**Decision.** WS3 relocates the four Space-only modules into the maintained tree. Reordering
`sys.path` alone is not a fix and would break the Go tab.

**Rationale.** Root `src/` and root `config/` both contain `__init__.py` — they are *regular*
packages, not namespace packages. Whichever `sys.path` entry is found first claims the entire
namespace; there is no merging across entries. `dashboard/tabs/game_tab.py` imports
`config.board`, `src.endgame`, `src.game_manager`, and `src.rendering.board_renderer`, none of
which exist in the main tree (`config/board.py` is absent at root). So putting `ROOT` first —
the obvious reading of the shadowing bug — makes the Go tab raise `ModuleNotFoundError`
immediately. The dashboard's own tests never catch this: `tests/dashboard/conftest.py` puts
`ROOT` first *and* mocks `_ensure_loaded`, so the real import path is never exercised.

**Consequence.** WS3 creates a package under `src/`, which the charter's *Scope Integrity*
Requirement binds to `ARCHITECTURE.md`'s package map. WS3 therefore needs its own charter delta
and cannot ride along with WS1/WS2. It must also be scoped as *relocating what the dashboard
imports* — not single-sourcing the mirror, which the charter's deviation register declares out
of scope.

### AD5 — `results/lambda_scheduling.{csv,png}` stays out of the Results tab

**Decision.** WS4's Results tab renders the three artifact pairs with live producer scripts and
excludes `lambda_scheduling`.

**Rationale.** The charter's deviation register already discloses that these outlive their
producer — the `thermo` module was cut on 2026-07-22 and the artifacts were kept as the only
in-tree evidence of a negative result. Rendering them in a live UI would re-promote an orphaned
artifact to a headline, which is the failure mode this whole change exists to close.

### AD6 — The physics demo's zero-model output is relabelled now, fixed in WS4

**Decision.** WS1 makes `PhysicsDemo.predict()` label its output as a placeholder when no model
is loaded. WS4 wires the real checkpoint through.

**Rationale.** `predict()` returns zeros when `self.model is None`, and both `dashboard/app.py`
and `hf_space/app.py` construct the tab with `model=None`. The "transfer MSE" it reports is
therefore `mean(ground_truth²)` — a property of the *data*, mislabelled as model error. That is
an evidence violation independent of the transfer figures, so it cannot wait for WS4; but
loading a checkpoint into two demo tabs is genuinely WS4's job.

## Data Flow

```
config/baselines/transfer_ci.json ──┐
                                    ├─► AQA test (tests/dashboard/test_config.py)
dashboard/config.py TransferMilestone┘         │
         │                                     └─► charter guard: figures agree ± tolerance
         ├─► poc_tab.show_transfer_milestone ──► 3-arm bar chart + operator 9→13→19 curve
         └─► poc_tab tab blurb ────────────────► honest 19×19 framing

dashboard/**/*.py ──► charter guard: no FABRICATED_FIGURE, no RETRACTED_BLANKET_CLAIM
```

## File Changes

| File | Change |
| --- | --- |
| `openspec/specs/project-charter/spec.md` | New `### Requirement: UI Claim Fidelity` |
| `openspec/changes/dashboard-uplift/**` | This change package |
| `dashboard/config.py` | `TransferMilestone` defaults → committed values; three new baseline fields; `0.000209` literal removed from the description |
| `dashboard/tabs/poc_tab.py` | `show_transfer_milestone` rewritten; synthetic curve and `N× better` deleted; tab blurb corrected |
| `dashboard/app.py` | About-table transfer row → committed figures + zero-retraining framing |
| `hf_space/app.py` | Zero-retraining framing beside the board-size table (text-only; Gradio 4.44.1) |
| `src/demos/physics_demo.py` (+ mirror) | Placeholder label when `model is None` |
| `tests/docs/test_charter_alignment.py` | `_GUARDED` entry + `test_ui_claims_match_committed_artifacts` |
| `tests/dashboard/test_config.py` | AQA agreement test; new-field assertions |
| `tests/dashboard/test_poc_tab.py` | Assert honest framing present, retracted strings absent |
| `CHANGELOG.md`, `CLAUDE.md` | Unreleased entry; Regression Surface row |

## Alternatives Considered

**Put the guard in `tests/dashboard/`.** Rejected: the charter's both-directions
Requirement↔guard meta-guard requires the guard to be resolvable in the charter guard module.
Splitting it would mean weakening that meta-guard — trading a real invariant for file tidiness.

**Delete `show_transfer_milestone` entirely and wait for WS4's Results tab.** Rejected: it
leaves the dashboard with no transfer story for the interval, and the fix is cheap. The rewrite
also gives WS4 a working precedent for rendering committed artifacts.

**Ban the retracted figure across the whole repository, not just `dashboard/`.** Already done
for `hf_space/` by the mirror guard and for the charter by the retraction-marker check.
Extending the bare ban repo-wide would fail on `tests/support/cut_modules.py`, which must
*contain* the string to define it, and on `CHANGELOG.md`, which is immutable history. Scoping
the new guard to `dashboard/` closes the actual gap without those exemptions.
