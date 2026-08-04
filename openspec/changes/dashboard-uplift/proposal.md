# Proposal: `dashboard-uplift`

## Why

The charter's evidence standard (*Evidence-Backed Claims*) and its retraction rule (*Novelty
Claim Discipline*) are machine-enforced across the charter itself, `docs/related-work.md`,
`README.md`, and `hf_space/**`. They are **not** enforced anywhere in `dashboard/` — and the
dashboard is the project's most public surface.

The gap is not theoretical. An audit of the Gradio UI found four live violations:

1. `dashboard/config.py` ships `TransferMilestone.achieved_mse = {9: 2.5e-6, 13: 2.04e-4,
   19: 3.93e-4}`, attributed to `scripts/demo_transfer.py` — a script that writes only to
   `outputs/`. These are **uncommitted-spike numbers**, the exact class the charter calls
   inadmissible. The committed benchmark says 19×19 ≈ **2.3e-3**, ~6× worse.
2. `dashboard/tabs/poc_tab.py::show_transfer_milestone` renders `MILESTONE ACHIEVED` and
   annotates each bar `N× better` against an arbitrary 0.05 threshold — currently **127×**,
   **245×**, and **20000×**. `specs/transfer_baseline_compare.spec.md` retracts precisely this
   framing: the honest comparison is against a *retrained CNN baseline*, which the operator
   **loses** to by ≈14×.
3. The same function plots a `np.random.default_rng(7)` curve titled *"Training curve (9×9
   Poisson data)"* with no disclaimer — a fabricated curve presented as a measurement.
   `training_tab.py` labels its simulated curves honestly; this one does not.
4. The tab's own blurb prints `min(achieved_mse.values())` — the **9×9 in-distribution** number
   — as the zero-shot transfer result, and `dashboard/app.py` repeats the spike figure in its
   About table.

This is the same failure mode as the fabricated `0.000209 / 240×` headline retracted on
2026-07-22, surviving in the one place no guard looks.

Separately, the UI has drifted from the project it exists to showcase. It exposes 3 of the 10
registered scenarios, renders none of the four committed artifact pairs in `results/`, prefers
a knowingly-diverged code mirror to the maintained tree, and sits outside every CI quality gate.

## What Changes

**Claims first.** Every figure the UI renders is replaced with one traceable to a committed
artifact, and the retracted framings — "milestone achieved", "N× better than threshold", the
synthetic training curve — are removed. The honest result is stated as the charter states it:
the operator transfers with **zero retraining**, and a retrained CNN is ≈14× more accurate.

**Then the charter grows a guard for the surface it was missing.** A new Requirement, *UI Claim
Fidelity*, with a guard test registered in `tests/docs/test_charter_alignment.py::_GUARDED`.
It bans retracted figures across `dashboard/**` and asserts the dashboard's transfer numbers
agree with `config/baselines/transfer_ci.json` within that file's own tolerance — so the next
spike figure fails CI rather than shipping to users.

**Then the uplift, phased.** Four further workstreams are designed here and delivered
separately: un-shadowing the mirror, a registry-driven scenario tab plus a Results tab over the
committed artifacts, a clickable Go board, and bringing `dashboard/` inside the CI quality
gates. They are specified now so the sequencing and its charter consequences are visible; only
WS1 and WS2 land with this change.

## In Scope

- **WS1** — replacing spike figures and retracted framings in `dashboard/config.py`,
  `dashboard/tabs/poc_tab.py`, `dashboard/app.py`, `hf_space/app.py`, and the physics demo's
  mislabelled zero-model output.
- **WS2** — the *UI Claim Fidelity* Requirement, its guard, and the AQA tests binding the
  dashboard's figures to `config/baselines/transfer_ci.json`.
- Design (not implementation) of **WS3–WS6**, recorded in `design.md` and `tasks.md`.

## Out of Scope

- `hf_space/` single-sourcing — an explicit charter non-goal. WS3 relocates only the four
  modules the dashboard imports, and is deliberately deferred because it triggers a scope-register
  amendment (see `design.md` AD4).
- Converging the Gradio 4.44.1 (Space) / ≥6.0 (dashboard) split. WS1's `hf_space/` edit is
  text-only for exactly this reason.
- Re-running any benchmark. This change makes the UI cite numbers that already exist; it
  produces no new ones.
- Rewriting `CLAUDE.md`'s milestone history — append-only, as established.
- `results/lambda_scheduling.{csv,png}` — already disclosed in the charter's deviation register
  as outliving its producer; WS4 excludes it rather than re-promoting it.

## Impact

- **Affected docs:** `openspec/specs/project-charter/spec.md` (one new Requirement),
  `CHANGELOG.md`, `CLAUDE.md` (Regression Surface row).
- **Affected tests:** one new guard in `tests/docs/test_charter_alignment.py`; extensions to
  `tests/dashboard/test_config.py` and `tests/dashboard/test_poc_tab.py`.
- **Affected CI:** none for WS1/WS2 — `tests/docs/` and `tests/dashboard/` are already collected
  by the blanket `pytest tests/` in the fast and coverage jobs. WS6 changes CI; it is deferred.
- **Risk:** low. No `src/` behaviour change beyond a labelling fix in `src/demos/physics_demo.py`.
  The figures move from favourable-but-inadmissible to unfavourable-but-committed, so the UI
  will read as a weaker result — that is the point, and the charter requires it.
