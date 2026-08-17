# Project Charter Specification

> **Status:** Active · **Supreme** — on any conflict with another document in this repository,
> this charter wins. See [`openspec/project.md`](../../project.md) for the full precedence order.

## Purpose

AlphaGalerkin applies **resolution-independent continuous operator learning** (Galerkin
Transformers, FNet mixing) and **Monte Carlo Tree Search** to two problem domains that share one
search abstraction:

- **Board games** (Go, Chess) — zero-shot transfer across board sizes.
- **PDE solving** — MCTS-guided adaptive mesh refinement and Galerkin basis selection.

The connective tissue is MCTS: `src/mcts/` is the domain-agnostic engine, and each domain adapts
into it (`src/games/` via `GameInterface`, `src/pde/` via `src/pde/mcts_adapter.py`).

This charter exists because the project's identity was previously spread across six documents
that drifted independently, and because **numeric headline claims repeatedly outran their
evidence** — first the fabricated `0.000209 / 240×` transfer figure (retracted 2026-07-22), then
a retracted AMR win and an overstated transfer MSE that survived until 2026-07-31. Requirement 3
below is the direct, executable response.

Each Requirement carries a guard test in
[`tests/docs/test_charter_alignment.py`](../../../tests/docs/test_charter_alignment.py). A
Requirement without a guard is not a Requirement — it is a wish.

## Requirements

### Requirement: Scope Integrity

The charter's scope register SHALL enumerate exactly the `src/` packages documented in
[`ARCHITECTURE.md`](../../../ARCHITECTURE.md)'s package map, and every package it names SHALL
exist on disk.

The register below is intentionally a *pointer*, not a copy: `ARCHITECTURE.md` owns the per-package
purpose and maturity labels, and `tests/docs/test_architecture_map.py` already binds that map to
disk. This Requirement binds the charter to `ARCHITECTURE.md`, so a single root cause produces a
single failure.

<!-- charter:scope:start -->
| Package | Domain |
| --- | --- |
| `src/mcts/` | shared |
| `src/modeling/` | shared |
| `src/math_kernel/` | pde |
| `src/pde/` | pde |
| `src/refinement/` | pde |
| `src/alphagalerkin/` | pde |
| `src/training/` | shared |
| `src/games/` | game-ai |
| `src/engines/` | game-ai |
| `src/tournament/` | game-ai |
| `src/analysis/` | game-ai |
| `src/curriculum/` | game-ai |
| `src/physics/` | pde |
| `src/research/` | pde |
| `src/data/` | shared |
| `src/backend/` | shared |
| `src/distributed/` | shared |
| `src/deployment/` | shared |
| `src/agents/` | shared |
| `src/integrations/` | shared |
| `src/poc/` | shared |
| `src/templates/` | shared |
| `src/tools/` | shared |
| `src/experiments/` | pde |
| `src/demos/` | shared |
| `src/prototyping/` | shared |
| `src/core/` | shared |
| `src/video_compression/` | video |
<!-- charter:scope:end -->

#### Scenario: A new package is added without charter update
- GIVEN a contributor adds `src/newthing/__init__.py`
- WHEN CI runs `tests/docs/test_charter_alignment.py`
- THEN the scope guard SHALL fail naming `newthing`
- AND it SHALL stay failing until both `ARCHITECTURE.md` and this register list it

#### Scenario: The charter names a package that does not exist
- GIVEN the scope register contains a row for a package with no `src/<name>/__init__.py`
- WHEN the scope guard runs
- THEN it SHALL fail naming that package

### Requirement: Non-Goal Exclusion

Subsystems removed in the 2026-07-22 cut-to-the-core SHALL NOT exist as `src/` packages. The
project is deliberately narrowed to the Galerkin/MCTS core; re-adding these is a scope decision
that requires amending this charter first.

<!-- charter:non-goals:start -->
| Removed package | Removed | Why it is a non-goal |
| --- | --- | --- |
| `src/reentry/` | 2026-07-22 | Domain PoC; not on the core solver path |
| `src/vertex/` | 2026-07-22 | Cloud-training launcher; infrastructure, not thesis |
| `src/intercept/` | 2026-07-22 | Domain PoC; not on the core solver path |
| `src/firefighting/` | 2026-07-22 | Domain PoC; not on the core solver path |
| `src/thermo/` | 2026-07-22 | λ-window scheduling ablation; negative result preserved in `CHANGELOG.md` and `results/lambda_scheduling.{csv,png}` |
<!-- charter:non-goals:end -->

Provenance note: the cut is recorded in `CHANGELOG.md` and the 2026-07-22 `CLAUDE.md` milestone.
It is **not** verifiable from git tags — CI checks out shallow with no tags, so no guard may
depend on git history.

#### Scenario: A cut module reappears
- GIVEN someone recreates `src/thermo/`
- WHEN the non-goal guard runs
- THEN it SHALL fail naming `thermo`

### Requirement: Evidence-Backed Claims

Every numeric headline claim the project makes SHALL cite a committed artifact that exists in the
repository, and the cited number SHALL be the one that artifact contains.

A claim whose artifact is missing, or whose number comes from an uncommitted spike run, is
inadmissible regardless of how plausible it is. This Requirement is machine-checkable only for
*artifact existence*; that the number matches the artifact's contents remains a human review
duty — but existence alone would have caught three of the four P0 defects this charter was
written in response to.

<!-- charter:evidence:start -->
| Claim | Value | Artifact |
| --- | --- | --- |
| Zero-shot transfer MSE, 19×19 from 9×9 training | ≈2.3e-3 (3-seed median) | `results/transfer_baseline_compare.csv` |
| Retrained-CNN baseline MSE, 19×19 | ≈1.6e-4 | `config/baselines/transfer_ci.json` |
| Operator-vs-retrained-CNN ratio | ≈14× (operator loses) | `specs/transfer_baseline_compare.spec.md` |
| L-shape AMR, MCTS vs Dörfler at matched DOF | median ratio 1.0996 (MCTS **loses** ~10%), wins 1/5 seeds | `results/lshape_mcts_vs_dorfler.csv` |
| L-shape AMR at matched compute | median ratio 2.04, MCTS wins 0/5 seeds | `specs/lshape_amr_compare.spec.md` |
| L-shape substrate, uniform-refinement L2 rate | O(h^1.31) ≈ O(N^-0.65) | `tests/research/test_lshape_convergence_gate.py` |
| L-shape adaptive Dörfler vs uniform at matched DOF | Dörfler 5–9× **worse** (tensor-product refinement defect) | `results/lshape_mcts_vs_dorfler.csv` |
| Stochastic Galerkin density MSE | 2.3e-8 | `results/stochastic_galerkin_compare.csv` |
| Test-suite size | 7,000+ test functions | `tests/` |
| Global coverage gate | 85% branch | `pyproject.toml` |
<!-- charter:evidence:end -->

#### Scenario: A claim cites a nonexistent artifact
- GIVEN a claim row citing `benchmarks/results/headline_2026_04/pareto_plot.png`
- WHEN the evidence guard runs
- AND that path does not exist
- THEN the guard SHALL fail naming the path

#### Scenario: A spike number is promoted to a headline
- GIVEN an uncommitted exploratory run produces a more favourable number
- WHEN it is quoted as the project's headline
- THEN this is a charter violation
- AND the claim SHALL be either backed by a committed artifact or labelled a spike

### Requirement: UI Claim Fidelity

Every numeric figure an interactive surface renders SHALL be traceable to a committed artifact,
and no interactive surface SHALL state a retracted claim as live.

*Evidence-Backed Claims* governs what the project's documents may assert. It does not reach the
Gradio dashboard, which renders figures from Pydantic defaults and hardcoded markdown — and which
reaches more people than any document. A number shown to a user is a claim regardless of whether
it appears in prose.

The interactive surfaces this Requirement governs are `dashboard/` and `hf_space/` — the two
roots the guard scans. Adding a third interactive surface SHALL extend that scan.

This Requirement also bans the self-comparison framing the transfer benchmark retracts: a ratio
against an arbitrary pass threshold is not a result. Where a committed baseline exists, the
comparison SHALL be against that baseline, reported in whichever direction the artifacts support.
`dashboard/config.py` holds the dashboard's figures; `config/baselines/transfer_ci.json` is the
committed source they are checked against.

#### Scenario: A spike figure is rendered by the dashboard
- GIVEN `dashboard/config.py` declares a transfer MSE that no committed artifact contains
- WHEN the UI claim guard runs
- THEN it SHALL fail naming the metric and the committed value it disagrees with

#### Scenario: A retracted figure reappears in a UI surface
- GIVEN a file under `dashboard/` contains the fabricated transfer figure or the retracted
  blanket novelty claim
- WHEN the UI claim guard runs
- THEN it SHALL fail naming the file

### Requirement: Novelty Claim Discipline

The project's novelty SHALL be stated only in its narrow, defensible form: **MCTS multi-step
look-ahead for error-driven adaptive refinement and Galerkin basis selection**. The blanket claim
that no published work combines MCTS with Galerkin/finite-element methods is **retracted** —
TreeMesh ([arXiv:2111.07613](https://arxiv.org/abs/2111.07613)) couples MCTS+RL with finite-element
mesh *generation*, a distinct problem.

Novelty is a *method* delta, not a demonstrated win. The honest `lshape_amr_compare` result — MCTS
**losing** at matched DOF (ratio 1.0996, wins 1/5 seeds) and losing further at matched compute
(ratio 2.04, 0/5 seeds) — SHALL be reported alongside any favourable framing.

The previously reported "~4% matched-DOF win" (ratio 0.9605) is **retracted (2026-08-16)**: it was
produced by a boundary-condition defect in `lshape_inside_predicate`, which removed the *open*
rather than the *closed* fourth quadrant and so never imposed the `u=0` Dirichlet condition on the
L-shape's two reentrant edges. The substrate diverged under uniform refinement (L2 error rising
from 5.0e-2 at 65 DOF to 1.15e-1 at 12545 DOF), so both arms were compared on a problem neither was
solving. Guarded going forward by `tests/research/test_lshape_convergence_gate.py`.

A second, now-unmasked defect SHALL also be reported: the shared discretisation refines by
tensor-product grid *lines*, so adaptive Dörfler marking is **5–9× worse than plain uniform
refinement at matched DOF**, with the gap widening as DOF grows. Until refinement is element-local,
no marking-policy comparison on this substrate measures refinement quality.

`docs/related-work.md` owns the per-entry novelty-boundary register and is guarded by
`tests/regression/test_related_work_guard.py`. `docs/business/proposals/PRIOR_ART_REVIEW.md` owns
the prior-art analysis.

#### Scenario: A retracted claim resurfaces
- GIVEN either retracted claim recorded in `tests/support/cut_modules.py` is stated in the
  charter as a live claim, rather than described as retracted
- WHEN the retraction guard runs
- THEN it SHALL fail naming the offending line

### Requirement: Capability Register Accuracy

The charter's capability register SHALL equal the PoC scenarios registered at runtime.

The registry — not this table, and not any grep — is the source of truth. Scenario decorators take
module constants, so a string-literal grep finds only 4 of the 10 scenarios. The guard SHALL
enumerate `ScenarioRegistry().list_scenarios()` **in a subprocess**: the registry is a
process-wide singleton that `tests/poc/*` autouse fixtures `clear()` without teardown, which makes
an in-process read order-dependent (measured: 10 scenarios under `pytest tests/poc tests/docs`,
but 0 under a narrower `tests/poc` selection).

<!-- charter:capabilities:start -->
| Scenario | What it demonstrates |
| --- | --- |
| `complexity` | O(N) attention scaling benchmark |
| `llm_prior_ablation` | LLM-prior MCTS basis selection vs random/trained |
| `lshape_amr_compare` | MCTS vs Dörfler on L-shaped Poisson AMR |
| `noyron_basis` | MCTS basis selection on the Leap 71 helical operator |
| `noyron_hx` | Zero-shot 3D heat-transfer transfer on a helical SDF |
| `scaling_law` | Residual vs MCTS-simulation-budget scaling fit |
| `stability` | LBB / inf-sup stability monitoring |
| `stochastic_galerkin_compare` | NKE stochastic Galerkin vs deterministic arm |
| `transfer` | Zero-shot resolution transfer |
| `transfer_baseline_compare` | Operator vs a retrained discrete CNN |
<!-- charter:capabilities:end -->

#### Scenario: A scenario is registered but undocumented
- GIVEN a new `@scenario("newscenario")` is registered
- WHEN the capability guard enumerates the registry in a subprocess
- THEN it SHALL fail naming `newscenario` as missing from the register

### Requirement: Quality Gate Fidelity

Coverage gates documented by the project SHALL be the gates CI actually enforces.

The charter records the gates; `.github/workflows/ci.yml` enforces them. The guard checks
charter ⊆ CI (a gate the charter claims must exist in CI at that value). The reverse direction is
deliberately unchecked — adding a CI gate should not nag a charter edit.

<!-- charter:gates:start -->
| Target | Gate |
| --- | --- |
| `src/mcts` | 90 |
| `src/refinement` | 85 |
| `src/modeling` | 85 |
| `src/training` | 85 |
| `src/research` | 85 |
| `src/alphagalerkin` | 85 |
| `src/games` | 80 |
| `src/pde` | 75 |
| `src/physics` | 75 |
| `src/distributed` | 60 |
| `dashboard` | 84 |
| `src/agents` | 85 |
| `src/tools` | 89 |
| `src/experiments` | 87 |
| `src/curriculum` | 87 |
| `src/engines` | 82 |
| `src/data` | 77 |
<!-- charter:gates:end -->

#### Scenario: A documented gate is not enforced
- GIVEN the charter records `src/mcts` at 90
- WHEN CI's `src/mcts` step gates at a different value
- THEN the gate guard SHALL fail

### Requirement: Accepted Deviation Disclosure

Every known, deliberate divergence between documentation and code reality SHALL be recorded here
with a stated reason. An undisclosed deviation is indistinguishable from drift.

<!-- charter:deviations:start -->
| Deviation | Reason |
| --- | --- |
| `hf_space/` mirrors `src/` but is not kept in sync | A manual, partial, independently-formatted deploy copy; asserting parity would force a single-sourcing refactor that is out of scope. `tests/hf_space/test_mirror_guard.py` guards a floor (importability + scrubbing) instead. |
| `CLAUDE.md`'s directory tree lags `ARCHITECTURE.md` | Self-labelled a convenience excerpt; `ARCHITECTURE.md` wins and is drift-guarded by `tests/docs/test_architecture_map.py`. |
| CI runs `mypy --strict` with `continue-on-error: true` | Diagnostics depend on the installed torch version, so a hard gate is flaky across environments; `ruff check` and `ruff format` are the hard gates and mypy is a local pre-merge discipline. |
| `claude-code-platform/` is a repo within the repo | Its own `pyproject.toml`, tests, and CLAUDE.md; excluded from the root package build, the docs site, and the link checker. |
| `CHANGELOG.md` still references removed modules | Immutable append-only history; excluded from the link checker by design. |
| `results/lambda_scheduling.{csv,png}` outlive their producer | The `thermo` module was cut, but these are the only in-tree evidence of that negative result, and `ARCHITECTURE.md` declares changelog-referenced artifacts deliberate. |
| `RefinementGameRegistry` has zero runtime registrants | Real PDE games register in `GameRegistry`; the refinement registry is a forward-looking abstraction. Tracked as a dead-abstraction follow-up, not a silent gap. |
<!-- charter:deviations:end -->

#### Scenario: A deviation is recorded without a reason
- GIVEN a deviation row whose Reason cell is empty or `TBD`
- WHEN the deviation guard runs
- THEN it SHALL fail naming that deviation
