# Delta: `project-charter` — look-ahead vs greedy (Gate 1)

This change modifies four Requirements. **Capability Register Accuracy** gains the scenario's
row, landed with the code (task 2.7). After the runs (task 4, through `claims-ledger`):
**Evidence-Backed Claims** gains two register rows and the arena row's pointer to Gate 1;
**Novelty Claim Discipline** states the result; and **Accepted Deviation Disclosure** amends the
frozen-tracks row. The other four Requirements are untouched: no package is added or removed
(Scope Integrity), no cut module returns (Non-Goal Exclusion), no interactive surface renders a
Gate 1 figure (UI Claim Fidelity — `dashboard/` and `hf_space/` carry none), and no coverage
gate moves (Quality Gate Fidelity).

## MODIFIED Requirements

### Requirement: Capability Register Accuracy

The charter's capability register SHALL equal the PoC scenarios registered at runtime. This change
registers one scenario and adds its row:

| Scenario | What it demonstrates |
| --- | --- |
| `lookahead_vs_greedy` | Gate 1 go/no-go: MCTS look-ahead vs the best classical marking policy (greedy, Dörfler θ ∈ {0.1, 0.3, 0.5}, uniform) on the L- and Z-shape testbeds |

The row is mechanical — the existing guard
(`tests/docs/test_charter_alignment.py::test_capability_register_matches_scenario_registry`)
enumerates the registry in a subprocess and fails in both directions — so it lands in the same
commit that registers the scenario, never before (an unregistered row fails as "extra") and never
after (a registered scenario without a row fails as "missing").

#### Scenario: The Gate 1 scenario is registered without its register row
- GIVEN `@scenario("lookahead_vs_greedy")` is imported by `src.poc.scenarios`
- WHEN the capability guard enumerates the registry in a subprocess
- AND the capability region has no `lookahead_vs_greedy` row
- THEN the guard SHALL fail naming `lookahead_vs_greedy` as missing from the register

#### Scenario: The register row describes a scenario, not a result
- GIVEN the `lookahead_vs_greedy` row
- WHEN a reader takes it as evidence
- THEN it SHALL state what the scenario measures and no numeric outcome; the GO/NO-GO result is
  evidence and belongs in the evidence register, citing its `.run.json`
- AND this is a review obligation, not a mechanical check: the capability guard compares names only

### Requirement: Evidence-Backed Claims

The Requirement text is unchanged. Two register rows are added and one is amended. Every figure
below is read from the cited proposal-grade sidecars (`results/lookahead_vs_greedy_lshape.run.json`,
recorded at `d47bf23`, config hash `c8bf9aaff8cc86f8`; `results/lookahead_vs_greedy_zshape.run.json`,
recorded at `77af462`, config hash `d216b4e9ca5df579`); both hashes recompute from their recorded
configs (`tests/docs/test_sidecar_config_hash.py`).

**Added** — live rows (applied when task 4.1 lands):

| Claim | Value | Artifact |
| --- | --- | --- |
| Gate 1 (pre-registered, `specs/lookahead_vs_greedy.spec.md`) T1 L-shape: MCTS look-ahead vs the best classical marking policy (`skfem_tri`; greedy, Dörfler θ ∈ {0.1, 0.3, 0.5}, uniform; 30 single-element steps, 64 simulations, top-4 ranked legal set, untrained evaluator) | **NO-GO**, 0/5 criteria: MCTS/best classical 1.0216 at matched DOF 361 (best: Dörfler θ=0.3). The deterministic MCTS arm made greedy's identical 30 decisions, so **search contributed no decisions**, at a median realized tree depth of 10 (max 11). Root-noise arm median 1.0539, 0/5 seeds below 1; DOF saving vs greedy 0 (K* infinite). An elliptic control: it does not close the thesis, which waits on the deferred moving-front testbed (T3). | `results/lookahead_vs_greedy_lshape.{csv,run.json}` |
| Gate 1 T2 Z-tetromino (two 270° corners, strengths c1 = 1.0 and c2 = 0.25): MCTS look-ahead vs the best classical marking policy, with T1's arms (greedy, Dörfler θ ∈ {0.1, 0.3, 0.5}, uniform) and budgets | **NO-GO**, 0/5 criteria: MCTS/best classical 1.0296 at matched DOF 398 (best: Dörfler θ=0.5). The deterministic MCTS arm again made greedy's identical 30 decisions, so **search contributed no decisions**, at a median realized tree depth of 10 (max 10). Root-noise arm median 1.0616, 0/5 seeds below 1; DOF saving vs greedy 0 (K* infinite). Does not close the thesis (T3 deferred). | `results/lookahead_vs_greedy_zshape.{csv,run.json}` |

**Amended.** The arena row's "look-ahead is untested by this artifact (Gate 1 pending)" now reads
"(Gate 1, the next two rows, tests it)". Nothing else in that row changes.

#### Scenario: A Gate 1 claim omits that the search made greedy's decisions
- GIVEN an MCTS-vs-Dörfler policy-ratio claim in README or the evidence register
- AND it cites a sidecar whose `harness` is `scripts.run_lookahead_vs_greedy`
- AND that sidecar records `primary_decisions_diverging_from_greedy == 0`
- WHEN `tests/docs/test_lookahead_attribution.py` runs
- THEN the claim SHALL carry the label "search contributed no decisions"
- AND the root-noise arm's divergence (`robust_decisions_diverging_from_greedy_*`) SHALL neither
  require nor excuse the label: its departures from greedy are noise in the decision rule, not
  search

#### Scenario: A new harness records divergence without a schema
- GIVEN a committed `.run.json` that records a metric containing
  `decisions_diverging_from_greedy`
- AND its `harness` is not declared in the guard's `DIVERGENCE_SCHEMAS`
- WHEN the guard runs
- THEN `test_every_harness_recording_divergence_is_declared` SHALL fail naming the sidecar

### Requirement: Novelty Claim Discipline

The arena paragraph's closing changes; the rest of the Requirement is unchanged.

**Amended.** "Look-ahead is untested by this artifact; whether it beats greedy anywhere is Gate 1
of `docs/business/COMMERCIALIZATION_PEER_REVIEW.md`" becomes the result:

> Look-ahead is untested by this artifact. Gate 1 of
> `docs/business/COMMERCIALIZATION_PEER_REVIEW.md`, pre-registered in
> `specs/lookahead_vs_greedy.spec.md`, tested it on two elliptic testbeds and returned **NO-GO** on
> both (0/5 criteria each): with 64 simulations over a top-4 ranked legal set, the deterministic
> search made greedy marking's 30 decisions on the L-shape and on the Z-tetromino alike (**search
> contributed no decisions**), and the best classical arm beat it at matched DOF (MCTS/best
> classical 1.0216 against Dörfler θ=0.3, and 1.0296 against Dörfler θ=0.5;
> `results/lookahead_vs_greedy_lshape.{csv,run.json}`,
> `results/lookahead_vs_greedy_zshape.{csv,run.json}`). That is the control result the
> pre-registration expected where adaptive-FEM theory predicts greedy marking is near-optimal. It
> does **not** close the thesis, which waits on the deferred moving-front testbed (T3), and no
> text SHALL present the method delta as demonstrated. At matched solves the arena's MCTS is at
> 9.23 (ungated).

#### Scenario: The T1/T2 NO-GO is read as closing the thesis
- GIVEN a statement that the look-ahead thesis is refuted or closed
- WHEN it rests only on the T1/T2 results
- THEN it SHALL be corrected: the pre-registration names T3 as decisive and a NO-GO on T1/T2 the
  expected control result
- AND this is a **review** obligation; no phrase ban is added

### Requirement: Accepted Deviation Disclosure

One phrase of one row changes. **Amended.** The frozen-tracks row's "the thesis question is open
until Gate 1" becomes "the thesis question stays open: Gate 1 returned NO-GO on both of its
elliptic testbeds, where the search again made greedy's decisions, and the decisive moving-front
testbed (T3) is deferred". Retirement condition unchanged.

> **Note on guard coverage.** The two new Evidence-Backed Claims scenarios are mechanical
> (`tests/docs/test_lookahead_attribution.py`, whose docstring records the defects each test
> killed, 12/12 for this extension). The Novelty paragraph, the deviation row, `docs/FOCUS.md`,
> the arena spec and the peer review's status table are a **review** obligation, as they were
> for `arena-lookahead-attribution`.
