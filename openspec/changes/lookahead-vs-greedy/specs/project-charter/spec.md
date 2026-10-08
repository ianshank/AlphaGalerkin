# Delta: `project-charter` — look-ahead vs greedy (Gate 1)

This change modifies the **Capability Register Accuracy** Requirement only. The other seven
Requirements — Scope Integrity, Non-Goal Exclusion, Evidence-Backed Claims, UI Claim Fidelity,
Novelty Claim Discipline, Quality Gate Fidelity and Accepted Deviation Disclosure — are untouched.
In particular this change writes **no** evidence, novelty or deviation row; the Gate 1 result's
register wording is proposed separately and lands through `claims-ledger`.

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
