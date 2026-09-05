# Delta stub: `mcts_classical_amr_arena`

The authoritative per-feature contract will live at
`specs/mcts_classical_amr_arena.spec.md` (repo spec format with MetricThreshold / AQA), created
in Phase 0 task 0.1.

This openspec delta records project-level intent only:

## ADDED Requirement (project-level)

### Requirement: Arena claims require pre-registration

No numeric MCTS-vs-classical AMR headline MAY be added to README, charter evidence, or SBIR
facing docs unless:

1. `specs/mcts_classical_amr_arena.spec.md` exists with locked falsifiers and configs, and
2. A committed `results/*` artifact + `*.run.json` manifest match the claim, and
3. Rates are quoted with θ and DOF window.

#### Scenario: Headline without manifest
- GIVEN a PR adds an AMR ratio to README or charter
- WHEN Phase 0 guards run
- THEN they SHALL fail unless the claim cites an existing artifact path from the pre-reg contract
