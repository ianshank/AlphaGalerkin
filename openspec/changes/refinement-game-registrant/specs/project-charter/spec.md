# Delta: project-charter — `refinement-game-registrant`

## MODIFIED Requirements

### Requirement: Accepted Deviations

**Retire** the deviation that `RefinementGameRegistry` has zero runtime registrants, once a
production `@register_refinement_game` exists and is imported via an explicit `register_*`
module.

**Retire** (or narrow) the deviation that `RefinementSubstrateRegistry` has zero runtime
*lookups*, once a non-test path resolves a substrate by key.

**Retire** the staged `fingerprint` / `_STAGED_FOR_UPCOMING_TASK` exemption once a solve-cache
consumer reads fingerprints in production code.

**Add** a time-boxed deviation: two-path period where legacy `LShapeAMRGame` /
`lshape_amr_compare` remain as golden back-compat alongside the substrate-backed
`RefinementGame`. Retirement condition: the golden test is the sole remaining consumer of the
legacy harness (arena and production paths use the registrant only).

#### Scenario: Empty refinement game registry after this change
- GIVEN this change's tasks are complete
- WHEN charter deviation guards / docs are read
- THEN they SHALL NOT claim zero `RefinementGameRegistry` registrants as an accepted standing deviation
- AND they MAY cite the time-boxed two-path deviation until its retirement condition holds

#### Scenario: Supersession of defective-substrate AMR compare
- GIVEN `specs/lshape_amr_compare.spec.md` previously status `Implemented`
- WHEN this change's governance tasks complete
- THEN that spec SHALL be marked superseded with a pointer to the substrate + arena successors
- AND legacy MCTS-lose artifacts SHALL be labeled non-informative for element-local policy comparisons
