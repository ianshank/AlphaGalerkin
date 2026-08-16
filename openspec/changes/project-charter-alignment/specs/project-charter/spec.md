# Delta for project-charter

This change introduces the project charter. There is no prior `project-charter` spec, so every
requirement below is ADDED; nothing is MODIFIED or REMOVED.

## ADDED Requirements

### Requirement: Scope Integrity

The charter's scope register SHALL enumerate exactly the `src/` packages documented in
`ARCHITECTURE.md`'s package map, and every package it names SHALL exist on disk.

#### Scenario: A new package is added without charter update
- GIVEN a contributor adds `src/newthing/__init__.py`
- WHEN CI runs `tests/docs/test_charter_alignment.py`
- THEN the scope guard SHALL fail naming `newthing`
- AND it SHALL stay failing until both `ARCHITECTURE.md` and the charter register list it

#### Scenario: The charter names a package that does not exist
- GIVEN the scope register contains a row for a package with no `src/<name>/__init__.py`
- WHEN the scope guard runs
- THEN it SHALL fail naming that package

### Requirement: Non-Goal Exclusion

Subsystems removed in the 2026-07-22 cut-to-the-core SHALL NOT exist as `src/` packages.

#### Scenario: A cut module reappears
- GIVEN someone recreates `src/thermo/`
- WHEN the non-goal guard runs
- THEN it SHALL fail naming `thermo`

#### Scenario: Provenance does not depend on git history
- GIVEN CI checks out shallow with no tags
- WHEN the non-goal guard runs
- THEN it SHALL NOT consult `git tag` or commit history
- AND it SHALL rely only on in-tree evidence

### Requirement: Evidence-Backed Claims

Every numeric headline claim the project makes SHALL cite a committed artifact that exists in the
repository, and the cited number SHALL be the one that artifact contains.

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

### Requirement: Novelty Claim Discipline

The project's novelty SHALL be stated only as MCTS multi-step look-ahead for error-driven adaptive
refinement and Galerkin basis selection. The blanket "no MCTS + Galerkin/FEM" claim SHALL NOT be
used, and any favourable framing SHALL be accompanied by the honest matched-compute result.

#### Scenario: A retracted claim resurfaces
- GIVEN either retracted claim recorded in `tests/support/cut_modules.py` is stated in the
  charter as a live claim, rather than described as retracted
- WHEN the retraction guard runs
- THEN it SHALL fail naming the offending line

### Requirement: Capability Register Accuracy

The charter's capability register SHALL equal the PoC scenarios registered at runtime, enumerated
in a subprocess rather than by grep or in-process import.

#### Scenario: A scenario is registered but undocumented
- GIVEN a new `@scenario("newscenario")` is registered
- WHEN the capability guard enumerates the registry in a subprocess
- THEN it SHALL fail naming `newscenario` as missing from the register

#### Scenario: Registry reads survive singleton pollution
- GIVEN `tests/poc/*` autouse fixtures call `ScenarioRegistry().clear()` without teardown
- WHEN the capability guard runs after them in the same session
- THEN it SHALL still read the true registered set

### Requirement: Quality Gate Fidelity

Coverage gates documented by the project SHALL be the gates CI actually enforces, checked in the
charter ⊆ CI direction only.

#### Scenario: A documented gate is not enforced
- GIVEN the charter records `src/mcts` at 90
- WHEN CI's `src/mcts` step gates at a different value
- THEN the gate guard SHALL fail

### Requirement: Accepted Deviation Disclosure

Every known, deliberate divergence between documentation and code reality SHALL be recorded in the
charter with a stated reason.

#### Scenario: A deviation is recorded without a reason
- GIVEN a deviation row whose Reason cell is empty or `TBD`
- WHEN the deviation guard runs
- THEN it SHALL fail naming that deviation
