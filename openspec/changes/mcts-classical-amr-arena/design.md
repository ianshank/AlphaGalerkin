# Design: `mcts-classical-amr-arena`

## Technical Approach

Separate **contract** from **execution**. The first mergeable unit is a pre-registration document
plus machine-checkable hooks (config schema, manifest schema, abort-on-failed-adequacy). Only
then run arms. This is the charter's Evidence-Backed Claims requirement applied prospectively
rather than after a spike.

## Architecture Decisions

### AD1 — Pre-registration is a first-class artifact

**Decision.** `specs/mcts_classical_amr_arena.spec.md` (repo spec format) +
`config/benchmarks/` (or `config/research/`) YAML hold the locked parameters. Tasks.md Phase 0
checkboxes mirror the spec's MetricThreshold / AQA blocks.

**Rationale.** SQE peer review: without this, Slice E plumbing invites opportunistic runs — the
failure mode the charter was written for.

### AD2 — Shared substrate instance for all arms

**Decision.** One `SubstrateConfig` / mesh lineage per seed; arms differ only in *how they choose
what to refine*. Manifest records `kind`, θ, DOF window, and `dof_convention`.

**Rationale.** Substrate adequacy is only meaningful if the comparison does not silently change
discretisation mid-flight.

### AD3 — Adequacy is a precondition, not a result

**Decision.** Before policy comparison, run the existing gate predicate on the exact config; abort
and record if it fails. Do not re-tune gate thresholds to admit a broken config.

**Rationale.** Gate constants are pinned in `specs/refinement_substrate.spec.md`.

### AD4 — MCTS arm consumes `RefinementGame`, not `LShapeAMRGame`

**Decision.** Phase 2 imports the registrant from `refinement-game-registrant`. Legacy
`LShapeAMRGame` remains golden-only.

**Rationale.** Impure apply_action and tensor-grid semantics would re-poison the comparison.

### AD5 — Parallelism boundary

| Work | Parallel with registrant? |
| --- | --- |
| Phase 0 pre-reg docs/schemas | **Yes** |
| Phase 1 classical sweeps | **Yes** (substrate-only) |
| Phase 2 MCTS arm | **No — wait for registrant 7.x** |
| Trained evaluator | Out of scope |

### AD6 — Headline discipline

**Decision.** Every published rate includes θ and DOF window. Charter/README updates require a
committed artifact path. Human review still required for number match (existence guard is not
enough).

**Rationale.** Prior θ=0.3 vs θ=0.5 rate drift (`-1.2515` vs `-1.3109`) is a documented footgun.

## Non-goals (design-level)

- Vectorising the ZZ estimator (separate change).
- Multi-field / Navier–Stokes pilot.
- Dashboard or HF Space claim updates beyond pointing at new artifacts (frozen track).
