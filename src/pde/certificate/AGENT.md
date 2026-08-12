# AGENT.md — `src/pde/certificate/` (foundation)

## Persona
**Name:** Certificate Auditor
**Expertise:** A posteriori error estimation, verified numerics, forward-compat schemas.
**Mindset:** Every number ships with its own justification. An unverifiable headline is a bug — the F0 backup defect and the fabricated 0.000209 transfer-MSE headline are the failure class this subpackage exists to prevent.

## Scope (this PR — foundation only)
- **Certificate artifact** (`certificate.py`): Pydantic, schema-versioned, forward-compat migration, frozen, `extra="ignore"`. Enforces AC5 stability-consistency at validator level.
- **Stability-constant registry** (`stability.py`): thread-safe singleton keyed on `src.pde.config.PDEType`. Every enum value has a declared source; Helmholtz / Biharmonic / Navier-Stokes ship `unbounded_with_warning` — honesty > premature rigor.
- **Config** (`config.py`): reuses canonical `src.poc.config.MetricThreshold`; no parallel schema.
- **Logging** (`logging.py`): `structlog` binder with a closed set of `certificate.*` event names; stable 16-char hex `certificate_id`.

## Out of scope (follow-on PRs)
- Track A residual a posteriori estimator (PR #2 — reuses `DorflerAMRSolver._compute_indicators_2d` after rigorous-estimator audit).
- Track B certified residual (PR #3 — heuristic dense-grid tier for CI; rigorous `autoLiRPA`/`∂-CROWN` behind `[certificate-rigorous]` optional extra).
- Scenario wiring (PR #4 — `lshape_amr_compare`, `transfer_baseline_compare` emit certificates).

## Invariants (regression-guarded)
1. **Batch artifact, never hot path** — no code in `src/pde/certificate/` may be imported from `src/mcts/**` and vice versa (AST guard: `tests/pde/certificate/test_import_isolation.py`).
2. **No parallel schema** — thresholds are `list[src.poc.config.MetricThreshold]`.
3. **AC5 honesty** — every `PDEType` has a registered `StabilityEntry`. `stability_constant is None` iff `stability_source == 'unbounded_with_warning'` (validator-enforced on the artifact).
4. **Documented event names** — only names in `CERTIFICATE_LOG_EVENTS` may appear as the first arg of `.info/.debug/.warning/.error/.critical(...)` in this subpackage.
5. **Backwards compatible** — additive-only; never change an existing `PDEType` value; forward-compat `extra="ignore"` on the artifact; schema-versioned migration.

## Design patterns reused (see `docs/architecture/c4_mermaid.md`)
- Double-check locking singleton — mirrors `src.templates.registry.BaseRegistry`.
- Schema-versioned migration — mirrors `src.poc.baselines.schema.ScenarioBaselineDocument`.
- Frozen Pydantic artifact — mirrors `src.experiments` result objects.
- `structlog` binder w/ stable identity — mirrors `src.integrations.lm_studio.client`.

## Validation gates (`.claude/skills/certificate-validation/SKILL.md`)
This PR ships Gates **1** (static), **2** (schema round-trip), **6** (stability honesty), partial **5** (import-isolation only — hot-path timing gates land with the estimators), and **7** (coverage). Gates **3**, **4**, **8** land with Track A / B / scenario PRs.
