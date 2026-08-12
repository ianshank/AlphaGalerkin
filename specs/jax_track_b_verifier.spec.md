# Spec: jax_track_b_verifier — JAX Track B verifier + backend-neutral contract

> **Status:** Draft (v1)
> **Owner:** AlphaGalerkin core (certificate stream)
> **Primary module(s):** `src/pde/certificate/{interface,registry,types,verifiers/}` (new files, additive to PR #117 foundation); optional `src/backend/jax_batch/` (WS3 only)
> **Config class:** `src.pde.certificate.config.CertificateConfig` (extended, backwards-compat) — thresholds reuse the canonical `src.poc.config.MetricThreshold` (no parallel schema, per the spec-tree peer-review correction inherited from `verified_error_certificate.spec.md`)
> **Tracking:** CLAUDE.md Next-Steps row "Verified error certificates (two-track)"; sibling spec `specs/verified_error_certificate.spec.md`
> **Origin:** Follow-up to PR #117 addressing the addendum "JAX for AlphaGalerkin — narrow scope, high leverage". Reconciles the addendum with PR #1 of the certificate series (which already shipped `CertificateConfig`, `Certificate`, and the AST hot-path guard).

---

## 1. Motivation

The verified_error_certificate spec (§2 Track B) commits AlphaGalerkin to
certified uniform residual bounds for neural-operator outputs via
dReal / autoLiRPA / ∂-CROWN (arXiv:2603.19165). Two backend choices exist:

1. **PyTorch-only Track B** (autoLiRPA is torch-native, mature but slow on the
   published benchmarks: 2,705 s per 2D Poisson certificate, Table 2 of the
   reference paper).
2. **JAX Track B** via `jax_verify` (google-deepmind, CROWN/IBP family in JAX)
   with `jax.jit`/`vmap` for cold-start amortisation across a batch of
   certificates.

Both are legitimate; neither is universally better. The **verifier boundary
is the correct place** for JAX because it is a pure functional pipeline
(no mutation, no tree traversal, well-typed input/output). Migrating MCTS
selection/expansion/backup to JAX would be a rewrite for negligible gain
and would break the `BatchMCTS` semantic invariants guarded by
`tests/mcts/test_backup_modes.py`.

## 2. Design principle

**Backend selection is explicit and per-run, never implicit.** The
`ResidualVerifier` protocol is backend-neutral; the concrete implementation
enforces its own backend requirement. There is no automatic conversion
between Torch and JAX models inside a certification run — a Torch model
means a Torch verifier, a JAX model means a JAX verifier, and mismatches
raise at dispatch time with an actionable error.

## 3. Scope

**In scope**
- Extended `CertificateConfig`: `verifier_backend` (Literal), `budget`,
  `dtype`, `device`, `record_compile_time`. All new fields have defaults
  chosen so **every existing `CertificateConfig()` construction from PR #1
  produces a byte-identical config** (backwards-compat AC).
- `ResidualVerifier` `runtime_checkable` Protocol in
  `src/pde/certificate/interface.py`, mirroring `src.backend.interface.BackendInterface`.
- `VerifierRegistry` reusing `src.templates.registry.create_typed_registry`.
  All verifier dispatch goes through the registry — no `if/elif` chains.
- Typed artifacts in `src/pde/certificate/types.py`:
  `CertifiedResidualBound`, `CertifiedModel`, `DomainSpec`,
  `CertificationBudget`, `HardwareMeta`, `DomainCoverage`, `VerifierBackend`.
- `HeuristicGridResidualVerifier` (WS1) — reference implementation.
  Sources the stability constant from PR #1's `StabilityConstantRegistry`;
  emits `rigor="heuristic"`, `domain_coverage="grid_sampled"`.
- Stub sentinel registrations for `autolirpa` / `delta_crown` / `jax_verify`
  / `dreal` that raise `VerifierUnavailableError` — so error-path tests
  exercise the real dispatch mechanism before WS2 ships the real backends.
- `TorchResidualVerifier` (WS2) — autoLiRPA / ∂-CROWN adapter behind the
  new `[certificate-rigorous]` optional extra.
- `JaxVerifyResidualVerifier` (WS2) — `jax_verify` adapter behind
  `[certificate-rigorous]` + existing `[jax]` extra.
- **`src/backend/jax_batch/`** (WS3) — optional `jax.jit`/`vmap` wrapper
  for `BatchMCTS`'s pure-function evaluator boundary. Placed under
  `src/backend/` (next to `jax_backend.py`), **not** under
  `src/pde/certificate/`, because the certificate subpackage carries a
  two-way AST import guard against `src/mcts/` (PR #1 Gate 5) and the
  batched-evaluator lives on the MCTS side of that boundary.

**Out of scope**
- Migration of MCTS tree traversal / node expansion / backup logic to
  JAX. This is a hard stop — `test_backup_modes.py` semantics are load-
  bearing and MCTS's Python control flow does not benefit from JAX tracing.
- Automatic framework conversion between Torch and JAX models inside a
  single certification run.
- Heuristic-tier certificates presented as rigorous — the artifact
  validator makes this structurally impossible (see AC3).
- Performance claims without measured cold-start (`compile_wall_s`) and
  steady-state (`steady_state_wall_s`) numbers on a pinned fixture.

## 4. Acceptance Criteria

### AC1 (J1) — Availability and fail-closed
- **Given** `CertificateConfig(verifier_backend="jax_verify")` and the
  `[jax]` extra is not installed, **when** `get_verifier(...)` is called,
  **then** it raises `VerifierUnavailableError` with an install-hint message
  including the missing extra name.
- **Given** `verifier_backend="jax_verify"` and `jax_verify` is unimportable
  even though `jax` is installed, **when** `certify(...)` runs, **then**
  the returned `CertifiedResidualBound` has `rigor="failed"` and a
  `failure_reason` string naming the missing dependency. Silent fallback
  to heuristic is forbidden.
- Metric: `verifier_availability_error_count = 0` on a clean base install
  when no rigorous backend is requested.

### AC2 (J2) — Backend parity
- **Given** a pinned manufactured Poisson fixture on the unit box with
  fixed `(seed, dtype, device, shape)`, **when** the same residual is
  evaluated through both `TorchResidualVerifier` and
  `JaxVerifyResidualVerifier`, **then** the residual norms match within
  a dtype-specific tolerance (`1e-6` for float64, `1e-4` for float32).
- Metric: `torch_jax_residual_L2_gap ≤ tolerance` where the tolerance is a
  `MetricThreshold` from `CertificateConfig.thresholds`, not a hardcoded
  constant.

### AC3 (J3) — Soundness on manufactured solutions (inherits verified_error_certificate.spec.md AC2)
- **Given** manufactured-solution problems with a known truth, **when**
  a rigorous verifier (torch or jax) is run, **then** the true error
  never exceeds the certified bound across a Hypothesis sweep.
- Structural guard: a `CertifiedResidualBound` with `rigor="rigorous"`
  **must** have `domain_coverage="full"` (validator-enforced). A grid-
  sampled or partial-coverage bound **cannot** claim rigor. This
  eliminates the "heuristic passed as rigorous" failure mode at the
  type-system level.
- Metric: `bound_violations = 0`.

### AC4 (J4) — MCTS semantic invariance under WS3 JAX batch wrap
- **Given** `CertificateConfig.use_jax_batch_eval = True` and a JAX-traceable
  evaluator, **when** `BatchMCTS.search(...)` runs, **then** the visit
  distribution matches the pre-WS3 Torch/eager baseline byte-for-byte on
  a fixed seed and fixed game sequence.
- The regression oracle is the existing `tests/mcts/test_backup_modes.py`
  suite plus a new `tests/mcts/test_jax_batch_evaluator_boundary.py` that
  replays those cases with the JAX wrapper enabled.
- If any `BatchMCTS` test drifts under the JAX path, the feature is
  disabled by default and requires an explicit opt-in.
- Metric: `mcts_semantic_drift_count = 0`.

### AC5 (J5) — Cost accounting
- Every `CertifiedResidualBound` carries three non-negative wall-clock
  fields: `compile_wall_s`, `cert_wall_s`, `steady_state_wall_s`.
- `HardwareMeta` records `torch_version`, `jax_version`,
  `jax_verify_version`, `device`, `cuda_capability`, `dtype`.
- On rigorous JAX runs the first call must record `compile_wall_s > 0`;
  a subsequent call on the same shape/dtype must record
  `compile_wall_s = 0.0` (JIT cache hit).

### AC6 — Backwards compatibility with PR #1
- Every `CertificateConfig()` constructor call that worked at the tip of
  PR #1 must produce a `CertificateConfig` that JSON-dumps to a superset
  of the pre-WS1 dump (only new-with-default fields added). Guarded by
  `tests/pde/certificate/test_config_backcompat.py`.
- Every pre-WS1 `Certificate(...)` constructor must still validate.
- The Gate 5 AST hot-path guard from PR #1 stays green: no imports between
  `src/pde/certificate/**` and `src/mcts/**` in either direction. WS3's
  JAX batch wrapper lives under `src/backend/jax_batch/` for exactly this
  reason.

## 5. Cost model (normative)

| Track / verifier | Backend | Expected cost (unit box 2D Poisson) | When run |
|---|---|---|---|
| A | Torch or NumPy residual estimator | ≤ 10% of solve | Every pinned run |
| B | `HeuristicGridResidualVerifier` | seconds | CI smoke, `rigor="heuristic"` |
| B | `TorchResidualVerifier` (autoLiRPA) | ~10³ s (Table 2 of arXiv:2603.19165) | Batch, opt-in |
| B | `JaxVerifyResidualVerifier` (jax_verify) | cold: 10²–10³ s incl. compile; steady-state: 10¹–10² s | Batch, opt-in, GPU-recommended |

## 6. Dependencies and risks

- **Depends on:** PR #117 foundation (`Certificate`, `StabilityConstantRegistry`,
  logging binder, AST guard); `src/backend/{interface,jax_backend,torch_backend}.py`
  for the Protocol pattern and JAX backend availability; `src.templates.registry.create_typed_registry`.
- **Risk (accepted, documented in ADR):** `jax_verify` on PyPI is at version
  1.0 only and appears unmaintained. Mitigation: WS2 ships the Torch verifier
  first; JAX verifier is behind `[certificate-rigorous]`, `[jax]`, and a
  runtime import guard. Users can rely entirely on the Torch tier if
  `jax_verify` breaks.
- **Risk:** the existing `[jax]` extra pins `jax==0.4.30` for `flax==0.9`
  compatibility. `jax_verify==1.0` compatibility with this pin must be
  proven in WS2 before landing.
- **Risk:** heuristic-tier bounds use the placeholder stability constants
  from PR #1's registry (magnitude 1.0 for Poisson/Heat). This does not
  affect *soundness* — the ratio (true error / bound) is what's tested —
  but downstream numeric magnitudes are uncalibrated until the stability
  registry is calibrated per the `stochastic_galerkin_nke` convention.

## 7. References (all verified)

- arXiv:2603.19165 — Mukherjee, Fitzsimmons, Del Rey Fernández, Liu (Waterloo):
  Track B methodology, Tables 2/5 cost figures.
- Google-DeepMind `jax_verify` — https://github.com/google-deepmind/jax_verify
  (CROWN/IBP family in JAX; version 1.0 only on PyPI at the time of writing).
- autoLiRPA — https://github.com/Verified-Intelligence/auto_LiRPA
  (Torch-native LiRPA family; primary rigorous backend).
- Repo: `specs/verified_error_certificate.spec.md` (parent spec);
  `src/backend/interface.py` (BackendInterface Protocol pattern);
  `src/backend/jax_backend.py` (JAX backend already integrated);
  `src/templates/registry.py::create_typed_registry` (registry pattern);
  `tests/mcts/test_backup_modes.py` (WS3 regression oracle);
  `CLAUDE.md` 2026-08-12 "Certificate soundness (foundation)" row.
