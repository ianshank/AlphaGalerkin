# ADR-0001: JAX at the Track B verifier and batched-evaluator boundaries only

**Status:** Accepted
**Date:** 2026-08-12
**Deciders:** AlphaGalerkin core (certificate stream)
**Related:** [`specs/jax_track_b_verifier.spec.md`](../../specs/jax_track_b_verifier.spec.md), [`specs/verified_error_certificate.spec.md`](../../specs/verified_error_certificate.spec.md)

## Context

`specs/verified_error_certificate.spec.md` §2 Track B commits AlphaGalerkin
to certified uniform residual bounds for neural-operator outputs, following
arXiv:2603.19165. Three backend candidates for the residual bound
propagation exist:

1. **autoLiRPA** — mature, torch-native, well-cited. Baseline cost per
   published benchmark (Table 2): ~2,705 s per 2D Poisson certificate.
2. **jax_verify** — CROWN/IBP family in JAX, `jit`/`vmap` amortises cold-
   start across a certificate batch. Version-1.0-only on PyPI.
3. **dReal** — SMT-based, torch-agnostic, high cost at high dimensionality.

The addendum "JAX for AlphaGalerkin — narrow scope, high leverage"
proposes an additional axis: use JAX not just at the verifier boundary
but also at the pure batched-evaluator boundary of `BatchMCTS`. It also
proposes migrating parts of MCTS itself, which is where the design
question turns.

## Decision

**JAX is admitted only at two boundaries:**

1. The Track B verifier boundary — as one of several selectable
   `verifier_backend` implementations in
   `src/pde/certificate/verifiers/jax_verifier.py` (WS2).
2. The pure batched-evaluator boundary of `BatchMCTS` — as an optional
   `jax.jit` + `jax.vmap` wrapper around the leaf evaluator, gated on
   `CertificateConfig.use_jax_batch_eval` and located under
   `src/backend/jax_batch/` (WS3).

**JAX is explicitly forbidden inside** MCTS tree traversal, node
expansion, backup, and selection. Those code paths are Python control
flow — JAX gains nothing from tracing them and the byte-for-byte
semantic invariants guarded by `tests/mcts/test_backup_modes.py` are
load-bearing (see the F0 backup retraction, CLAUDE.md 2026-07-05).

**Backend selection is explicit and per-run.** A Torch model must be
paired with a Torch verifier; a JAX model with a JAX verifier. No
automatic framework conversion inside a certification run.

## Alternatives considered

### A. PyTorch-only Track B (autoLiRPA)

Ship autoLiRPA as the sole rigorous Track B backend; do not touch JAX
for verification.

* **Pros:** single dependency chain; mature; the `[certificate-rigorous]`
  optional extra stays small and stable.
* **Cons:** the 2,705 s / certificate cost is a hard scaling ceiling —
  batch certification would benefit measurably from JAX's amortisation
  properties. Also forgoes the leverage of the existing `src/backend/`
  layer, which already ships a working `JaxBackend`.

### B. Full-JAX rewrite of MCTS + verifier

Migrate MCTS selection/expansion/backup to JAX for uniform tracing.

* **Pros:** consistency; potential end-to-end `jit`.
* **Cons:** MCTS control flow is inherently branching and stateful —
  JAX gains nothing here. The rewrite risks the `test_backup_modes`
  semantic invariants and re-opens the F0 backup class of bug.
  Prohibitive engineering cost for zero measurable win.
  **Rejected.**

### C. Chosen — bounded JAX at verifier + evaluator batch

* **Pros:** confines JAX to code paths that are already pure functional
  (verifier residual programs, batched evaluator forward passes); leaves
  the byte-for-byte semantic guarantees of the tree algorithm untouched;
  reuses the existing `src/backend/` layer; keeps `[jax]` extras
  optional; the AST hot-path guard from PR #1 continues to enforce
  isolation between `src/mcts/` and `src/pde/certificate/`.
* **Cons:** two backend chains to maintain (torch + jax). Mitigated by
  the shared `ResidualVerifier` Protocol and the fact that the heuristic
  and Torch tiers are always-available fallbacks — a broken
  `jax_verify` never breaks certification, only the JAX subvariant.

## Consequences

### Positive
- Additive-only change to PR #1's foundation. `CertificateConfig` gains
  fields with defaults so pre-WS1 constructors keep working byte-identically.
- Reuses `src/backend/interface.py::BackendInterface` as the pattern for
  `src/pde/certificate/interface.py::ResidualVerifier` (Protocol,
  runtime-checkable, backend-neutral at the interface level).
- The heuristic tier remains torch-and-numpy-only, so CI stays fast and
  the `[jax]` extra remains optional.

### Negative / accepted risk
- **`jax_verify==1.0` is unmaintained.** Mitigation: the Torch verifier
  ships first and remains an always-available fallback. If `jax_verify`
  breaks against a future `[jax]` pin bump, `verifier_backend="jax_verify"`
  fails closed with `VerifierUnavailableError` — no silent degradation.
- **Existing `[jax]` extra pins `jax==0.4.30` for `flax==0.9` compatibility.**
  `jax_verify==1.0` compatibility with this pin must be proven in WS2
  before landing. If incompatible, the JAX verifier is deferred until
  the `JaxBackend` migrates to a newer JAX/flax stack.
- Two verifier code paths to test in parity (AC2 / Gate J2).

### Neutral
- WS3's JAX batch wrapper is opt-in and defaults off. The `BatchMCTS`
  hot path remains torch-native for every existing configuration.

## Guardrails

1. **Gate 5 hot-path AST guard** from PR #1 stays in force —
   `src/pde/certificate/**` and `src/mcts/**` may not import each other
   in *either* direction. WS3's evaluator wrapper lives outside both
   under `src/backend/jax_batch/`.
2. **Fail-closed** — a certificate that cannot be computed within budget
   or that cannot import its verifier's backend emits with `rigor="failed"`
   or `rigor="heuristic"` and never silently claims rigor. Structural
   guard: `rigor="rigorous"` requires `domain_coverage="full"` at the
   type-system level.
3. **No hardcoded values** — every dtype, device, budget, tolerance,
   grid resolution is a typed `CertificateConfig` field or a
   `MetricThreshold`.
4. **Backwards compatibility** is a first-class AC (spec AC6) —
   pre-WS1 constructor calls must produce byte-identical dumps.

## References

- Meta-review of the addendum "JAX for AlphaGalerkin — narrow scope,
  high leverage" (2026-08-12; internal peer review).
- arXiv:2603.19165 — Mukherjee et al., *Rigorous Error Certification for
  Neural PDE Solvers*.
- Google-DeepMind `jax_verify` — https://github.com/google-deepmind/jax_verify
- `src/backend/interface.py`, `src/backend/jax_backend.py`,
  `src/backend/torch_backend.py` — existing dual-backend layer.
- CLAUDE.md 2026-07-05 (F0 backup retraction — the failure class that
  motivates the "no MCTS-JAX" invariant).
- PR #1 of the certificate series (2026-08-12; foundation layer).
