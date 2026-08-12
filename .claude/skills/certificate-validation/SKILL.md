---
name: certificate-validation
description: Kick off and deterministically validate the verified_error_certificate spec implementation — run when building or reviewing src/pde/certificate/
---

# Certificate Validation Skill

Kickoff checklist and deterministic validation harness for
`specs/verified_error_certificate.spec.md` (two-track certified error bounds).
Every gate below is a command with a binary pass/fail — no judgment calls.

## Kickoff checklist

1. **Read the spec** (`specs/verified_error_certificate.spec.md`) end-to-end; confirm
   Status and the CLAUDE.md Next-Steps row "Verified error certificates (two-track)".
2. **Scaffold the module** additively: `src/pde/certificate/` with `config.py`
   (`CertificateConfig` — thresholds reuse the canonical `src.poc.config.MetricThreshold`;
   no parallel schema), `certificate.py` (Pydantic `Certificate` artifact),
   `tracks/` (`track_a.py`, `track_b.py`), `run.py` (batch CLI).
3. **Stability registry** keyed on `src.pde.config.PDEType`: per-operator
   `C0`/`beta` source in `{analytic, estimated, unbounded_with_warning}`.
   Shared module — `specs/operator_gate.spec.md` will import it.
4. **Track A audit first:** review `DorflerAMRSolver._compute_indicators_2d`
   (`src/research/baselines.py`) against rigorous-estimator constants before reuse.
   The SBIR P40 surface must stay green (Gate 8).
5. **Sequencing:** land `specs/operator_gate.spec.md` first if both are in flight; the gate
   protects live `llm_prior_ablation` runs.

## Deterministic validation gates

Run top to bottom. Any failure blocks merge and blocks certificates from `results/`.

### Gate 1 — Static
```bash
ruff check src/pde/certificate/ tests/pde/certificate/
ruff format --check src/pde/certificate/ tests/pde/certificate/
mypy --strict src/pde/certificate/
```
Pass: all clean.

### Gate 2 — Artifact schema
```bash
pytest tests/pde/certificate/test_schema.py -v
```
Pass: `Certificate` JSON round-trip is lossless (Pydantic subclass field-loss class —
see `src/refinement/config.py` precedent); `track ∈ {A, B}` and
`rigor ∈ {rigorous, heuristic}` enums enforced; `cert_wall_s`, `cert_peak_mem_mb`,
stability-constant provenance present on every artifact.

### Gate 3 — Track routing (AC3)
```bash
pytest tests/pde/certificate/test_track_routing.py -v
```
Pass: known-provenance fixtures route correctly — exact solve in V_h → Track A,
network output → Track B; pointwise-at-training-nodes bounds are a hard failure;
`track_misrouting = 0`.

### Gate 4 — Soundness on manufactured solutions (AC2)
```bash
pytest tests/pde/certificate/test_soundness.py -v
```
Pass: true error ≤ certified bound on every manufactured-solution problem
(Poisson, L-shape `u = r^(2/3) sin(2θ/3)`), Hypothesis sweep over seeds and
parameters; `bound_violations = 0`. Effectivity index recorded, thresholds
calibrated from first measured run — **no invented constants**.

### Gate 5 — Cost budget (AC4)
```bash
pytest tests/pde/certificate/test_cost_budget.py -v
```
Pass: Track A overhead ≤ 10% of solve wall-clock on pinned 2D Poisson; Track B
records backend + hardware and respects the configurable budget (default 1 h CPU);
certificates are batch-only — assert no import of `src/pde/certificate/` from
`src/mcts/` rollout paths.

### Gate 6 — Stability honesty (AC5)
```bash
pytest tests/pde/certificate/test_stability_registry.py -v
```
Pass: every `PDEType` has a declared stability source
(`undocumented_stability_constants = 0`); Helmholtz entries carry a
wavenumber-dependent estimate or `UNBOUNDED`; `UNBOUNDED` certificates render
"residual bound only — no error guarantee".

### Gate 7 — Coverage
```bash
COVERAGE_CORE=pytrace pytest tests/pde/certificate/ -v --cov=src/pde/certificate --cov-branch
```
Pass: ≥ 85% branch coverage on `src/pde/certificate/`.

### Gate 8 — Shared regression surface (AC6)
```bash
pytest tests/research/test_baselines.py tests/research/test_baselines_2d.py \
       tests/research/test_pde_benchmarks.py tests/research/test_ns_baseline.py -q
```
Pass: all green (any touch of `src/research/baselines.py` — byte-for-byte
`inside=None` behaviour preserved, per `lshape_amr_compare` AC1).

## Guardrails

- **Additive only** — new subpackage, new optional registry keys; never change an
  existing `PDEType` value.
- **Heuristic tier is labeled** — `rigor: heuristic` certificates must never appear
  as rigorous in `results/` or business docs (F0 / transfer-MSE precedent:
  unverifiable numbers get retracted here).
- **Fail closed** — a certificate that cannot be computed within budget is emitted as
  failed, never silently dropped; `certificate_coverage = 1.0` counts failures too.
- **Regression surface** — add/extend the CLAUDE.md "Certificate soundness" row when
  the module lands.

## References

- Spec: `specs/verified_error_certificate.spec.md`
- Method: arXiv:2603.19165 (dReal / autoLiRPA / ∂-CROWN; Example 2 counterexample;
  Tables 2/5 cost reference points)
- Conventions: `.claude/skills/new-pde-operator/SKILL.md` (manufactured-solution
  proofs), `specs/stochastic_galerkin_nke.spec.md` (placeholder calibration),
  `specs/lshape_amr_compare.spec.md` (shared-solver guard)
