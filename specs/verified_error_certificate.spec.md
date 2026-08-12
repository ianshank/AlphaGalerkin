# Spec: verified_error_certificate — two-track certified error bounds for all PDE solutions

> **Status:** Draft (v2 — post-verification pass)
> **Owner:** AlphaGalerkin core
> **Primary module(s):** `src/pde/certificate/` (new subpackage), `src/research/baselines.py` (`DorflerAMRSolver` indicator machinery), `src/poc/scenarios/llm_prior_ablation.py`, `src/research/transfer_baseline_compare.py`
> **Config class:** `src.pde.certificate.config.CertificateConfig` — thresholds **must reuse the canonical `src.poc.config.MetricThreshold`** (no parallel schema; per the spec-tree peer-review correction)
> **Tracking:** CLAUDE.md Next-Steps row "Verified error certificates (two-track)"
> **Origin:** Response to peer review of `docs/mathematical-superintelligence-x-alphagalerkin.md`. v2 corrections from evidence-verification pass: indicator machinery path fixed (`src/research/baselines.py`, not the game file), MetricThreshold reuse mandated, internal honesty precedent cited, exact certification cost figures, `PRIOR_ART_REVIEW.md` referenced rather than duplicated.

---

## 1. Motivation

AlphaGalerkin's differentiation rests on a claim currently asserted but not shipped:
every solution carries a *rigorous, machine-checkable error bound*. The inheritance
argument (`docs/doe_genesis/theory.md`) shows MCTS only ever selects conforming
subspaces \(V_h \subset V\), so classical bounds apply at every node — but the repo
emits no certificate artifact (confirmed: the only "certificate" in the codebase is
the critique that FNO lacks one). Meanwhile `DIFFERENTIATION_MATRIX.md` lists
"a posteriori error bounds" as Classical AMR's sole advantage.

**Internal precedent (why this is P0, not nice-to-have):** the project has twice been
burned by unverifiable numbers — the F0 backup defect produced a fabricated
"~11–14% win" headline (retracted 2026-07-10/23, corrected to 0.9605), and the
"zero-shot transfer MSE 0.000209 / 240×" headline was a hardcoded notebook cell with
no artifact (corrected 2026-07-22 to a falsifiable benchmark). Certificates are the
*systematic* fix for the failure class those incidents belong to: a number that ships
with its own machine-checkable justification cannot be silently fabricated.

**Key mathematical correction (from meta-review):** certification splits into two
tracks. Galerkin orthogonality (\(r \perp V_h\)) holds *only* for the exact Galerkin
projection in \(V_h\); the neural-operator path produces arbitrary admissible
\(\tilde{u} \in V\) and must use the general residual bound. Conformity (exact
boundary-condition enforcement), not orthogonality, is the load-bearing property.

## 2. Mathematical basis

For any admissible \(\tilde{u} \in V\) (boundary conditions exactly enforced) and an
operator with stability constant \(C_0 > 0\) (inf-sup \(\beta\) in the coercive case):

\[
\|u - \tilde{u}\| \;\le\; C_0\,\|r(\tilde{u})\|, \qquad r(\tilde{u}) = f - A\tilde{u}.
\]

- **Track A (exact-Galerkin):** \(\tilde{u} = u_h\), the exact solve in the
  MCTS-selected \(V_h\). Orthogonality holds; classical residual-based a posteriori
  estimators apply. Cheap (≤ 10% of solve wall-clock), always on.
- **Track B (neural-operator):** \(\tilde{u}\) is network output (zero-shot transfer
  path). Orthogonality fails; follow arXiv:2603.19165 — certified uniform residual
  bound \(R_{cert}\) **over the whole domain** via dReal / autoLiRPA / ∂-CROWN, then
  \(\|u - \tilde{u}\| \le C_0 R_{cert}\). Certification cost is real: 2,705 s for the
  2D Poisson residual bound via autoLiRPA; 2,580 s for 1D Burgers via ∂-CROWN
  (Tables 2 and 5 of the paper). Boundary/IC terms are cheap (10–60 s).
- **Counterexample absorbed (2603.19165, Example 2):** vanishing residual *at
  collocation points* does not imply solution convergence (oscillating
  interpolants). Track B certificates must be domain-wide. Their fix — a compactness
  condition on the hypothesis class — is *naturally satisfied* by AlphaGalerkin's
  finite-dimensional conforming basis structure; the spec must state this rather than
  assume it.
- **Stability constants are inputs, not afterthoughts.** Helmholtz at high wavenumber
  is indefinite with degrading stability — the estimate must reflect that or the
  certificate is fiction.

## 3. Scope

**In scope**
- `Certificate` artifact: Pydantic model, versioned schema, JSON-serializable; fields
  include `track ∈ {A, B}`, bound value, norm, stability-constant provenance,
  verifier backend, cost metrics, and a `rigor ∈ {rigorous, heuristic}` label.
- Track A estimator for pinned scenarios (L-shaped Poisson first), reusing —
  after audit — `DorflerAMRSolver._compute_indicators_2d` in
  `src/research/baselines.py`. Reference implementation available in-repo: scikit-fem
  (already a benchmarked baseline per `src/alphagalerkin/__init__.py`).
- Track B estimator for the zero-shot transfer path (`transfer_baseline_compare`).
- Stability-constant registry keyed on `src.pde.config.PDEType`: per-operator declared
  \(C_0\)/\(\beta\) source (`analytic | estimated | unbounded_with_warning`).
- Batch certification CLI: `python -m src.pde.certificate.run --scenario ...`.
- Cost instrumentation per certificate: `cert_wall_s`, `cert_peak_mem_mb`,
  `cert_cost_usd` (on metered hardware).

**Out of scope (Tier 2+)**
- Lean 4 formalization of the bounds. Prior art is explicit: complete Coq proof of
  Lax–Milgram (Boldo, Clément, Faissole, Martin, Mayero, CPP 2017). Any future claim
  is "ported to Lean 4 / Mathlib," never "first."
- Online/in-search certification inside MCTS rollouts (cost-prohibitive: minutes–hours
  per solution vs. per-rollout latency; see the matched-compute result in
  `lshape_amr_compare` — ~350× solve cost already killed the naive MCTS arm there).
- hp-refinement estimators beyond pinned operator families.

## 4. Acceptance Criteria

### AC1: Coverage
- **Given** any pinned scenario run (`lshape_amr_compare`, `transfer_baseline_compare`,
  `llm_prior_ablation` ID arm), **when** the run completes, **then** every emitted
  solution carries a `Certificate` (schema fields per §3).
- Metric: `certificate_coverage = 1.0` on pinned scenarios.

### AC2: Soundness on manufactured solutions
- **Given** manufactured-solution problems (truth known, per the `new-pde-operator`
  skill convention), **when** certificates are computed, **then** the true error never
  exceeds the certified bound across a Hypothesis parameter sweep.
- Metric: `bound_violations = 0`; effectivity index (bound / true error) thresholds
  are **placeholders calibrated from the first measured run**, per the
  `stochastic_galerkin_nke` gate-calibration convention. No invented constants ship.
  (Reference sanity check, not a gate: 2603.19165 Table 2 shows certified bound
  4.03e-3 vs reference error 1.80e-3 — effectivity ≈ 2.2 on 2D Poisson.)

### AC3: Track correctness
- Routing is by **provenance**, not by flag: exact solve in \(V_h\) → Track A;
  network output → Track B with domain-wide bound. Pointwise-at-training-nodes
  bounds are a test failure, not a downgrade.
- Metric: `track_misrouting = 0` (property test with known-provenance fixtures).

### AC4: Cost budget
- Certification is a **batch artifact**, never on the MCTS rollout hot path.
- Track A: ≤ 10% wall-clock overhead over the solve itself on pinned 2D Poisson.
- Track B: configurable budget, default **1 h per solution on CPU**; backend and
  hardware recorded on the certificate. Reference points: 2,705 s (2D Poisson,
  autoLiRPA), 2,580 s (1D Burgers, ∂-CROWN).
- Budget overruns are warnings during calibration, hard gates after — same pattern
  as `llm_call_p95_latency_ms` in `headline_runs.spec.md`.

### AC5: Stability-constant honesty
- Every `PDEType` registry entry declares its \(C_0\)/\(\beta\) source. Helmholtz
  entries carry a wavenumber-dependent estimate or an explicit `UNBOUNDED` marker;
  certificates on `UNBOUNDED` operators render as "residual bound only — no error
  guarantee."
- Metric: `undocumented_stability_constants = 0`.

### AC6: Regression surface
- New CLAUDE.md row: "Certificate soundness" — `pytest tests/pde/certificate/ -v`
  (with `COVERAGE_CORE=pytrace` per repo convention), `mypy --strict` clean,
  additive-only changes, coverage gate on `src/pde/certificate/`.
- Shared-surface guard: any touch of `src/research/baselines.py` must keep
  `tests/research/test_baselines*.py` green (documented SBIR P40 surface).

## 5. Cost model (normative)

| Track | Backend | Expected cost | When run |
|---|---|---|---|
| A | Residual estimator (computable constants) | ≤ 10% of solve | Every pinned run |
| B | autoLiRPA / ∂-CROWN / dReal | 10³ s scale per solution | Batch, opt-in per scenario |
| B (cheap tier) | Dense-grid residual check, explicit coverage caveat | Seconds | CI smoke; `rigor: heuristic` |

Heuristic-tier certificates exist so CI stays fast but must never be presented as
rigorous in `results/` or business docs — the fabrication-precedent guard.

## 6. Dependencies and risks

- **Depends on:** `DorflerAMRSolver._compute_indicators_2d` in
  `src/research/baselines.py` (audit against rigorous-estimator constants first —
  element indicators with correct scaling, not just relative marking signals);
  `PDEType` registry in `src/pde/config.py`; `src.poc.config.MetricThreshold`.
- **Risk:** Track B verifier integration (autoLiRPA/∂-CROWN/dReal) is the heavy lift.
  Mitigate: ship Track A + heuristic-tier B first; rigorous B behind a GPU runbook
  (mirrors `llm_prior_ood`'s "Implemented (CPU) + Runbook (GPU)" pattern).
- **Risk:** stability constants for the Leap 71 helical operators (`helical_heat` /
  `helical_stokes` / `helical_magnetostatics`) are not classical — likely
  `UNBOUNDED` initially; that is an acceptable certificate state, not a blocker.
- **Prior art:** extend `docs/business/proposals/PRIOR_ART_REVIEW.md` (already
  corrected the blanket "no MCTS+FEM" claim via TreeMesh / arXiv:2111.07613) with a
  certified-bounds-for-RL-AMR pass before any SBIR novelty claim.
- **Interaction:** `operator_gate.spec.md` (LLM-prior verification) consumes the
  stability registry — land it in a shared module both specs import. Gate lands
  first (days; protects live `llm_prior_ablation` runs), certificates second (weeks).

## 7. References (all verified)

- arXiv:2603.19165 — Mukherjee, Fitzsimmons, Del Rey Fernández, Liu (Waterloo):
  Theorem 4 stability condition; dReal/autoLiRPA/∂-CROWN pipeline; Example 2
  counterexample; Tables 2/5 cost figures.
- Boldo, Clément, Faissole, Martin, Mayero — *A Coq Formal Proof of the Lax–Milgram
  Theorem*, CPP 2017 (prior art; Lean 4 port is the defensible claim).
- Hillebrecht et al. 2025 — a posteriori bounds for PDE-defined PINNs.
  *(Distinct from arXiv:2509.26122 — peer review conflated them.)*
- arXiv:2510.01346 — Aristotle (informal reasoner + Lean kernel; the architectural
  template for `operator_gate.spec.md`).
- Repo: `docs/doe_genesis/theory.md` (inheritance argument); `CLAUDE.md` changelog
  2026-07-05 / 2026-07-10 / 2026-07-22 (F0 retraction; transfer-benchmark correction);
  `specs/lshape_amr_compare.spec.md`; `specs/transfer_baseline_compare.spec.md`;
  `docs/business/proposals/PRIOR_ART_REVIEW.md`.
