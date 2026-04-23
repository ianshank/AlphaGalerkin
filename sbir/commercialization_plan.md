# AlphaGalerkin — Commercialization Plan (DOE ASCR SBIR)

**Required for:** DOE ASCR SBIR (all DOE SBIR programs require a commercialization plan section).
**Not required for:** DOE Genesis Mission Phase I (Genesis is a research solicitation, not SBIR).

**Status of this document:** Draft scaffolding. Strategic revenue-model and customer decisions are `[HUMAN DECISION]`. IP strategy references the existing `/home/user/AlphaGalerkin/docs/proposals/IP_STRATEGY.md`.

---

## 1. Target markets

AlphaGalerkin's core capability — MCTS-guided selection among classical Galerkin discretizations, preserving classical convergence guarantees — is marketable across three tiers:

### Tier 1 — Engineering simulation software vendors
- **Ansys** (stock: ANSS, ~$30B market cap prior to Synopsys acquisition announcement `[VERIFY current status]`) — AMR is used extensively in Ansys Fluent and Mechanical; AlphaGalerkin's multi-step planning is a direct performance upgrade.
- **COMSOL Multiphysics** — private; adaptive meshing is a named feature.
- **Siemens Simcenter** — public (Siemens Digital Industries Software); AMR is part of STAR-CCM+.
- **Altair**, **Dassault SIMULIA (Abaqus)** — secondary targets.

**Go-to-market motion:** licensing AlphaGalerkin as a mesh-refinement module; revenue via per-seat royalty `[HUMAN DECISION]` or flat-fee OEM license `[HUMAN DECISION]`.

### Tier 2 — Scientific research computing users
- **DOE National Labs** — direct license or CRADA. See `partners/lab_candidates.md`. Each of the 5 labs targeted there has an in-house PDE workflow AlphaGalerkin could plug into (MFEM at LLNL, PETSc at ANL).
- **Academic HPC centers** — NSF-funded (NCSA, TACC, SDSC) and DOE-funded (NERSC, OLCF, ALCF) centers; distribution via open-source channel + paid support.
- **NIH / NIST / NASA** — secondary government-lab licensing.

**Go-to-market motion:** open-source core + commercial support contracts; alternatively, DOE lab cooperative licensing.

### Tier 3 — Applied R&D in energy, materials, and fluids industries
- **Advanced nuclear reactor developers** (TerraPower, X-energy, Kairos `[VERIFY commercial status]`) — neutronics and CFD.
- **Fusion startups** (Commonwealth Fusion, TAE, Helion, Tokamak Energy `[VERIFY]`) — plasma and MHD simulation.
- **Battery and materials R&D** (QuantumScape, Solid Power `[VERIFY]`) — electrochemistry PDEs.
- **Aerospace / defense primes** — CFD + structural; long sales cycles, high-value contracts.

**Go-to-market motion:** consulting + integration, with AlphaGalerkin as the core capability. High-touch, low-volume.

---

## 2. Competitive landscape

### Classical FEM stack (incumbent)
| Competitor | Type | AlphaGalerkin position |
|-----------|------|-----------------------|
| **Ansys Mechanical / Fluent** | Commercial FEM/CFD | AlphaGalerkin is a module/upgrade, not a replacement |
| **COMSOL Multiphysics** | Commercial multi-physics | Same positioning |
| **MOOSE** (Idaho National Lab) | Open-source FEM framework | Potential integration target rather than competitor |
| **deal.II** | Open-source FEM library | Adjacent — AlphaGalerkin could drive deal.II under the hood |
| **FEniCS / DOLFINx** | Open-source FEM | Adjacent — similar integration opportunity |
| **MFEM** (LLNL) | Open-source FEM, GPU-ready | **Highest-priority integration target** — see `partners/lab_candidates.md` Priority 1 |

### ML-based PDE stack (emerging)
| Competitor | Type | AlphaGalerkin position |
|-----------|------|-----------------------|
| **NVIDIA Modulus (formerly SimNet)** | PINN / FNO-based SciML framework | Differentiates: we retain classical convergence guarantees; Modulus does not |
| **DeepMind AlphaFold-like PDE analogs** | Proprietary neural solvers | Differentiates: they learn solutions on specific PDEs; we learn discretization strategy, transferable across PDEs |
| **Neural operators (FNO, DeepONet, GNO)** | Academic/open-source | Differentiates: we do not replace the numerical method; no training-data dependency |
| **RL-for-AMR (Yang 2023, Foucart 2023, Freymuth 2024)** | Research prototypes | Differentiates: AlphaZero-style look-ahead vs. myopic policy gradient |

Full differentiation matrix is in `/home/user/AlphaGalerkin/docs/proposals/DIFFERENTIATION_MATRIX.md`.

---

## 3. Revenue model options

All three are `[HUMAN DECISION]`. They are not mutually exclusive — a blended model is the likely end state.

### Option A — Dual-license open-source core + commercial enterprise tier
- Open-source core (MIT or Apache 2.0 `[HUMAN DECISION]`) with all research-use functionality.
- Commercial enterprise tier adds: (i) MFEM / PETSc / Modulus integration adapters; (ii) priority support SLA; (iii) pre-trained policy checkpoints for common PDE classes; (iv) on-prem deployment tooling.
- Revenue: annual enterprise license per organization, tiered by seat count.
- Pro: low-friction adoption drives funnel; customers self-serve evaluate before buying.
- Con: must carefully gate what stays commercial vs open-source.

### Option B — Consulting + support
- All code open-source.
- Revenue: integration consulting engagements, custom policy training, production-deployment support.
- Pro: fastest path to revenue; validates product-market fit with real customers.
- Con: does not scale; staffing-limited.

### Option C — Exclusive DOE lab licensing
- License to a DOE national lab (via CRADA or exclusive OSS adaptation agreement) for a specific mission area (e.g., fusion simulation).
- Revenue: upfront licensing fee + milestone payments.
- Pro: single customer, deep engagement, mission-critical use case anchors credibility.
- Con: single-customer concentration risk; limits commercial market expansion.

**Recommended blend (Claude Code draft, `[HUMAN DECISION]` to ratify):** Option A as core revenue model, Option B as near-term bridge revenue, Option C opportunistically when a lab partner moves from Phase II collaborator to production customer.

---

## 4. Traction plan — 3 Phase II pilot customers

Phase II pilots establish reference customers for Phase III scaling. Target three pilots, one from each tier of §1:

| Pilot # | Tier | Candidate | Status |
|---------|------|-----------|--------|
| 1 | Commercial ISV | `[HUMAN DECISION]` — Ansys, COMSOL, or Siemens | Not yet approached |
| 2 | DOE National Lab | `[HUMAN DECISION]` — see `partners/lab_candidates.md` priority 1–3 | Partner outreach in progress |
| 3 | Applied R&D | `[HUMAN DECISION]` — fusion startup or battery materials R&D | Not yet approached |

Pilot selection criteria:
- Active budget for mesh/solver improvements.
- Named technical sponsor who will defend the pilot internally.
- Measurable success metric (wall-clock reduction, DOF-for-equal-accuracy reduction, engineer-hour reduction).
- Willingness to provide a reference or testimonial if successful.

**Pilot deal structure `[HUMAN DECISION]`:** Typical options are (a) paid POC at below-cost rates in exchange for reference rights; (b) joint-development agreement with IP rights shared for pilot-specific extensions; (c) grant-funded subaward where DOE pays for both parties.

---

## 5. Intellectual property

See `/home/user/AlphaGalerkin/docs/proposals/IP_STRATEGY.md` for the full strategy. Summary:

- **Trade secrets (already protected):** MCTS reward engineering, specific training hyperparameters, MCTS-Galerkin integration methodology, data augmentation strategies.
- **Provisional patents (12-month window):**
  - Claim 1: MCTS-Guided Adaptive Mesh Refinement — core system+method claim.
  - Claim 2: Resolution-Independent Neural Operator Learning (Galerkin attention).
  - Claim 3: LBB-Stabilized Neural Attention Training.
- **Strategic publications:** arXiv-first for priority-date establishment on non-patented aspects.

`[HUMAN DECISION]` — file provisional applications before first public preprint or conference submission of §5-preliminary-results in the Genesis proposal. Coordinate filing date against proposal submission date.

---

## 6. Key decisions the human must make

1. `[HUMAN DECISION]` Revenue model (A / B / C / blend).
2. `[HUMAN DECISION]` Open-source license (MIT vs Apache 2.0) for the core.
3. `[HUMAN DECISION]` What stays commercial vs open-source in Option A.
4. `[HUMAN DECISION]` Specific 3 Phase II pilot customer targets.
5. `[HUMAN DECISION]` Provisional patent filing timeline relative to first public disclosure.
6. `[HUMAN DECISION]` Whether to incorporate commercialization milestones into Phase I work plan as risk-reduction evidence for SBIR reviewers.
