# AlphaGalerkin — DOE Genesis Mission Phase I Full Proposal Outline

**Solicitation:** DOE Genesis Mission DE-FOA-0003612 (Phase I, $500K–$750K, 9 months)
**Secondary track:** DOE ASCR SBIR Phase I (Release 2, TBD)
**Filed with Phase I:** Phase II Letter of Intent (same-day filing)

**Template note:** Download `phase_i_template.xlsx` (or current equivalent) from `science.osti.gov` before finalizing and check for the current amendment (Amendment 000003+ at time of writing `[VERIFY]`). Section names below mirror the user's Execution Plan (Track C.3) and the existing `docs/proposals/templates/sbir_phase1.md` skeleton.

---

## Ownership legend

| Tag | Meaning |
|---|---|
| `Claude Code` | Claude Code drafts; PI reviews and edits |
| `[HUMAN WRITES]` | PI or collaborator writes from scratch; Claude Code does not draft |
| `[HUMAN DECISION]` | Strategic / legal / budget decision that Claude Code will not pre-empt |

---

## Section map

| # | Section | Owner | Length target (pages) | Source material in repo |
|---|---------|-------|----------------------|-------------------------|
| 0 | Cover page / metadata | `[HUMAN WRITES]` | 1 | SAM.gov UEI, entity record |
| 1 | Executive summary | `[HUMAN WRITES]` | 1 | Distillation of §2–§8 + concept_note.md |
| 2 | Motivation / National S&T Challenge alignment | `[HUMAN DECISION]` + `Claude Code` | 1–2 | `[HUMAN DECISION]` pick 2–3 of 17 Genesis National S&T Challenges; then Claude drafts alignment |
| 3 | Background + related work | `Claude Code` | 2 | `docs/doe_genesis/related_work.md` (Track B.5 deliverable); pulls from `IP_STRATEGY.md` and `docs/proposals/DIFFERENTIATION_MATRIX.md` |
| 4 | Technical approach | `Claude Code` | 3–4 | `docs/doe_genesis/mdp_specification.md` (Track B.4); `src/pde/games/`, `src/pde/mcts_adapter.py`, `src/mcts/gumbel.py`, `src/modeling/attention.py` |
| 5 | Preliminary results | `Claude Code` assembles | 2 | `benchmarks/results/headline_2026_04/` (Pareto plot, CSV, HTML report from Track B.6); zero-shot transfer result (MSE 0.000209) from CLAUDE.md |
| 6 | Phase I work plan (9 months × 3mo milestones) | `Claude Code` drafts, human approves | 2 | Execution plan Track B+C milestones |
| 7 | Phase II transition plan | `[HUMAN DECISION]` + `Claude Code` | 1–2 | `[HUMAN DECISION]` specific scientific target (plasma turbulence FES / electronic structure BES / subsurface / other) |
| 8 | Team & management | `[HUMAN WRITES]` | 1–2 | Biosketches from PI and co-PI(s); `partners/academic_candidates.md` shortlist |
| 9 | Facilities, resources, compute | `Claude Code` drafts | 1 | Existing repo compute footprint; `[HUMAN DECISION]` on cloud vs on-prem vs DOE lab allocation |
| 10 | Budget + justification | `Claude Code` drafts, human finalizes | 1–2 | `proposal/budget_justification.md` (this package) |
| 11 | Data Management Plan (DMSP) | `Claude Code` drafts | 1 | `docs/doe_genesis/dmsp.md` to be created; references open-source release in Phase I work plan |
| 12 | Letters of support (appendix) | `[HUMAN WRITES]` / `[HUMAN DECISION]` | n/a | From co-PI + lab partner; see `partners/outreach_template.md` |
| 13 | Biosketches (appendix) | `[HUMAN WRITES]` | 2 per person | From PI + co-PI(s) |
| 14 | Current and pending support (appendix) | `[HUMAN WRITES]` | 1 per person | Per DOE template |
| 15 | Phase II LOI (separate same-day filing) | `Claude Code` drafts, human approves | 1–2 | See §17 below |

**Targeted total page count:** ≤ 20 pages for main technical volume, excluding biosketches, budget, DMSP, letters. `[VERIFY]` against current amendment page limits.

---

## Section detail — drafting notes

### §2 Motivation / National S&T Challenge alignment
- `[HUMAN DECISION]` Pick 2–3 of the 17 Genesis National S&T Challenges. Candidate areas mentioned across this package (not binding): fusion energy / plasma confinement; advanced materials electronic structure; subsurface flow for carbon storage; turbulent combustion for decarbonization; climate sub-grid parameterization.
- Claude Code draft output: map each chosen challenge to a specific PDE class → tie to AlphaGalerkin benchmark already in repo (Poisson L-shaped ≈ fracture/singular-geometry surrogate; Burgers shock ≈ combustion/shock hydrodynamics; Taylor-Green NS ≈ turbulence closure).
- Do not commit numerical claims in §2; those belong in §5.

### §3 Background + related work
- Pulls directly from `docs/doe_genesis/related_work.md` (to be created in Track B.5).
- Compare to: PINN (Raissi 2019), FNO (Li 2020), DeepONet (Lu 2021), MeshGraphNets (Pfaff 2021), RL-for-AMR (Yang 2023, Foucart 2023, Freymuth 2024).
- Key framing: *learn the method, not the solution*; classical convergence guarantees preserved; no training data required.

### §4 Technical approach
- MDP specification: state, action, reward, transition, termination — all in `docs/doe_genesis/mdp_specification.md` (Track B.4).
- Galerkin attention: `src/modeling/attention.py` — Q(K^T V), Monte Carlo normalization, LBB constraint (dim(K) ≥ dim(Q)).
- FNet mixing: `src/modeling/` — torch.fft.rfft2, O(N log N).
- Gumbel MCTS: `src/mcts/gumbel.py` — sequential halving, improved policy targets, Gumbel-Top-k sampling.
- PDE game wrappers: `src/pde/games/basis_selection.py`, `src/pde/games/mesh_refinement.py`.
- Adapter: `src/pde/mcts_adapter.py` bridges PDE games to the generic MCTS `GameInterface`.
- Integration entry point: `src/alphagalerkin/solver.py` (Track B.1 deliverable).

### §5 Preliminary results
- Headline figure: `benchmarks/results/headline_2026_04/pareto_plot.png` — L² error vs wall-clock on L-shaped Poisson, Burgers shock, Taylor-Green NS, comparing AlphaGalerkin to uniform FDM, Dörfler AMR, hp-adaptive scikit-fem.
- CSV with every run: `(problem, method, refinement_level, n_dof, l2_error, wall_time_seconds, seed)`.
- Supplementary: zero-shot transfer MSE = 0.000209 on 19×19 (trained on 9×9) per CLAUDE.md milestone 2026-01-26. Framed honestly as *physics PoC validation*, not a Genesis benchmark result.
- **Go/no-go:** If AlphaGalerkin is not on the Pareto frontier on at least one benchmark, surface to PI *before* writing §4–§7 around a null result (per Execution Plan hard constraint).

### §6 Phase I work plan

| Month | Milestone | Deliverable | Repo artifact |
|-------|-----------|-------------|---------------|
| 1 | Finalize benchmark suite + scikit-fem baseline | hp-adaptive FEM solver | `src/research/fem_baseline.py` |
| 2 | Unified solver entry point | `solve(operator, target, budget) → SolverResult` | `src/alphagalerkin/solver.py` |
| 3 | Pareto plot regeneration from committed code | Reproducible HTML + CSV | `benchmarks/results/` |
| 4–5 | DOE-relevant problem integration | `[HUMAN DECISION]` e.g. plasma/fusion sub-problem | new `src/pde/operators/*` |
| 6 | Public open-source release | Tagged release | GitHub public repo `[HUMAN DECISION]` |
| 7 | Paper draft submitted (arXiv + venue) | Preprint | `docs/paper/` |
| 8 | Phase II proposal draft | Full Phase II proposal | `proposal/phase_ii/` |
| 9 | Final report + transition package | DOE final report | `docs/doe_genesis/final_report.md` |

### §7 Phase II transition plan
- `[HUMAN DECISION]` Specific DOE scientific target. Candidates: plasma turbulence (FES), electronic structure (BES), subsurface reactive transport (BES/FES joint), turbulent combustion (BES).
- Scaling axes to discuss: problem dimensionality (2D→3D), multi-physics coupling, lab-scale HPC integration (MFEM / PETSc).
- Technology transition mechanisms: DOE lab subaward, open-source adoption, commercial dual-license (see `sbir/commercialization_plan.md`).

### §9 Facilities, resources, compute
- Current: PI personal compute; existing repo runs on single-GPU workstation.
- Phase I compute demand: `[HUMAN DECISION]` — cloud credits vs DOE INCITE/ALCC proposal (long lead time, likely out of scope for Phase I) vs lab partner compute allocation.
- Software: existing repo (MIT-licensed core `[HUMAN DECISION]`), scikit-fem (BSD), PyTorch (BSD-style).

### §10 Budget + justification
- See `proposal/budget_justification.md`.

### §11 Data Management Plan
- Public release of all Phase I code under permissive license (`[HUMAN DECISION]` MIT or Apache 2.0).
- Benchmark datasets synthetic and regenerable — `benchmarks/results/` serves as archival artifact.
- Papers: arXiv preprint + open-access journal venue (`[HUMAN DECISION]` SIAM SISC, JCP, CMAME, JMLR).

### §12 Letters of support
- Minimum: 1 academic co-PI (or committed collaborator) + 1 DOE lab partner.
- Optional: 1 industrial endorser for Phase II transition credibility (`[HUMAN DECISION]` — industrial partner identity).

---

## §15 Phase II Letter of Intent (same-day filing)

Phase II LOI is filed with the Phase I proposal to preserve the Direct-to-Phase-II pathway regardless of Phase I funding outcome. Structure:

| Section | Owner | Length |
|---|---|---|
| Phase II technical objectives | `Claude Code` drafts from §7 | ~1 page |
| Scientific target + DOE program office alignment | `[HUMAN DECISION]` | ~0.5 page |
| Team continuity (same PI, co-PI, lab partner) | `[HUMAN WRITES]` | ~0.5 page |
| Budget scaling ($1.5M–$3.75M over 3 years) | `Claude Code` drafts template, human finalizes | — |

LOI file location: `proposal/phase_ii_loi.md` (to be drafted once Phase I §7 text is locked).

---

## Cross-reference checklist before submission

- [ ] Every numerical claim in §3, §4, §5 is traceable to a committed file in the repository.
- [ ] `[HUMAN DECISION]` blocks in §2, §7, §9, §10, §11, §12 are all resolved.
- [ ] `[VERIFY]` items across `partners/academic_candidates.md` and `partners/lab_candidates.md` are resolved for any name cited in §8 or §12.
- [ ] Red-team review complete (2 external reviewers per Track C.5): one applied mathematician, one ML researcher. `[HUMAN DECISION]` pick reviewers; deliver full draft ≥ 10 days before submission.
- [ ] SAM.gov UEI active; Grants.gov registered; PAMS registered; Genesis Consortium membership active (Track A).
- [ ] Phase II LOI drafted and filed same-day (§15).
