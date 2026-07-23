# Prior-Art Review: MCTS look-ahead for AMR / Galerkin basis selection

> **Status:** Reviewed 2026-07-22. Supersedes the unqualified "No published papers combine
> MCTS with Galerkin methods for PDE/mesh refinement" claim used in earlier SBIR framing.
>
> **Sourcing caveat (read before citing).** Every identification below rests on
> WebSearch-surfaced metadata (titles, authors, venues, abstracts) corroborated across
> multiple independent result pages — **not** on full-text retrieval (the session egress
> policy blocked direct academic-host fetches). Nothing here is fabricated, but a human must
> confirm any citation before it enters a submitted proposal. Preprint-vs-journal status and
> exact page/DOI for two entries are flagged inline.

## Why this review exists

An earlier internal claim asserted that three references — **Yang 2023, Foucart 2023,
Huergo 2024** — already occupy the "RL/MCTS for AMR-planning" ground, and a stronger claim
asserted that *no* published work combines MCTS with finite-element / Galerkin methods. Both
needed independent verification before any SBIR positioning. This review verifies the three
references, surveys the broader landscape, and states the honest defensible delta.

## Finding 1 — the three named references are real, but all single-step RL (no MCTS)

The three papers define the **RL-for-AMR** canon. None uses MCTS or any multi-step
tree-search look-ahead; the internal claim conflated "RL" with "MCTS".

| Ref | Identity | Method | MCTS? |
|---|---|---|---|
| **Yang et al. 2023** | "Reinforcement Learning for Adaptive Mesh Refinement," AISTATS 2023, PMLR v206:5997–6014 (arXiv:2103.01342); Yang, Dzanic, Petersen, … Kolev, Anderson, Faissol (LLNL) | AMR as an MDP; deep policy-gradient refinement policy vs error estimators | **No** (single-step policy) |
| **Foucart, Charous, Lermusiaux 2023** | "Deep Reinforcement Learning for Adaptive Mesh Refinement," *J. Comput. Phys.* 491:112381 (arXiv:2209.12351); MIT MSEAS | Local **POMDP** policy networks trained from simulation; train small → deploy large | **No** (deep RL, POMDP) |
| **Huergo, Rubio, Ferrer 2024** | "A reinforcement learning strategy for p-adaptation in high-order solvers," *Results in Engineering* v21 (arXiv:2306.08292); UPM | **PPO** actor-critic chooses polynomial order p per element in HORSES3D | **No** (PPO) |

## Finding 2 — the landscape is uniformly single-step RL

A broader survey confirms the pattern: every located AMR-ML method picks the *immediate*
refine/mark/coarsen action with a learned policy or value function — none plans multiple
steps ahead with a search tree.

| Citation | What it does | MCTS? |
|---|---|---|
| Dzanic/Yang et al., "Multi-Agent RL for AMR," AAMAS 2023 (arXiv:2211.00801); "Learning robust marking policies…" (arXiv:2207.06339) | AMR as multi-agent RL; robust marking vs Dörfler | No (MARL) |
| Freymuth et al., "Swarm RL for AMR," NeurIPS 2023 (arXiv:2304.00818); "ASMR: …Local Rewards" (arXiv:2406.08440) | Mesh-as-swarm RL; ~30× vs uniform, matches error-based AMR without an oracle | No (swarm RL) |
| Lorsung & Barati Farimani, "MeshDQN," *AIP Advances* 13:015026 (arXiv:2212.01428) | Graph-NN Deep-Q-Network coarsens CFD meshes | No (value-based DQN) |

## Finding 3 — MCTS + finite elements DOES exist (for a different problem)

This is the qualifier that **falsifies the blanket claim** and must be stated honestly:

- **TreeMesh** — Hua Tong, "Generate plane quad mesh with neural networks and tree search"
  (arXiv:2111.07613, 2021): explicitly couples **RL + MCTS** with finite-element **quad mesh
  generation** (element extraction). Genuine MCTS+FEM — but mesh *generation*, not
  error-driven refinement and not basis selection. *(Reads as a single-author preprint; treat
  as such, not a journal paper.)*
- **MCTS-AL** — "Highly Efficient Discovery of 3D Mechanical Metamaterials via Monte Carlo
  Tree Search": MCTS + CNN + FEM for materials **design**, not discretization.
- **SETS** — "Monte Carlo Tree Search with Spectral Expansion…," *Science Robotics* 2024
  (arXiv:2412.11270): MCTS + spectral expansion for robot **planning**; "spectral" is
  linearized-dynamics eigenstructure, not a Galerkin PDE discretization.

Searches for MCTS applied to error-driven **h/p AMR** or to **Galerkin/spectral basis
selection** returned **no** prior work. "Galerkin" is the cleanest differentiator.

## Honest defensible delta

The **narrow methodological delta survives**: no published work applies **MCTS multi-step
look-ahead to error-driven adaptive refinement or to Galerkin/spectral basis selection**.
The entire AMR-RL canon is single-step policy RL, so "multi-step tree-search planning vs.
myopic RL / greedy Dörfler marking" is a genuine, previously-unoccupied framing, and
"MCTS + Galerkin basis selection" in particular returns zero prior art.

**Two honesty constraints on how this is framed:**

1. **Do not use the blanket claim.** "No published papers combine MCTS with FEM / mesh
   refinement" is **false** as written (TreeMesh). Use the defensible form:
   > *"MCTS multi-step look-ahead for error-driven adaptive **refinement** and **Galerkin
   > basis** selection is unpublished; the only prior MCTS+finite-element work (TreeMesh)
   > targets mesh **generation**, a distinct problem."*

2. **Novelty ≠ superiority.** The delta is a *method* novelty, not a demonstrated win. The
   single-step RL papers already match classical error-estimator marking, so multi-step
   look-ahead must earn an **empirical** advantage — ideally at **matched wall-clock**, not
   just matched DOF — to be compelling. The repo's own `lshape_amr_compare` result is the
   relevant honest data point: an *untrained* MCTS refinement policy beats Dörfler by a few
   percent at matched DOF but is far worse at matched wall-clock, so the compelling claim
   depends on a trained evaluator that closes the wall-clock gap (out of scope today).

## Addendum (2026-07-23) — NKE and the stochastic Galerkin layer

The Neural Kolmogorov Equations paper (arXiv:2607.19173) is now cited and partially
reused as the basis of the additive stochastic Galerkin operator-splitting layer
(`src/pde/stochastic/`, `specs/stochastic_galerkin_nke.spec.md`). Its novelty-boundary
entry lives in `docs/related-work.md` (guard-tested): NKE does **not** do MCTS or
planning of any kind, does **not** do adaptive basis or mesh selection, and makes **no**
LBB/inf-sup claims — so it neither overlaps with nor undercuts the narrow delta above,
and AlphaGalerkin claims no LBB properties for the stochastic layer either. Provenance
caveat: the layer was implemented from the standard moment-Galerkin derivation because
the paper was unreachable at implementation time; a paper-exact cross-check remains an
open reviewer follow-up.
