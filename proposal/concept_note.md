# AlphaGalerkin — DOE Genesis Mission Phase I Concept Note

**Solicitation:** DOE Genesis Mission DE-FOA-0003612 (Phase I, $500K–$750K, 9 months)
**PI:** Ian Cruickshank | **Co-PI:** `[HUMAN DECISION]` | **Lab partner:** `[HUMAN DECISION]`
**Date:** 2026-04-23

---

## Problem

PDE solving is the computational bottleneck throttling DOE scientific-discovery workflows. Two concrete examples from the Genesis National S&T Challenges — `[HUMAN DECISION — pick from Genesis 17 topics; candidate areas include plasma-confinement design for fusion power, electronic-structure calculations for advanced materials, subsurface-flow modeling for carbon storage, and turbulent-combustion simulation for decarbonization]` and `[HUMAN DECISION — pick a second Genesis topic]` — each require tens of thousands of adaptively refined PDE solves to explore design space. Current adaptive-mesh-refinement (AMR) heuristics such as Dörfler marking and Zienkiewicz-Zhu indicators are *myopic*: they optimize the next refinement step with no look-ahead. This produces suboptimal meshes, wastes wall-clock on poorly placed refinement, and forces expert-in-the-loop tuning for every new problem class.

## Approach

AlphaGalerkin frames discretization choice as a sequential decision problem and applies Monte Carlo Tree Search with a learned policy/value network — the same planning algorithm behind AlphaZero — to select among classical Galerkin discretizations (basis order, element refinement, quadrature). **We learn the method, not the solution.** The MCTS search tree explores compositions of (element, h-refine/p-increase/coarsen/terminate) actions with multi-step look-ahead over a reward that balances L² error reduction against compute cost. The inner solve is a classical Galerkin FEM step — so every action the policy takes still inherits the convergence theory of the underlying discretization.

## Differentiation

Unlike PINN, FNO, and DeepONet, AlphaGalerkin does not replace the numerical method with a surrogate — classical convergence guarantees are preserved because every rollout uses a bona-fide Galerkin solve. Unlike policy-gradient RL-for-AMR (Yang 2023, Foucart 2023, Freymuth 2024), AlphaGalerkin has AlphaZero-style multi-step look-ahead, provable UCB exploration bounds, and requires no training data — it operates directly on the PDE. To our knowledge, no published work combines MCTS with Galerkin discretization selection for PDE solving or mesh refinement.

## Preliminary evidence

Headline Pareto plot (L² error vs wall-clock) across L-shaped Poisson, viscous Burgers with shock formation, and 2D Navier-Stokes Taylor-Green vortex — comparing AlphaGalerkin against uniform refinement, Dörfler AMR, and scikit-fem hp-adaptive — is produced by the repository's benchmark suite and archived at `benchmarks/results/headline_2026_04/pareto_plot.png`. Artifact regeneration: `scripts/run_sbir_demo.py`. No numerical performance claim is made in this concept note that is not traceable to that artifact.

## Team

**PI:** Ian Cruickshank — AI/ML systems engineer; prior AlphaZero implementation work (Civ6-AlphaZero); leads AlphaGalerkin repository development including Gumbel MCTS, Galerkin attention, and the PDE game framework.
**Co-PI (academic):** `[HUMAN DECISION]` — see `partners/academic_candidates.md` shortlist; target profile is a tenured applied mathematician with prior DOE ASCR funding and a 2022–2026 record in adaptive FEM or ML-for-PDEs.
**Lab partner:** `[HUMAN DECISION]` — see `partners/lab_candidates.md`; priority targets are LLNL (MFEM integration) and PNNL (PhILMs precedent).

## Ask

**Phase I:** $500K–$750K over 9 months, delivering the headline Pareto frontier on three benchmark PDEs, an MDP specification document, a public open-source release, and a DOE-relevant problem demonstration. **Phase II transition:** $1.5M–$3.75M over 3 years, scaling to a specific DOE scientific target (`[HUMAN DECISION]`).
