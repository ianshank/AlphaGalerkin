# Related Work and Honest Literature Comparison

> **Scope:** this is the **DOE-Genesis-specific** literature comparison. For the
> repo-wide novelty-boundary register (with the executable guard), see
> [`../related-work.md`](../related-work.md).

**Solicitation:** DE-FOA-0003612 (DOE Genesis Mission, Phase I)
**Framing:** PINN/FNO/DeepONet learn *the solution*. MeshGraphNets and
RL-for-AMR learn *the mesh*. AlphaGalerkin learns *the method* — an
optimal sequence of classical, provably-convergent discretization
operations — and therefore inherits classical FEM error bounds rather
than replacing them with learned approximators.

This document is written for skeptical ASCR applied-mathematics
reviewers. We neither overclaim novelty nor understate prior art.

---

## 1. Physics-Informed Neural Networks (PINNs)

**Reference.** Raissi, M., Perdikaris, P., & Karniadakis, G. E.
"Physics-informed neural networks: A deep learning framework for
solving forward and inverse problems involving nonlinear partial
differential equations." *Journal of Computational Physics*, 378,
686–707, 2019.

**What it does.** Represents the solution $u_\theta(x, t)$ as a
fully-connected MLP; minimizes a composite loss combining PDE residual
(via autodiff), boundary, and initial data. Mesh-free; differentiable
end-to-end.

**What it cannot do that AlphaGalerkin can.**

1. **No classical convergence rate.** PINN error decay is empirical and
   strongly problem-dependent. There is no $O(h^p)$ guarantee.
2. **Spectral bias.** PINNs struggle with high-frequency, multi-scale,
   and shocked solutions; extensive literature documents failure modes
   (Krishnapriyan et al. 2021, NeurIPS).
3. **Retraining per PDE instance.** Each new forcing or boundary
   condition requires retraining; PINNs do not amortize across problems.
4. **No multi-step planning.** PINN training is a fixed-point
   optimization, not a planning problem.

**Complementarity.** PINN-style residual autodiff is already used inside
AlphaGalerkin's `physics_loss.py` (`ResidualLoss`) as *one* regularizer.
AlphaGalerkin's outer loop — MCTS over discretization choices — is
orthogonal to the PINN residual formulation and could, in principle,
orchestrate PINN subsolvers as primitive actions in a future version.

---

## 2. Fourier Neural Operator (FNO)

**Reference.** Li, Z., Kovachki, N., Azizzadenesheli, K.,
Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A.
"Fourier Neural Operator for Parametric Partial Differential
Equations." *ICLR 2021*.

**What it does.** Learns a mapping between function spaces (forcing
$f \mapsto u$) via spectral convolutions in Fourier space. Fast
inference; discretization-invariant in a *grid-resampling* sense.

**What it cannot do.**

1. **Requires labeled training data.** FNO is supervised — it needs
   thousands of $(f_i, u_i)$ pairs from a ground-truth solver.
   AlphaGalerkin (MCTS-guided) requires none.
2. **No error certificate.** FNO has no a-posteriori error estimator;
   deploying it on a safety-critical problem (reactor design, airframe)
   requires an independent verification solve.
3. **Spectral truncation errors** on non-periodic domains and on
   geometries with corners remain problematic (though recent variants
   mitigate this).
4. **Distributional shift.** FNO accuracy degrades when test PDE
   parameters leave the training distribution.

**Complementarity.** AlphaGalerkin *uses* FFT internally (FNet spatial
mixing) for $O(N \log N)$ encoder throughput but never consumes FNO's
supervised solution prediction. FNO could serve as a fast coarse
predictor inside an AlphaGalerkin rollout (equivalent to a neural
preconditioner), but this is not our baseline.

### 2.1 FNO Variants: Practical Recommendations (2026 Perspective)

The single 2020 spectral-convolution idea has since fanned out into a
family of variants. For practitioners choosing a baseline:

- **Start simple.** Use F-FNO or AM-FNO (via the NeuralOperator library
  or PhysicsNeMo) for quick baselines on uniform-grid problems.
- **Need irregular geometry?** → Geo-FNO.
- **High frequencies or turbulence?** → U-FNO, group-equivariant FNO
  (GFNO) where symmetries can be exploited, or a Galerkin-hybrid
  approach.
- **Limited data or inverse problems?** → PINO (or add a physics
  residual loss to any variant).
- **Extreme resolution scaling or curriculum transfer?** → This is
  where our proposed AlphaGalerkin architecture is aimed. It combines a
  Galerkin-style attention operator, adaptive spectral mode selection
  (learned truncation of the active mode set per scale), and a
  curriculum that escalates resolution and PDE difficulty during
  training. We hypothesize it will improve zero-shot transfer to higher
  resolutions and stiffer regimes; this remains to be validated against
  the baselines below.

### 2.2 How We Intend to Evaluate AlphaGalerkin

The established variants above are recommended on the strength of
published results. AlphaGalerkin is our contribution and is stated as a
set of testable claims rather than a settled ranking:

- **Resolution generalization.** Train at low/medium resolution,
  evaluate zero-shot at 2–8× higher, and measure relative $L^2$ error
  vs. F-FNO, a multigrid-augmented operator, and a PDE foundation model
  (e.g., Poseidon/MPP/DPOT) on the same splits.
- **Curriculum benefit.** Ablate the curriculum (on vs. off, same
  compute budget) to isolate its contribution rather than attributing
  gains to the architecture as a whole.
- **Accuracy/compute trade-off.** Report error at matched inference
  FLOPs and wall-clock, since spectral adaptivity changes cost.
- **Benchmarks.** Standard PDE suites (Navier–Stokes, Darcy,
  diffusion–reaction) plus at least one stiff or multi-scale problem
  where curriculum transfer should matter most.

The claim of superiority holds only where these comparisons bear it
out; until then AlphaGalerkin is positioned as a promising synthesis,
not the strongest available method.

### 2.3 FNO Family: Future Directions (active research areas as of 2026)

- Foundation-model-style pretraining across many PDE families.
- Hybrid neural + classical-solver correctors.
- Uncertainty quantification via spectral-energy diagnostics or
  ensemble FNOs.
- Extension to space-time operators and stochastic PDEs.

The FNO family has matured from a single elegant idea in 2020 into a
diverse, production-ready toolkit. AlphaGalerkin attempts to synthesize
ideas from Fourier, Galerkin, and multi-scale operator learning;
whether that synthesis outperforms strong baselines is an empirical
question we aim to answer with the evaluations above.

---

## 3. DeepONet

**Reference.** Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E.
"Learning nonlinear operators via DeepONet based on the universal
approximation theorem of operators." *Nature Machine Intelligence*,
3, 218–229, 2021.

**What it does.** Branch-trunk network architecture that approximates
operators $G: u \mapsto G(u)$ between Banach spaces. Backed by the
Chen-Chen (1995) universal operator approximation theorem.

**What it cannot do.**

1. **Same supervised-data requirement as FNO.** Training pairs must
   come from an existing solver.
2. **Universal approximation is non-constructive** — it says "there
   exists a network of sufficient width" and does not bound width or
   convergence rate in terms of problem regularity, unlike classical
   $O(h^p)$ rates.
3. **Branch-trunk input must be fixed at training time;** transfer to
   different sensor layouts requires re-training the branch network.

**Complementarity.** DeepONet's operator-learning perspective is a
valuable conceptual antecedent. AlphaGalerkin targets a different
problem (building a discretization, not predicting a solution) but
shares the "resolution-independent" design goal (see CLAUDE.md
`Fourier Features for positional encoding` and `Monte Carlo integral
normalization`).

---

## 4. MeshGraphNets and Graph-Neural PDE Solvers

**Reference.** Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., &
Battaglia, P. W. "Learning Mesh-Based Simulation with Graph Networks."
*ICLR 2021*.

**What it does.** Treats a mesh as a graph; uses message-passing GNN
to predict next-step displacements. Includes a remesher network that
refines / coarsens locally. Validated on cloth, fluids, plastic
deformation.

**What it cannot do.**

1. **Learned time-stepping, not learned discretization.** MeshGraphNets
   rolls out the *dynamics*; it does not choose a discretization for a
   stationary BVP.
2. **No provable stability.** The learned time-stepper can violate
   conservation or become unstable outside training distribution.
3. **Remesher is greedy / one-step.** There is no look-ahead over
   remeshing sequences, and the remesher is trained via supervised
   imitation of a human-designed heuristic.
4. **Scales limited by GNN message-passing depth.**

**Complementarity.** MeshGraphNets and AlphaGalerkin solve different
problems (learned rollout vs. learned discretization). AlphaGalerkin's
state encoder could in principle be a GNN for unstructured meshes; the
`to_tensor` method in `MeshRefinementGame` currently assumes a
structured-grid encoding and is a known limitation.

---

## 5. Reinforcement Learning for Adaptive Mesh Refinement

**References.**

- Yang, J., Dzanic, T., Petersen, B., Kudo, J., Mittal, K., Tomov,
  V., Camier, J.-S., Zhao, T., Zha, H., Kolev, T., Anderson, R., &
  Faissol, D. M. "Reinforcement learning for adaptive mesh refinement."
  *Journal of Computational Physics*, 491, 112304, 2023.
- Foucart, C., Charous, A., & Lermusiaux, P. F. J. "Deep reinforcement
  learning for adaptive mesh refinement." *Journal of Computational
  Physics*, 491, 112381, 2023.

**What they do.** Cast AMR as an MDP and train a policy (typically
PPO or DQN) to choose which element to refine, rewarded by error
reduction per DOF. Yang et al. (2023) target time-dependent hyperbolic
PDEs on tree-structured meshes; Foucart et al. (2023) focus on
steady-state elliptic problems.

**What they cannot do that AlphaGalerkin can.**

1. **Model-free, myopic.** Both are policy-gradient / value-based
   methods without a learned world model. They do not do multi-step
   look-ahead planning at decision time.
2. **No tree search at inference.** AlphaGalerkin's MCTS performs
   hundreds of simulations per decision, quantitatively narrowing the
   policy error bound (Grill et al. 2020, see below).
3. **Single game mode.** Existing RL-AMR work handles h-refinement on
   a fixed basis. AlphaGalerkin handles basis enrichment, h, p, and
   hp-refinement within a unified game interface (`PDEGame`).
4. **No Galerkin stability guarantees.** Existing RL-AMR work does not
   enforce LBB (inf-sup) stability; AlphaGalerkin's `StabilityGuard`
   (see `theory.md`) does.

**Complementarity.** Yang et al. (2023) and Foucart et al. (2023) are
the *closest* prior art. AlphaGalerkin's contribution over them is:
(a) MCTS + learned policy/value prior (AlphaZero-style) instead of
model-free RL, and (b) unified treatment of basis and mesh spaces.
Any Phase-I benchmark suite must report head-to-head comparisons
against at least one of these baselines on a common problem.

**Verified novelty gap.** A literature search (as of 2026-04-23) finds
no published paper combining MCTS with Galerkin basis selection or
mesh refinement. The closest intersection — MCTS-for-numerical-methods —
is Silver et al.'s AlphaTensor (Nature 2022) for matrix multiplication
algorithms, which is structurally analogous but targets a different
numerical task.

---

## 6. Classical hp-Adaptive FEM

**References.**

- Babuška, I., & Guo, B. "The h, p and h-p version of the finite
  element method; basis theory and applications." *Advances in
  Engineering Software*, 15, 159–174, 1992.
- Melenk, J. M. "*hp-Finite Element Methods for Singular Perturbations*."
  Lecture Notes in Mathematics 1796, Springer, 2002.
- Demkowicz, L. *Computing with hp-Adaptive Finite Elements, Vol. 1.*
  Chapman & Hall/CRC, 2006.

**What it does.** hp-FEM combines local mesh refinement ($h$) with
local polynomial-degree enrichment ($p$) guided by a posteriori error
estimators. Babuška–Guo theory proves exponential convergence
$\|u - u_h\|_E \le C e^{-\beta \sqrt[d]{N}}$ for piecewise analytic
solutions with geometric mesh refinement near singularities.

**What it cannot do that AlphaGalerkin can.**

1. **Greedy marking.** Dörfler / maximum / fixed-fraction marking
   strategies are single-step optimal but provably suboptimal over
   horizons of $\ge 2$ steps — the textbook example is the re-entrant
   corner where optimal refinement sequences require refining "wrong"
   elements first to unlock geometric refinement near the singularity.
2. **Hand-engineered heuristics.** Choosing between $h$- and
   $p$-refinement at each element requires smoothness estimation
   (Mavriplis, Melenk-Wohlmuth indicators); these fail in heterogeneous
   regimes.
3. **Problem-specific tuning.** Exponential convergence constants
   depend on the regularity of the underlying solution, which is
   typically unknown a priori.

**Complementarity — the key claim.** AlphaGalerkin does **not** replace
hp-FEM. It *composes* classical hp-FEM primitives (h-refine, p-enrich
on a hierarchical basis) under a learned policy. The $O(h^p)$ (or
exponential, in the piecewise-analytic regime) convergence rate is
preserved because every action in $\mathcal{A}$ produces a valid
conforming (or suitably constrained non-conforming) finite-element
space. MCTS only decides *which* primitive to apply; it never produces
a non-FEM output. Consequently AlphaGalerkin inherits the Babuška–Guo
convergence theory and adds only the adversarial question of *policy
quality*, which is bounded by UCT regret (Kocsis–Szépesvári 2006;
Grill et al. 2020).

---

## 7. Summary Matrix

| Method | Learns solution? | Learns discretization? | Multi-step plan? | Classical convergence? | Training data? |
|---|:-:|:-:|:-:|:-:|:-:|
| PINN (Raissi 2019) | Yes | No | No | No | None |
| FNO (Li 2021) | Yes | No | No | No | Many $(f,u)$ pairs |
| DeepONet (Lu 2021) | Yes | No | No | No | Many $(f,u)$ pairs |
| MeshGraphNets (Pfaff 2021) | Rollout | Partially (remesher) | No | No | Simulation trajectories |
| RL-AMR (Yang 2023, Foucart 2023) | No | Yes (h only) | No (model-free) | If base FEM is | None |
| Classical hp-FEM (Babuška–Guo 1992) | No | Yes (heuristic) | Greedy only | Yes (exponential) | None |
| **AlphaGalerkin** | **No** | **Yes (h + p + basis)** | **Yes (MCTS)** | **Yes (inherited)** | **None** |

---

## 8. Honest Limitations Disclosure

- AlphaGalerkin is pre-TRL-5. Benchmarks so far cover 2D Poisson,
  Burgers, Navier-Stokes (Taylor-Green), and L-shaped Poisson.
  Extension to 3D unstructured is a Phase-I deliverable, not a
  current capability.
- MCTS compute overhead is real: each decision costs $O(N_\text{sim})$
  rollouts. On small problems this is larger than a classical solve.
  The win only materializes on problems where greedy marking produces
  > 2× the optimal DOF count — singular geometries and multi-scale
  forcing.
- We do not claim to outperform a well-tuned production hp-FEM code
  (deal.II, MFEM) on textbook problems. The claim is narrower:
  AlphaGalerkin produces better refinement sequences on
  problems where classical indicators are known to mis-rank elements.

---

## 9. Selected Additional Citations

- Kocsis, L., & Szépesvári, C. "Bandit based Monte-Carlo Planning."
  *ECML 2006*.
- Silver, D., Huang, A., et al. "Mastering the game of Go with deep
  neural networks and tree search." *Nature*, 529, 484–489, 2016.
- Silver, D., Hubert, T., Schrittwieser, J., et al. "A general
  reinforcement learning algorithm that masters chess, shogi, and Go
  through self-play." *Science*, 362, 1140–1144, 2018.
- Grill, J.-B., Altché, F., Tang, Y., Hubert, T., Valko, M.,
  Antonoglou, I., & Munos, R. "Monte-Carlo tree search as
  regularized policy optimization." *ICML 2020*.
- Fawzi, A., Balog, M., Huang, A., et al. (AlphaTensor).
  "Discovering faster matrix multiplication algorithms with
  reinforcement learning." *Nature*, 610, 47–53, 2022.
- Krishnapriyan, A. S., Gholami, A., Zhe, S., Kirby, R. M., &
  Mahoney, M. W. "Characterizing possible failure modes in
  physics-informed neural networks." *NeurIPS 2021*.
