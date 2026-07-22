# Mathematical Foundations — Theory Sketch

**Solicitation:** DE-FOA-0003612 (DOE Genesis Mission, Phase I)
**Audience:** DOE ASCR applied-mathematics reviewers.
**Purpose:** Establish the theoretical scaffolding of AlphaGalerkin.
This is a sketch of argument structure with citations; full proofs are
deferred to an accompanying technical appendix and a planned journal
submission.

---

## 1. Problem Setting

Let $\Omega \subset \mathbb{R}^d$ be a bounded Lipschitz domain and
consider the abstract variational problem: find $u \in V$ such that
$$
a(u, v) = \ell(v) \qquad \forall v \in V,
$$
where $V$ is a Hilbert space, $a : V \times V \to \mathbb{R}$ is a
continuous bilinear form (bounded by $M$), and $\ell : V \to \mathbb{R}$
is continuous. The canonical examples handled by
`src/pde/operators.py` are the Poisson, Burgers, advection-diffusion,
heat, and (steady) Navier-Stokes operators.

---

## 2. Galerkin Projection and LBB (inf-sup) Stability

### 2.1 Petrov–Galerkin Setup

Choose finite-dimensional trial and test spaces $V_h \subset V$ and
$W_h \subset V$ with $\dim V_h = \dim W_h = N_h$. The
Petrov–Galerkin approximation $u_h \in V_h$ solves
$$
a(u_h, w_h) = \ell(w_h) \qquad \forall w_h \in W_h.
$$

### 2.2 Ladyzhenskaya–Babuška–Brezzi (LBB) Condition

Well-posedness of the discrete problem and quasi-optimality of $u_h$
require the discrete **inf-sup condition**: there exists $\beta > 0$
independent of $h$ such that
$$
\inf_{v_h \in V_h \setminus \{0\}}\ \sup_{w_h \in W_h \setminus \{0\}}
\frac{a(v_h, w_h)}{\|v_h\|_V\ \|w_h\|_V}\ \ge\ \beta.
$$

**Consequence (Babuška 1971; Brezzi 1974).** If the LBB condition
holds with constant $\beta$ and $a$ has continuity constant $M$, then
$$
\|u - u_h\|_V \le \left(1 + \frac{M}{\beta}\right)\, \inf_{v_h \in V_h} \|u - v_h\|_V.
$$

That is, the Galerkin error is within a constant factor of the best
approximation error in $V_h$. The constant depends on $M/\beta$, so
stability degradation inflates the constant.

### 2.3 StabilityGuard and LBB Monitoring in AlphaGalerkin

CLAUDE.md records the architectural decision *"LBB Stability:
dim(Key) >= dim(Query) to satisfy inf-sup condition"*, which is the
neural-operator analog of the classical LBB requirement: the test
(Key) basis must span at least as much as the trial (Query) basis.
`StabilityGuard` computes the smallest singular value of the
Key-to-Value projection on each training batch; the training loss
includes an LBB regularizer (`src/training/losses/`) that penalizes
$\sigma_\min(K) < \beta_0$. This is a numerical *surrogate* for the
inf-sup condition: on finite-dimensional operators the discrete
inf-sup constant equals the smallest singular value of the Gram-type
operator induced by $a$.

**Claim (to be demonstrated in Phase I).** With the LBB regularizer
active, the empirical $M/\beta$ stays bounded over the course of
training and the quasi-optimality constant of (2.2) remains uniformly
bounded along the MCTS-induced sequence of refinements.

---

## 3. Fredholm Integral Formulation

Per CLAUDE.md (*Chosen Kernel: Fredholm integral equation with
Green's function formulation*), AlphaGalerkin's attention
mechanism is derived from the Fredholm second-kind equation
$$
u(x) = f(x) + \int_\Omega G(x, y)\, u(y)\, dy,
$$
where $G$ is the Green's function for the differential operator of
interest. The Galerkin-attention kernel $K(x,y) = Q(x)\,K(y)^\top$
realizes a low-rank approximation of $G$.

**Monte Carlo normalization.** The CLAUDE.md decision *"Monte Carlo
integral normalization (1/n) for Galerkin attention"* replaces the
softmax's exponential normalization with $1/n$, so that the
projection $Q(K^\top V)/n$ is an unbiased Monte Carlo estimator of the
integral $\int_\Omega Q(x) K(y) V(y)\,d\mu(y)$ sampled uniformly at
$n$ collocation points. This produces the $O(N)$ complexity
advertised in CLAUDE.md and is central to the zero-shot
resolution-transfer result (measured MSE ≈ 0.00039 on 19×19 from 9×9 training,
CLAUDE.md milestone of 2026-01-26).

**Fourier-feature positional encoding.** The choice of Fourier features
(CLAUDE.md: *Basis function selection: Fourier Features for positional
encoding*) makes the encoder equivariant under translations on the
torus and gives an explicit spectral representation of the Green's
function on periodic domains — the same structural backbone as FNO
but used only for *state encoding*, not for solution prediction.

---

## 4. MCTS Convergence — UCT and PUCT Regularization

### 4.1 UCT (Kocsis–Szépesvári)

**Reference.** Kocsis, L., & Szépesvári, C. "Bandit based Monte-Carlo
Planning." *ECML 2006*.

For an MDP with bounded rewards $R \in [-R_\max, R_\max]$ and finite
horizon $T$, the UCT algorithm — applying UCB1 recursively at each
tree node — achieves *consistency*: the value of the root node's
recommended action converges to the optimal value as the number of
simulations $N \to \infty$. The simple-regret bound is
$$
\mathbb{E}[\text{regret}] = O\!\left(\frac{\log N}{N}\right)
$$
at the root (under mild non-degeneracy conditions on action-value
gaps).

**Applicability to AlphaGalerkin.** The MDP in `mdp_specification.md`
satisfies the preconditions: finite horizon (§2.6), bounded reward
(§2.7), deterministic transitions (§2.3). Hence UCT consistency
applies to AlphaGalerkin's basis-selection and mesh-refinement games.

### 4.2 PUCT under Regularization (Grill et al.)

**Reference.** Grill, J.-B., Altché, F., Tang, Y., Hubert, T.,
Valko, M., Antonoglou, I., & Munos, R.
"Monte-Carlo tree search as regularized policy optimization."
*ICML 2020*.

AlphaZero-style MCTS uses PUCT, which incorporates a learned prior
$\pi_\theta$:
$$
a^\star = \arg\max_a \left[ Q(s,a) + c_\text{puct}\,\pi_\theta(a \mid s)\,\frac{\sqrt{N(s)}}{1 + N(s,a)} \right].
$$
Grill et al. show that PUCT is equivalent to approximate regularized
policy iteration against a KL-penalty to $\pi_\theta$. Consequently,
when the prior is accurate, PUCT achieves strictly better sample
complexity than UCT; when the prior is uniform, PUCT reduces to UCT
(up to constants) and retains the UCT consistency guarantee.

**Implication for AlphaGalerkin.** Our MCTS implementation
(`src/mcts/gumbel.py` — Gumbel AlphaZero; `src/mcts/search.py` —
PUCT) inherits both guarantees:

- Even with an *untrained* network (uniform prior), MCTS converges to
  the optimal policy by UCT.
- A well-trained prior accelerates this convergence; training data
  comes from self-play, which requires no external labels.

---

## 5. Inherited Classical Convergence Rates

### 5.1 Key Structural Property

Every action in $\mathcal{A}$ produces a *conforming* (or suitably
constrained non-conforming) finite-element space. That is:

- `BasisSelectionGame` actions add basis functions to a Galerkin trial
  space $V_h^{(k)} \subset V_h^{(k+1)} \subset V$; the enriched space
  remains a valid Galerkin space and the discrete problem (2.1) is
  well-posed under the LBB guard of §2.3.
- `MeshRefinementGame` actions are local h-subdivision or
  p-enrichment on hypercube elements; the resulting mesh / polynomial
  space is a standard hp-FEM space (Demkowicz 2006, §3).

### 5.2 Classical Convergence Rates Survive

For a $P_p$-Lagrange finite element space on a quasi-uniform mesh of
size $h$, the standard a priori estimate (Ciarlet 1978; Ern &
Guermond 2004) gives
$$
\|u - u_h\|_{H^1(\Omega)} \le C\, h^p\, |u|_{H^{p+1}(\Omega)}
$$
for $u \in H^{p+1}(\Omega)$. For an hp-FEM space with geometric
refinement towards corner singularities (Babuška–Guo 1992; Melenk
2002), the error decays exponentially:
$$
\|u - u_h\|_E \le C\, e^{-b\sqrt[d]{N}}, \qquad N = \dim V_h.
$$

**The inheritance argument (sketch).** Because every MCTS-selected
action produces a valid $V_h \subset V$, the finite-element error at
*any* node of the MCTS tree is governed by the classical a priori or
a posteriori bounds applied to that specific $V_h$. MCTS influences
only *which* $V_h$ is selected, not the analytical bound that $V_h$
admits.

Therefore:

1. **Lower bound.** AlphaGalerkin's error is no worse than the
   asymptotic rate of the worst $V_h$ reachable by MCTS, which for
   $P_p$-Lagrange is $O(h^p)$.
2. **Upper bound (practical).** Since MCTS searches over refinement
   sequences and is guided by a value function that estimates
   error-per-DOF, the selected $V_h$ should achieve near-optimal
   error-DOF tradeoff among those reachable. Quantifying the gap to
   the oracle $V_h$ reduces to the UCT / PUCT regret bound of §4.

**This is the central theoretical contribution of the project:**
*A method that inherits classical finite-element error bounds while
producing DOF-efficient discretizations without hand-crafted
heuristics.* Unlike PINN/FNO/DeepONet, we do **not** need a new
approximation theory — we reuse the Babuška–Brezzi–Ciarlet corpus.

### 5.3 Dörfler-Marking Comparison Baseline

Dörfler (1996) established that a greedy marking strategy with bulk
parameter $\theta \in (0, 1)$ produces a sequence of meshes for which
the error contracts at a problem-dependent rate. Binev-Dahmen-DeVore
(2004) and Stevenson (2007) proved optimal convergence rates for
adaptive FEM with Dörfler marking (matching the best $N$-term
approximation rate). AlphaGalerkin's baseline (`DorflerAMRSolver`,
CLAUDE.md milestone 2026-04-02) is precisely this class of method;
the Phase-I benchmark reports the MCTS-vs-Dörfler gap on a suite of
operators.

---

## 6. Stability, Well-Posedness, and Assumptions

Full argument requires:

- $a$ continuous and LBB-stable on $V_h \times W_h$ for every $V_h,
  W_h$ reachable by MCTS. The LBB regularizer (§2.3) is the practical
  enforcement mechanism.
- Rewards bounded on every trajectory (verified in
  `mdp_specification.md` §2.7).
- Finite horizon (§2.6).
- Uniqueness of optimal action at each node up to the gap parameter
  required by UCT consistency.

We do not claim a formal end-to-end theorem in this document. What we
claim is that each ingredient (Galerkin stability, bounded MDP, UCT
regret, hp-FEM approximation) has a peer-reviewed theoretical home,
and the composition is mechanically sound. A formal theorem combining
these ingredients is a Phase-I paper deliverable.

---

## 7. Citations

- Babuška, I. "Error-bounds for finite element method." *Numer. Math.*,
  16, 322–333, 1971.
- Babuška, I., & Guo, B. "The h, p and h-p version of the finite
  element method; basis theory and applications." *Adv. Eng. Softw.*,
  15, 159–174, 1992.
- Binev, P., Dahmen, W., & DeVore, R. "Adaptive finite element methods
  with convergence rates." *Numer. Math.*, 97, 219–268, 2004.
- Brezzi, F. "On the existence, uniqueness and approximation of
  saddle-point problems arising from Lagrangian multipliers."
  *RAIRO Anal. Numér.*, 8, 129–151, 1974.
- Ciarlet, P. G. *The Finite Element Method for Elliptic Problems.*
  North-Holland, 1978.
- Demkowicz, L. *Computing with hp-Adaptive Finite Elements, Vol. 1.*
  Chapman & Hall/CRC, 2006.
- Dörfler, W. "A convergent adaptive algorithm for Poisson's
  equation." *SIAM J. Numer. Anal.*, 33, 1106–1124, 1996.
- Ern, A., & Guermond, J.-L. *Theory and Practice of Finite
  Elements.* Springer, 2004.
- Grill, J.-B., Altché, F., Tang, Y., Hubert, T., Valko, M.,
  Antonoglou, I., & Munos, R. "Monte-Carlo tree search as
  regularized policy optimization." *ICML 2020*.
- Kocsis, L., & Szépesvári, C. "Bandit based Monte-Carlo Planning."
  *ECML 2006*.
- Melenk, J. M. *hp-Finite Element Methods for Singular
  Perturbations.* Lecture Notes in Mathematics 1796, Springer, 2002.
- Stevenson, R. "Optimality of a standard adaptive finite element
  method." *Found. Comput. Math.*, 7, 245–269, 2007.
- Cao, S. "Choose a Transformer: Fourier or Galerkin." *NeurIPS 2021*.
  (Provides the Galerkin-attention formulation used in our encoder.)
