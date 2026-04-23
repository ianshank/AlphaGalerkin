# MDP Specification: MCTS over PDE Discretization

**Solicitation:** DE-FOA-0003612 (DOE Genesis Mission, Phase I)
**Scope:** Formal Markov Decision Process underlying AlphaGalerkin's
MCTS-guided basis selection and adaptive mesh refinement.
**Source of truth:** `src/pde/game.py`, `src/pde/games/basis_selection.py`,
`src/pde/games/mesh_refinement.py`, `src/pde/mcts_adapter.py`.

---

## 1. Motivation

Classical adaptive FEM uses greedy, single-step error indicators (Dörfler
marking, ZZ estimator). These are provably suboptimal when the
error-reducing action *k* steps ahead differs from the action with the
largest immediate indicator — a frequent pattern near corner singularities
and shocks. AlphaGalerkin reformulates discretization as a sequential
decision problem, which admits multi-step look-ahead via MCTS.

---

## 2. MDP Tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma, T)$

The two concrete games — `BasisSelectionGame` and `MeshRefinementGame` —
share the abstract `PDEGame` interface (`src/pde/game.py:245`).

### 2.1 State Space $\mathcal{S}$

State is represented by the dataclass `PDEState`
(`src/pde/game.py:47`). Every state contains:

| Field | Type | Meaning |
|---|---|---|
| `coords` | `float32 [N, d]` | Collocation / mesh-element centroid coordinates |
| `solution` | `float32 [N]` | Current approximation $u_h$ at those points |
| `residuals` | `float32 [N]` | Pointwise PDE residual $r = L u_h - f$ |
| `basis_coefficients` | `float32 [k]` \| `None` | Galerkin coefficients (basis game) |
| `mesh_levels` | `int32 [E]` \| `None` | Per-element refinement level (mesh game) |
| `polynomial_degrees` | `int32 [E]` \| `None` | Per-element $p$ (mesh game) |
| `error_estimate` | `float` | Current (exact or residual-based) L2 error |
| `dof` | `int` | Total degrees of freedom |
| `step` | `int` | Number of actions taken |
| `budget_remaining` | `float` | Remaining compute budget |
| `phase` | `GamePhase` | `INITIAL` / `EXPLORING` / `REFINING` / `CONVERGED` / `BUDGET_EXHAUSTED` |
| `history` | `list[int]` | Action sequence from $s_0$ |

The state includes sufficient statistics for the reward function
(`error_estimate`, `dof`, `budget_remaining`) and carries the full
action `history`, preserving the Markov property with respect to the
current discretization regardless of path (since the approximation is a
deterministic function of the action set for `BasisSelectionGame` and of
the mesh topology for `MeshRefinementGame`).

**State space cardinality.** Finite but combinatorially large:

- **BasisSelectionGame:** $|\mathcal{S}| \le \binom{N_\text{cand}}{\le k_\text{max}}$
  where $N_\text{cand} =$ `basis_config.n_candidate_bases` and
  $k_\text{max} =$ `basis_config.max_basis_functions`.
- **MeshRefinementGame:** $|\mathcal{S}| \le (2^d)^{L_\text{max}}$ times
  the number of element selections per step, where $L_\text{max} =$
  `mesh_config.max_refinement_level` and $d$ is the spatial dimension.

### 2.2 Action Space $\mathcal{A}$

Following the user's specification, the conceptual action alphabet is
$\{\text{h-refine},\ \text{p-increase},\ \text{coarsen},\ \text{terminate}\}$,
composed with an `element_id` or `basis_id`. The current code exposes a
subset of this alphabet per game mode:

#### 2.2.1 BasisSelectionGame (`src/pde/games/basis_selection.py:97`)

- $\mathcal{A}(s) =$ indices in $[0, N_\text{cand})$ not already in
  `state.history` (see `get_valid_actions`, line 300).
- Semantics: "enrich the trial space with candidate basis
  $\phi_a$" — conceptually equivalent to **p-increase** when the
  candidate set is hierarchic (Fourier, monomial, hierarchical RBF).
- Masking: `get_action_mask` (line 320) zeros already-selected bases and
  zeros all actions once `n_basis >= max_basis_functions`.
- Terminate action is **implicit**: MCTS selects no action when
  `is_terminal(state)` is true (line 480).

#### 2.2.2 MeshRefinementGame (`src/pde/games/mesh_refinement.py:321`)

- $\mathcal{A}(s) =$ leaf-element indices $i \in [0, |\text{leaves}|)$
  satisfying all of:
  - `element.level < max_refinement_level`
  - `element.size > min_element_size`
  - `element.polynomial_degree < max_polynomial_degree`

  (see `get_valid_actions`, line 425).
- Refinement semantics determined by
  `mesh_config.refinement_strategy`
  (`RefinementStrategy` enum in `src/pde/config.py`):
  - `H_REFINEMENT`: subdivide selected element into $2^d$ children
    (`Mesh._subdivide_element`, line 244).
  - `P_REFINEMENT`: increment `polynomial_degree` on selected element
    (line 224).
  - `HP_REFINEMENT`: h below level 2, then p (`refine_element`, line 233).
- **Coarsen** is not currently exposed as a primitive action; it is
  reachable only via `get_initial_state` re-roll. Adding coarsening is
  tracked as a Phase-I deliverable.
- **Terminate** is implicit via `is_terminal` (line 608).

### 2.3 Transition Kernel $P(s' \mid s, a)$

**Deterministic.** Given the current state and action, $s'$ is the
unique output of `apply_action` (basis: line 342; mesh: line 467). There
is no stochasticity in the environment — the only randomness in a full
MCTS rollout comes from the tree-search policy itself and any random
basis-candidate generation that happens at construction time
(controlled by `basis_config.seed`).

Pseudocode of the basis-selection transition (`apply_action`):

```
s' = s.clone(); s'.history.append(a); s'.step += 1
Φ = build_basis_matrix(selected_bases, coords)     # (N × k)
c = lstsq(Φ, target)                               # Galerkin coefficients
s'.solution = Φ @ c
s'.residuals = L(s'.solution) - f
s'.error_estimate = ‖s'.solution - u_exact‖_2      # if exact known
s'.dof = len(history);   s'.budget_remaining -= 1
```

The mesh-refinement transition invokes `Mesh.refine_element`,
regenerates element centroids, interpolates the previous solution
(`_interpolate_solution` — nearest-neighbor, line 540) to the new
centroids, and recomputes residuals.

### 2.4 Reward $R(s, a, s')$

Implemented reward (basis: `get_reward`, line 453; mesh: line 563) is:

$$
R(s, a, s') = \alpha_1 \cdot \Delta e - \alpha_2 \cdot \Delta n_\text{dof}
              + \mathbb{1}[\text{converged}] \cdot B
              + \underbrace{\mathbb{1}[\text{mesh}]\cdot\mathbb{1}[\eta > \tau] \cdot \mu (\eta - \tau)}_\text{mesh efficiency bonus}
$$

where $\Delta e = e(s) - e(s')$, $\Delta n_\text{dof} = n_\text{dof}(s') - n_\text{dof}(s)$,
$\eta = \Delta e / \Delta n_\text{dof}$ is per-DOF efficiency, and
$\alpha_1 =$ `config.reward_per_error_reduction`,
$\alpha_2 =$ `config.cost_per_dof`, $B =$ `config.terminal_bonus`,
$\tau =$ `mesh_config.efficiency_threshold`,
$\mu =$ `mesh_config.efficiency_multiplier`.

**Proposal-format reward.** The DOE proposal text uses the canonical
log-cost form
$$R(s,a,s') = -\alpha \cdot \log L_2(s') - \beta \cdot \log C(s'),$$
which is mathematically equivalent up to an affine reparameterization
on each episode (additive constants and the monotonicity of $-\log$
preserve the ordering of policies under any discount factor). A
`LogRewardAdapter` will be added to `PDEGameAdapter` in Phase I so that
the code-level reward matches the proposal form exactly for
reproducibility of reviewer-run experiments.

### 2.5 Discount $\gamma$

We use $\gamma = 1$ (undiscounted, bounded horizon). Justification:
horizon $T$ is finite and small (see §2.6); the reward is bounded; and
the cumulative return equals the final quality-vs-cost Pareto coordinate,
which is the quantity of physical interest.

### 2.6 Horizon $T$ (Bounded)

Termination (`is_terminal`):

- `error_estimate < config.error_tolerance`  → success
- `step >= config.max_steps`                  → step cap
- `budget_remaining <= 0`                     → budget
- basis game: `n_basis >= max_basis_functions`
- mesh game:  `dof > config.max_dof`
- no legal actions remain

Hence $T \le \min(\text{max\_steps},\ \text{max\_basis},\
\lceil\text{budget}\rceil)$, which is finite and configuration-specified.
Bounded horizon is a prerequisite for the UCT convergence result
(Kocsis & Szépesvári 2006) invoked in `theory.md`.

### 2.7 Well-Defined Reward

The reward is (i) measurable (every component is a computable function
of `PDEState` fields), (ii) bounded above and below on every finite
trajectory (error estimate is bounded below by 0 and above by the
initial error; DOF and budget are bounded), and (iii) zero-mean-free
(error-reduction terms can be negative, preventing degenerate positive
loops that would invalidate UCT regret bounds).

### 2.8 MCTS Coupling

`PDEGameAdapter` (`src/pde/mcts_adapter.py:47`) bridges the MDP above to
the board-game-style protocol consumed by `src/mcts/search.py::MCTS`.
Notable mappings:

- `get_state()`  → `pde_game.to_tensor(state).cpu().numpy()`
- `get_legal_actions()` → `pde_game.get_valid_actions(state)`
- `apply_action(a)` → mutates `self.state = pde_game.apply_action(...)` and
  appends to `self.error_history`
- `get_winner()` → $\{-1, 0, +1\}$ based on convergence and relative
  error reduction (line 142). This is the value target $v \in [-1, 1]$
  consumed by the PUCT prior.

---

## 3. Sanity Checks (Empirical Claims for Phase I)

### 3.1 Random-Policy Baseline

**Claim to demonstrate in Phase I:** a uniform-random policy over
$\mathcal{A}(s)$ produces strictly positive expected return on
$R^d$-domain Poisson and Burgers operators with smooth forcing.
Rationale: every action that adds a non-zero basis function or refines
any leaf element reduces `residual_norm` in expectation, because the
Galerkin least-squares projection is a contraction in the residual norm.
The proposal will report mean-return confidence intervals from
$\ge 1{,}000$ random rollouts per operator.

**Why this matters for reviewers.** A positive random-policy baseline
with tight variance confirms (i) the reward is well-scaled, (ii) there
are no degenerate positive loops, and (iii) MCTS improvement over
random is measurable with modest sample complexity.

### 3.2 Bounded-Return Check

On every terminal state, the absolute return $|G_T| \le
\alpha_1 \cdot e_0 + \alpha_2 \cdot n_\text{dof,max} + B + \mu \cdot
e_0$, which the test suite (`tests/pde/test_operators.py` and
forthcoming `tests/pde/test_mdp_properties.py`) will assert on a
Hypothesis-generated set of configurations.

### 3.3 Markov Property

The state fields collectively determine the next transition; `history`
is retained for trajectory logging and augmentation, but
`apply_action(state, a)` depends only on `state`. This is verified by
the `clone()` method (line 112): two independent clones that receive
the same action sequence produce identical terminal states modulo
floating-point determinism.

### 3.4 Reward-Monotonicity Spot Check

On any converged episode, the cumulative return must exceed the
cumulative return of the zero-action trajectory. This will be asserted
as a regression test against the current reward coefficients.

---

## 4. Open Design Items (Phase I Deliverables)

- Add explicit **coarsen** action to `MeshRefinementGame` so the action
  alphabet matches the proposal's $\{h, p, \text{coarsen}, \text{terminate}\}$.
- Introduce `LogRewardAdapter` for exact parity with the
  $-\alpha \log L_2 - \beta \log C$ reward used in the proposal narrative.
- Replace nearest-neighbor solution interpolation
  (`_interpolate_solution`, `mesh_refinement.py:540`) with a proper
  $L^2$-projection onto the refined space to restore consistency rates
  in mixed h/p regimes.
- Tighten `get_winner` thresholds (`mcts_adapter.py:142`) into
  Pydantic-validated configuration rather than hard-coded `0.1` / `0.5`.

---

## 5. Summary Table

| MDP element | Symbol | Code location |
|---|---|---|
| State | $\mathcal{S}$ | `PDEState`, `src/pde/game.py:47` |
| Action (basis) | $\mathcal{A}(s)$ | `BasisSelectionGame.get_valid_actions`, line 300 |
| Action (mesh) | $\mathcal{A}(s)$ | `MeshRefinementGame.get_valid_actions`, line 425 |
| Transition | $P$ (deterministic) | `apply_action` (both games) |
| Reward | $R$ | `get_reward` (both games) |
| Terminal predicate | $T$ | `is_terminal` (both games) |
| MCTS bridge | — | `PDEGameAdapter`, `src/pde/mcts_adapter.py:47` |
