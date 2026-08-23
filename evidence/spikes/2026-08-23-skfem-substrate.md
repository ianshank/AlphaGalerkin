# Spike: element-local AMR substrate (Gate 1.0, task zero)

> **Date:** 2026-08-23 · **Reproduce:** `pip install -e '.[dev,fem]'` then
> `python -m scripts.spikes.skfem_substrate_spike`
> **Environment:** scikit-fem **12.0.2**, numpy 2.4.6, scipy 1.17.1, Python 3.11

## Why this spike exists

The project's central claim — MCTS multi-step look-ahead beats classical adaptive
refinement — cannot be measured on the current substrate. The charter records why:
`DorflerAMRSolver._dorfler_mark_2d` (`src/research/baselines.py:969`) projects element marks
onto the x and y axes, so marking one element inserts full grid *lines*, and adaptive Dörfler
ends up **5–9× worse than plain uniform refinement at matched DOF**. Comparing two marking
policies on that substrate measures the substrate.

`src/research/fem_baseline.py` already contains a genuine element-local FEM solver that nothing
in the repository consumes. This spike establishes, before any of it is wired in, whether the
four assumptions the substrate design rests on actually hold — and whether an element-local
substrate inverts the adaptive-vs-uniform outcome.

## Result 1 — the four assumptions hold

| | Assumption | Verdict |
|---|---|---|
| **A** | `MeshTri.refined(elements)` does not mutate its input | **Holds** — `p`/`t` bytes unchanged, new object returned. **Caveat:** `mesh.p.flags.writeable` is `True`, so immutability is a property of the refinement *API*, not enforced by the array. The substrate must clear the write flag defensively, and a property test must pin it. |
| **B** | `refined(elements)` is local and conforming | **Holds** — refining 3 of 32 elements gives 49 elements (global refinement gives 128). **Zero** hanging edges after one local refine and after four successive local refines. Genuine conforming RGB. |
| **C** | `skfem.Functional` gives a mesh-independent quadrature L2 | **Holds** — see Result 3. |
| **D** | `basis.get_dofs()` exposes Dirichlet DOFs | **Holds** — returns a `DofsView`; `.flatten()` gives the boundary DOF indices. |

`tests/research/test_fem_baseline.py` — never previously executed in this environment, since it
opens with a module-level `pytest.importorskip("skfem")` — **passes 23/23 in 0.84 s** on
scikit-fem 12.0.2 against a `>=9.0` pin. No API drift. This was the plan's single largest
open risk and it is retired.

## Result 2 — the substrate inverts the outcome (the decisive measurement)

Uniform vs adaptive-Dörfler (θ = 0.5, ZZ recovered-gradient estimator) on the standard L-shaped
Poisson benchmark, `u = r^(2/3) sin(2(θ−π/2)/3)`, P1 elements, exact Dirichlet data:

| Substrate | Convergence rate | Adaptive vs uniform @ matched DOF |
|---|---|---|
| Tensor grid (current, committed) | — | **5–9× worse**, gap widening |
| **skfem element-local (this spike)** | uniform `L2 ~ N^-0.710`, adaptive `L2 ~ N^-1.256` | **4–10× better**, gap widening |

Matched-DOF readings (ratio = adaptive / uniform; below 1 means adaptive wins):

| DOF | uniform L2 | adaptive L2 | ratio |
|---|---|---|---|
| 164 | 1.022e-2 | 2.898e-3 | **0.284** |
| 413 | 5.221e-3 | 8.659e-4 | **0.166** |
| 1041 | 2.705e-3 | 3.645e-4 | **0.135** |
| 2625 | 1.419e-3 | 1.360e-4 | **0.096** |

The rates are the textbook AFEM result and are the real check: uniform refinement is
rate-limited by the `r^(2/3)` corner singularity at `N^-0.71` (theory: `N^-2/3 = N^-0.667`),
while element-local adaptive refinement recovers the optimal P1 rate at `N^-1.26`. **Gate 1's
exit criterion is achievable**, and the separation is a rate separation over a DOF range rather
than a single favourable point.

## Result 3 — the nodal-RMS landmine is real and measurable

`BaseSolver._compute_l2_error` (`src/research/baselines.py:238-253`) computes a plain nodal RMS,
`sqrt(sum(diff²)/n)`, with no area weighting — the exact bias `_area_weighted_l2` was written to
remove. This spike computes both metrics on every mesh. The ratio `nodalRMS / quadratureL2` is
**not a constant**, and it drifts *differently per arm*:

| arm | ratio at coarsest | ratio at finest |
|---|---|---|
| uniform | 0.341 | 0.528 |
| adaptive | 0.341 | 0.760 |

Because the adaptive arm clusters nodes at the singularity, its RMS drifts further. Using the
nodal RMS as the comparison metric would therefore **flatter whichever arm refines hardest** —
a fabricated result, not a measurement error. The substrate must report the quadrature L2 and
retain the nodal RMS only as an auxiliary field, so `ScikitFEMPoissonSolver.solve()`'s existing
asserted outputs stay unchanged.

## Result 4 — the estimator, not mesh cloning, is the cost to watch

Profiled at refinement level 3 (264 elements): `solve = 2.5 ms`, `ZZ estimator = 6.2 ms` —
**the estimator is ~2.5× the solve**, using a vectorised skfem projection. The repo's own
`_compute_zz_indicator` (`fem_baseline.py:360-375`) is a Python loop over elements with an
inner loop over vertices, and `_element_gradients` (`:385-395`) does a 3×3 `np.linalg.solve`
per element in a Python loop — so it will be materially slower again. Inside MCTS, where the
estimator runs once per solve at ~100 solves per refinement step, this dominates. Budget the
vectorisation, and capture the `fem_baseline` decomposition goldens **before** vectorising,
because vectorisation perturbs floating-point results.

## Result 5 — a geometry trap that silently destroys the experiment

`skfem.MeshTri.init_lshaped()` is **already** `[-1,1]²` minus the **first** quadrant with the
reentrant corner at the origin. An earlier iteration of this spike applied
`.translated([-1,-1]).scaled([2,2])`, which moves the domain to `[-4,0]²` and puts the origin
outside it. The benchmark solution is then smooth on the domain, both arms converge at the
optimal `N^-1.05`, and **adaptive refinement appears to lose (ratio ≈ 1.3–1.4)** — a confident,
entirely wrong "adaptive doesn't help on skfem either" result.

It was caught only because a `N^-1.05` uniform rate is impossible for a singular problem. The
substrate's initial-mesh construction needs an explicit assertion that the reentrant corner is
a mesh node at the origin, and the convergence gate needs the uniform-arm rate band as a
tripwire — a substrate that converges too *well* is as diagnostic as one that diverges.

## What this changes in the plan

- **Gate 1 proceeds.** The substrate choice is validated end-to-end, not assumed.
- **Gate 1.0's skfem-API risk is closed** (23/23 green on 12.0.2, no drift). Pin `>=9.0,<13`.
- **Assumption A needs a defensive guard**, not just a comment: clear the array write flag and
  pin it with a property test.
- **The two-error design is mandatory**, and now has a measured justification rather than an
  argued one.
- **Add a geometry assertion** to the substrate's initial mesh, and a uniform-rate tripwire to
  the convergence gate.
- **Gate 1.4's budget must include estimator vectorisation**, sequenced after golden capture.
