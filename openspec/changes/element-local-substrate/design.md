# Design: `element-local-substrate`

## The interface

```python
@dataclass(frozen=True)
class SubstrateSolveResult:
    values: NDArray[np.float64]
    indicators: NDArray[np.float64]   # FLAT, len == n_units(mesh)
    l2_error: float                   # mesh-independent (quadrature)
    n_dof: int                        # the declared comparison axis
    n_dof_free: int                   # unknowns actually solved for
    extra: Mapping[str, float]        # e.g. l2_error_nodal_rms

class RefinementSubstrate(Protocol[TMesh]):
    def initial_mesh(self) -> TMesh: ...
    def solve(self, mesh: TMesh) -> SubstrateSolveResult: ...
    def mark(self, indicators: NDArray, theta: float) -> NDArray[np.bool_]: ...
    def refine(self, mesh: TMesh, marked: NDArray[np.bool_]) -> TMesh: ...
    def n_units(self, mesh: TMesh) -> int: ...
    def refinable_mask(self, mesh: TMesh) -> NDArray[np.bool_]: ...
    def fingerprint(self, mesh: TMesh) -> bytes: ...
    def describe(self) -> dict[str, str | int | float]: ...
```

Every member has a real caller. CI's `audit_abstractions` step fails on a `Protocol` member
with no reader — the F1 defect class — so an eight-member protocol with seven consumers would
be rejected by the build, correctly.

## Placement, and why it matters

`src/refinement/substrate.py` imports **numpy only** — no scipy, no torch, no skfem. This is
not tidiness: `src/pde/games/__init__.py:11-19` documents a real SIGSEGV caused by rippling the
torch import graph into unrelated coverage gates under the C tracer. Keeping the domain-free
layer import-light is what makes it safe to depend on from anywhere.

The concrete substrates live in `src/research/substrates/`, and `skfem_tri.py` goes in
`pyproject.toml`'s coverage `omit` — the same treatment `fem_baseline.py` already gets, since
its tests skip without the optional extra and including it would unfairly tank the global gate.

## Three decisions worth arguing about

**1. Two error metrics, not one.** `BaseSolver._compute_l2_error` is a plain nodal RMS with no
area weighting — the exact bias `_area_weighted_l2` was written to remove. On a graded mesh it
over-weights the densely refined region. The spike measured the consequence: `nodalRMS /
quadratureL2` drifts 0.34→0.53 for the uniform arm but 0.34→0.76 for the adaptive one. Reusing
it would flatter whichever arm refines hardest — a *fabricated* result, not a measurement
error. So the substrate reports a quadrature L2 and carries the nodal RMS as an auxiliary
field, leaving `ScikitFEMPoissonSolver.solve()`'s asserted outputs untouched.

**2. `skfem` meshes are treated as immutable, with a guard.** `refined()` returns a new object
and does not mutate its input — but `mesh.p.flags.writeable` is `True`, so nothing *enforces*
it. The cheap-`clone()` design (share meshes by reference; MCTS clones the game per simulation)
depends on immutability holding, so the substrate clears the write flags and a property test
pins it. A `deep_copy` escape hatch exists for the case where that assumption breaks.

**3. Purity is a contract, and the existing game breaks it.**
`RefinementGame.apply_action` is specified as *"a pure function of `(state, action)` … must not
mutate `state`"*, because that is what lets MCTS identify a node by its action sequence.
`LShapeAMRGame.apply_action` mutates the game instance (`self._xs = self._bisect_edge(...)`),
with its docstring conceding the state argument is kept "for interface parity". Honouring the
contract removes a class of clone bugs *and* enables the prefix-keyed mesh cache — which is
what makes the per-simulation cost tractable.

## Cost model, corrected by measurement

The expected bottleneck was mesh cloning inside MCTS's per-simulation replay. The spike says
otherwise: `solve = 2.5 ms`, `ZZ estimator = 6.2 ms` — and that is a *vectorised* projection,
whereas `fem_baseline._compute_zz_indicator` is a Python loop over elements with an inner loop
over vertices and a 3×3 `np.linalg.solve` per element. The estimator dominates.

Sequencing follows from that: capture the `fem_baseline` decomposition goldens **first**, then
vectorise as a separate change with an explicit tolerance, because vectorisation perturbs
floating-point results and would otherwise collide with the bitwise back-compat discipline.

## Skip discipline

`scikit-fem` is optional, so its tests must skip on CPU CI — but **visibly**. A registered
`fem_required` marker plus a root-conftest hook (mirroring `gpu_required`) reports how many were
skipped, and `ALPHAGALERKIN_REQUIRE_EXTRAS=1` turns a missing extra into a hard collection error
inside `test-extras`.

This closes a gap in that job's own stated purpose. Its comment says it exists so optional-dep
suites "RUN … instead of masking those paths as 'passing'" — but if the install half-succeeds,
`pytest.importorskip` still skips silently and the job still goes green. `test-extras` is also
absent from `ci-success`'s `needs`, so it could not block a merge even if it did go red.
