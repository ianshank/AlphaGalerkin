"""Task-zero spike for the element-local AMR substrate (Gate 1.0).

Verifies, against the installed ``scikit-fem``, the four assumptions the
element-local substrate design rests on, then runs the decisive measurement:
does adaptive Dorfler marking beat uniform refinement at matched DOF?

On the *current* tensor-product substrate it does not -- adaptive is 5-9x
**worse** than uniform (``openspec/specs/project-charter/spec.md``), which is
why no marking-policy comparison on that substrate measures policy quality.
This spike establishes whether an element-local substrate inverts that.

Assumptions checked:

A. ``MeshTri.refined(elements)`` does not mutate its input.
B. ``MeshTri.refined(elements)`` is *local* and *conforming* (no hanging nodes).
C. ``skfem.Functional`` yields a mesh-independent quadrature L2 error, and it
   differs from the nodal RMS ``BaseSolver._compute_l2_error`` computes today.
D. ``basis.get_dofs()`` exposes Dirichlet DOFs on the installed version.

Run::

    pip install -e '.[dev,fem]'
    python -m scripts.spikes.skfem_substrate_spike

Geometry note (load-bearing): ``skfem.MeshTri.init_lshaped()`` is *already*
``[-1,1]^2`` minus the **first** quadrant with the reentrant corner at the
origin. Translating or scaling it moves the corner off the origin, which makes
the benchmark solution smooth on the domain and silently destroys the
experiment -- both arms then converge at the optimal rate and adaptive
refinement has nothing to gain.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Uniform refinement levels to sweep.
N_UNIFORM_LEVELS: int = 5
#: Adaptive refinement levels to sweep.
N_ADAPTIVE_LEVELS: int = 9
#: Uniform refinements applied to the coarse L-shape before the sweep starts.
INITIAL_REFINEMENTS: int = 2
#: Dorfler bulk-marking fraction for the adaptive arm.
MARKING_FRACTION: float = 0.5
#: Stop a sweep once the basis exceeds this many DOFs.
MAX_DOF: int = 12_000
#: Refinement level at which the solve/estimator profile is printed.
PROFILE_LEVEL: int = 3
#: Matched-DOF readings taken at this many log-spaced points.
N_MATCHED_READINGS: int = 5
#: Exponent of the reentrant-corner singular solution ``r^ALPHA``.
SINGULAR_EXPONENT: float = 2.0 / 3.0


def exact_solution(points: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Benchmark singular solution on ``init_lshaped()``'s orientation.

    ``u = r^(2/3) sin(2 (theta - pi/2) / 3)``, harmonic (``f = 0``) and zero on
    both reentrant edges. The branch cut sits inside the notch (the missing
    first quadrant), so ``theta`` runs over ``[pi/2, 2 pi]``.

    Args:
        points: Cartesian coordinates, shape ``(2, N)``.

    Returns:
        Solution values, shape ``(N,)``.

    """
    x = np.asarray(points[0], dtype=np.float64)
    y = np.asarray(points[1], dtype=np.float64)
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    theta = np.where(theta < np.pi / 2, theta + 2 * np.pi, theta)
    return radius**SINGULAR_EXPONENT * np.sin(2.0 * (theta - np.pi / 2) / 3.0)


def count_hanging_edges(mesh: Any) -> tuple[int, int]:
    """Return ``(n_edges_shared_by_more_than_two_elements, n_edges)``.

    A conforming triangulation has zero such edges.
    """
    counts: dict[tuple[int, int], int] = {}
    for element in range(mesh.t.shape[1]):
        v = mesh.t[:, element]
        for a, b in ((v[0], v[1]), (v[1], v[2]), (v[2], v[0])):
            key = (min(a, b), max(a, b))
            counts[key] = counts.get(key, 0) + 1
    return sum(1 for c in counts.values() if c > 2), len(counts)


def solve_poisson(mesh: Any, skfem: Any) -> tuple[Any, NDArray[np.float64]]:
    """Solve ``-Laplace u = 0`` with exact Dirichlet data on every boundary DOF."""
    from skfem.helpers import grad

    basis = skfem.Basis(mesh, skfem.ElementTriP1())

    @skfem.BilinearForm
    def stiffness(u: Any, v: Any, w: Any) -> Any:
        return sum(grad(u) * grad(v))

    @skfem.LinearForm
    def load(v: Any, w: Any) -> Any:
        return 0.0 * v

    matrix = stiffness.assemble(basis)
    rhs = load.assemble(basis)
    dirichlet = basis.get_dofs()
    solution = np.zeros(basis.N)
    interior = dirichlet.flatten()
    solution[interior] = exact_solution(basis.doflocs[:, interior])
    solved: NDArray[np.float64] = skfem.solve(*skfem.condense(matrix, rhs, x=solution, D=dirichlet))
    return basis, solved


def quadrature_l2(basis: Any, solution: NDArray[np.float64], skfem: Any) -> float:
    """Mesh-independent L2 error (assumption C)."""

    @skfem.Functional
    def squared_error(w: Any) -> Any:
        return (w["uh"] - exact_solution(w.x)) ** 2

    return float(np.sqrt(squared_error.assemble(basis, uh=basis.interpolate(solution))))


def nodal_rms(basis: Any, solution: NDArray[np.float64]) -> float:
    """The error metric ``BaseSolver._compute_l2_error`` computes today.

    A plain nodal RMS with no area weighting. On a graded mesh it over-weights
    the densely refined region -- the bias ``_area_weighted_l2`` exists to
    remove.
    """
    diff = solution - exact_solution(basis.doflocs)
    return float(np.sqrt(np.sum(diff**2) / len(diff)))


def zz_indicator(mesh: Any, solution: NDArray[np.float64], skfem: Any) -> NDArray[np.float64]:
    """Zienkiewicz-Zhu recovered-gradient error indicator, per element."""
    p1 = skfem.Basis(mesh, skfem.ElementTriP1())
    p0 = skfem.Basis(mesh, skfem.ElementTriP0())
    gradient = p1.interpolate(solution).grad
    raw_x, raw_y = p0.project(gradient[0]), p0.project(gradient[1])
    recovered_x, recovered_y = p1.project(p0.interpolate(raw_x)), p1.project(p0.interpolate(raw_y))

    @skfem.Functional
    def jump(w: Any) -> Any:
        return (w["rx"] - w["gx"]) ** 2 + (w["ry"] - w["gy"]) ** 2

    result: NDArray[np.float64] = jump.elemental(
        p1,
        rx=p1.interpolate(recovered_x),
        ry=p1.interpolate(recovered_y),
        gx=p0.interpolate(raw_x),
        gy=p0.interpolate(raw_y),
    )
    return result


def dorfler_mark(indicators: NDArray[np.float64], theta: float) -> NDArray[np.bool_]:
    """Smallest element set whose indicators reach ``theta`` of the total."""
    order = np.argsort(-indicators)
    cumulative = np.cumsum(indicators[order])
    n_marked = int(np.searchsorted(cumulative, theta * cumulative[-1])) + 1
    marked = np.zeros(len(indicators), dtype=bool)
    marked[order[:n_marked]] = True
    return marked


def check_assumptions(skfem: Any) -> None:
    """Assumptions A, B and D."""
    mesh = skfem.MeshTri.init_tensor(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    points_before, cells_before = mesh.p.tobytes(), mesh.t.tobytes()
    refined = mesh.refined(np.array([0, 1, 2]))

    print("A. refined() does not mutate its input:", end=" ")
    print(
        f"p={mesh.p.tobytes() == points_before} "
        f"t={mesh.t.tobytes() == cells_before} new_object={refined is not mesh}"
    )
    print(
        "A. CAVEAT: mesh.p.flags.writeable ="
        f" {mesh.p.flags.writeable} -- immutability is a property of the"
        " refinement API, not enforced by the array. The substrate must clear"
        " the write flag defensively."
    )

    full = mesh.refined()
    print(
        f"B. local refine(3 of {mesh.t.shape[1]}): "
        f"{mesh.t.shape[1]} -> {refined.t.shape[1]} elems "
        f"(global refine gives {full.t.shape[1]})"
    )
    hanging, n_edges = count_hanging_edges(refined)
    print(f"B. conforming after one local refine: {hanging}/{n_edges} bad edges")

    repeated = mesh
    for _ in range(4):
        centroids = repeated.p[:, repeated.t].mean(axis=1)
        worst = np.argsort(-np.linalg.norm(centroids, axis=0))
        repeated = repeated.refined(worst[: max(1, repeated.t.shape[1] // 8)])
    hanging, _ = count_hanging_edges(repeated)
    print(f"B. conforming after 4 local refines: {hanging} bad edges ({repeated.t.shape[1]} elems)")

    basis = skfem.Basis(mesh, skfem.ElementTriP1())
    print(
        f"D. get_dofs() -> {type(basis.get_dofs()).__name__}, "
        f"flatten()={basis.get_dofs().flatten().shape}, basis.N={basis.N}, "
        f"doflocs={basis.doflocs.shape}"
    )


def sweep(name: str, theta: float | None, skfem: Any) -> NDArray[np.float64]:
    """Run one refinement arm, returning rows of ``(n_dof, quad_l2, nodal_rms)``."""
    mesh = skfem.MeshTri.init_lshaped().refined(INITIAL_REFINEMENTS)
    levels = N_UNIFORM_LEVELS if theta is None else N_ADAPTIVE_LEVELS
    rows: list[tuple[float, float, float]] = []

    for level in range(levels):
        started = time.perf_counter()
        basis, solution = solve_poisson(mesh, skfem)
        solve_seconds = time.perf_counter() - started
        quad, rms = quadrature_l2(basis, solution, skfem), nodal_rms(basis, solution)
        rows.append((float(basis.N), quad, rms))
        print(f"{name:9s} {level:3d} {basis.N:7d} {quad:11.4e} {rms:11.4e} {rms / quad:7.3f}")
        if basis.N > MAX_DOF:
            break
        if theta is None:
            mesh = mesh.refined()
            continue
        started = time.perf_counter()
        indicators = zz_indicator(mesh, solution, skfem)
        estimator_seconds = time.perf_counter() - started
        if level == PROFILE_LEVEL:
            print(
                f"          [profile] solve={solve_seconds * 1e3:.1f}ms "
                f"zz={estimator_seconds * 1e3:.1f}ms "
                f"({mesh.t.shape[1]} elems) -> estimator is "
                f"{estimator_seconds / solve_seconds:.1f}x the solve"
            )
        mesh = mesh.refined(np.where(dorfler_mark(indicators, theta))[0])

    return np.array(rows, dtype=np.float64)


def main() -> None:
    """Run the assumption checks and the decisive matched-DOF comparison."""
    import skfem

    print(f"scikit-fem {skfem.__version__}\n")
    check_assumptions(skfem)

    print(
        f"\n{'arm':9s} {'lvl':>3s} {'ndof':>7s} {'quadL2':>11s} {'nodalRMS':>11s} {'rms/quad':>8s}"
    )
    curves = {
        "uniform": sweep("uniform", None, skfem),
        "adaptive": sweep("adaptive", MARKING_FRACTION, skfem),
    }

    uniform, adaptive = curves["uniform"], curves["adaptive"]
    lo = max(uniform[0, 0], adaptive[0, 0])
    hi = min(uniform[-1, 0], adaptive[-1, 0])
    print(f"\nmatched-DOF window: {lo:.0f} .. {hi:.0f}")
    print(f"{'DOF':>7s} {'uniform':>11s} {'adaptive':>11s} {'ratio a/u':>10s}")
    for dof in np.geomspace(lo, hi, N_MATCHED_READINGS):
        u = np.exp(np.interp(np.log(dof), np.log(uniform[:, 0]), np.log(uniform[:, 1])))
        a = np.exp(np.interp(np.log(dof), np.log(adaptive[:, 0]), np.log(adaptive[:, 1])))
        print(f"{dof:7.0f} {u:11.4e} {a:11.4e} {a / u:10.3f}")

    print()
    for label, curve in curves.items():
        slope = np.polyfit(np.log(curve[:, 0]), np.log(curve[:, 1]), 1)[0]
        print(f"rate {label:9s}: L2 ~ N^{slope:.3f}")


if __name__ == "__main__":
    main()
