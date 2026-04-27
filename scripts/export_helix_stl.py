"""Export the Noyron HX helical-tube geometry as a binary STL.

Writes the same helix that ``AnalyticalHelixSDF`` represents (the geometry
the headline ``noyron_hx`` scenario trains against) directly from its
parametric description: the centerline is a vertical helix and the
cross-section is a circle of radius ``r_minor``. We sweep the circle
along the centerline using a parallel-transport frame, then emit a
triangle strip mesh and serialize as binary STL.

This avoids the marching-cubes-on-SDF dependency chain (scikit-image)
that my offer originally suggested; the resulting mesh is functionally
identical for visualization purposes since it is the analytical SDF's
zero-level set, and the parametric construction is cleaner (no grid
artifacts, exact normals).

Defaults match ``config/scenarios/noyron_hx.yaml`` so the produced
``outputs/poc/noyron_hx/noyron_hx.stl`` represents the geometry the
headline GPU run was trained against.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def _build_vertices(
    R_major: float,
    r_minor: float,
    pitch: float,
    n_turns: int,
    n_t: int,
    n_theta: int,
) -> np.ndarray:
    """Sweep a circle of radius ``r_minor`` along a vertical helix.

    Returns vertices of shape ``(n_t+1, n_theta, 3)`` — the +1 in the
    centerline direction is to close the tube along its length without
    duplicating the seam ring.
    """
    # Centerline parameter (open interval is fine: tube is open-ended).
    t = np.linspace(0.0, 2.0 * np.pi * n_turns, n_t + 1)
    cx = R_major * np.cos(t)
    cy = R_major * np.sin(t)
    cz = pitch * t / (2.0 * np.pi)
    centerline = np.stack([cx, cy, cz], axis=-1)  # (n_t+1, 3)

    # Tangent along centerline (analytical derivative).
    tx = -R_major * np.sin(t)
    ty = R_major * np.cos(t)
    tz = np.full_like(t, pitch / (2.0 * np.pi))
    tangent = np.stack([tx, ty, tz], axis=-1)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)

    # Build a parallel-transport frame: pick an initial normal, then
    # rotate it minimally as the tangent changes. Avoids the twist that
    # arises if you naively cross-product against world-up.
    # Initial normal: any vector orthogonal to tangent[0].
    t0 = tangent[0]
    arbitrary = np.array([0.0, 0.0, 1.0]) if abs(t0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n0 = arbitrary - np.dot(arbitrary, t0) * t0
    n0 /= np.linalg.norm(n0)

    normals = np.empty_like(tangent)
    normals[0] = n0
    for i in range(1, len(tangent)):
        # Rotate previous normal to be perpendicular to current tangent.
        # Project out the new tangent component, renormalize.
        n_prev = normals[i - 1]
        t_curr = tangent[i]
        n_curr = n_prev - np.dot(n_prev, t_curr) * t_curr
        norm = np.linalg.norm(n_curr)
        if norm < 1e-12:
            # Degenerate; fall back to arbitrary perpendicular.
            arbitrary = np.array([0.0, 0.0, 1.0]) if abs(t_curr[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            n_curr = arbitrary - np.dot(arbitrary, t_curr) * t_curr
            norm = np.linalg.norm(n_curr)
        normals[i] = n_curr / norm

    binormals = np.cross(tangent, normals)  # already unit since both unit & orthogonal

    # Cross-section: circle of radius r_minor at angles theta.
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    # Tube vertices: c(t) + r_minor * (cos(theta) * normal + sin(theta) * binormal).
    # Shape: (n_t+1, n_theta, 3)
    verts = (
        centerline[:, None, :]
        + r_minor * (cos_th[None, :, None] * normals[:, None, :]
                     + sin_th[None, :, None] * binormals[:, None, :])
    )
    return verts


def _build_triangles(n_t: int, n_theta: int) -> np.ndarray:
    """Index the (n_t+1, n_theta) vertex grid into triangles.

    Each quad on the tube splits into two triangles. Theta is periodic so
    the seam wraps. The tube ends are left open (matches the analytical
    SDF, which is also unbounded along the tube length within the bbox).
    """
    triangles = []
    for i in range(n_t):
        for j in range(n_theta):
            j_next = (j + 1) % n_theta
            v00 = i * n_theta + j
            v01 = i * n_theta + j_next
            v10 = (i + 1) * n_theta + j
            v11 = (i + 1) * n_theta + j_next
            # CCW winding so normals point outward.
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))
    return np.asarray(triangles, dtype=np.int64)


def _write_binary_stl(path: Path, verts: np.ndarray, tris: np.ndarray) -> None:
    """Serialize triangle list as binary STL.

    Format: 80-byte header, uint32 triangle count, then per-triangle
    {float32[3] normal, float32[3] v0, float32[3] v1, float32[3] v2,
     uint16 attribute byte count}.
    """
    flat = verts.reshape(-1, 3).astype(np.float32, copy=False)
    n_tris = len(tris)
    with open(path, "wb") as f:
        f.write(b"AlphaGalerkin Noyron HX helix tube".ljust(80, b" "))
        f.write(struct.pack("<I", n_tris))
        for a, b, c in tris:
            v0 = flat[a]
            v1 = flat[b]
            v2 = flat[c]
            n = np.cross(v1 - v0, v2 - v0)
            nn = np.linalg.norm(n)
            if nn > 0.0:
                n = n / nn
            f.write(struct.pack("<3f", *n.astype(np.float32)))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Noyron HX helix as STL")
    parser.add_argument("--R-major", type=float, default=0.05, help="Helix radius (m)")
    parser.add_argument("--r-minor", type=float, default=0.012, help="Tube cross-section radius (m)")
    parser.add_argument("--pitch", type=float, default=0.02, help="Vertical rise per turn (m)")
    parser.add_argument("--n-turns", type=int, default=5, help="Number of helical revolutions")
    parser.add_argument(
        "--n-t",
        type=int,
        default=400,
        help="Centerline samples (total across all turns; ~80/turn at default)",
    )
    parser.add_argument(
        "--n-theta",
        type=int,
        default=64,
        help="Cross-section samples per ring",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/poc/noyron_hx/noyron_hx.stl"),
        help="Output STL path",
    )
    args = parser.parse_args()

    if args.r_minor >= args.R_major:
        raise SystemExit("r_minor must be < R_major (else the tube self-intersects)")

    verts = _build_vertices(
        R_major=args.R_major,
        r_minor=args.r_minor,
        pitch=args.pitch,
        n_turns=args.n_turns,
        n_t=args.n_t,
        n_theta=args.n_theta,
    )
    tris = _build_triangles(n_t=args.n_t, n_theta=args.n_theta)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_binary_stl(args.output, verts, tris)

    n_verts = verts.size // 3
    bbox_min = verts.reshape(-1, 3).min(axis=0)
    bbox_max = verts.reshape(-1, 3).max(axis=0)
    print(f"Wrote {args.output}")
    print(f"  vertices : {n_verts:,}")
    print(f"  triangles: {len(tris):,}")
    print(f"  bbox min : ({bbox_min[0]:+.4f}, {bbox_min[1]:+.4f}, {bbox_min[2]:+.4f}) m")
    print(f"  bbox max : ({bbox_max[0]:+.4f}, {bbox_max[1]:+.4f}, {bbox_max[2]:+.4f}) m")
    print(f"  size     : {args.output.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
