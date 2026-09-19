"""Meshing for 2-D domains.

Two entry points:

``star_mesh``
    A structured polar mesh of a star-shaped domain given its radius function
    ``r(theta)``. High quality, no external mesher, and cheap to rebuild, which
    is what the inverse solver needs since it remeshes on every iteration.

``polygon_mesh``
    An unstructured mesh of an arbitrary simple polygon, via Delaunay
    triangulation with the triangles outside the polygon discarded. Used for
    shapes the user supplies directly.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from matplotlib.path import Path
from scipy.spatial import Delaunay, cKDTree
from skfem import MeshTri

__all__ = [
    "FourierShape",
    "is_simple_polygon",
    "polygon_mesh",
    "resample_closed_path",
    "star_mesh",
]


class FourierShape:
    """A star-shaped domain whose boundary radius is a truncated Fourier series.

    ``r(theta) = a0 + sum_k a_k cos(k theta) + b_k sin(k theta)``

    The inverse solver optimises over these coefficients rather than moving
    boundary nodes directly. It keeps the search space small, and every point
    in it is a valid non-self-intersecting shape as long as ``r`` stays
    positive, which :meth:`is_valid` checks.

    Parameters
    ----------
    a0
        Mean radius. Sets the overall scale.
    cos_coeffs
        Coefficients ``a_1 .. a_n`` of the cosine terms.
    sin_coeffs
        Coefficients ``b_1 .. b_n`` of the sine terms.
    """

    def __init__(self, a0: float, cos_coeffs=(), sin_coeffs=()):
        self.a0 = float(a0)
        self.cos_coeffs = np.asarray(cos_coeffs, dtype=float)
        self.sin_coeffs = np.asarray(sin_coeffs, dtype=float)

    @classmethod
    def from_vector(cls, v) -> FourierShape:
        """Unpack ``[a0, a_1..a_n, b_1..b_n]``, the form the optimiser uses."""
        v = np.asarray(v, dtype=float)
        if v.size % 2 != 1:
            raise ValueError(
                f"expected an odd number of coefficients (a0 plus matched "
                f"cosine and sine terms), got {v.size}"
            )
        n = (v.size - 1) // 2
        return cls(v[0], v[1 : 1 + n], v[1 + n :])

    def to_vector(self) -> np.ndarray:
        return np.concatenate([[self.a0], self.cos_coeffs, self.sin_coeffs])

    def radius(self, theta):
        theta = np.asarray(theta, dtype=float)
        r = np.full(theta.shape, self.a0)
        for k, a in enumerate(self.cos_coeffs, start=1):
            r = r + a * np.cos(k * theta)
        for k, b in enumerate(self.sin_coeffs, start=1):
            r = r + b * np.sin(k * theta)
        return r

    def is_valid(self, n_probe: int = 512, min_radius: float = 1e-3) -> bool:
        """True if the boundary stays a positive distance from the origin."""
        theta = np.linspace(0.0, 2.0 * np.pi, n_probe, endpoint=False)
        return bool(np.all(self.radius(theta) > min_radius))

    def boundary(self, n: int = 400) -> np.ndarray:
        """Boundary polyline, shape ``(2, n)``."""
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        r = self.radius(theta)
        return np.vstack([r * np.cos(theta), r * np.sin(theta)])

    def area(self, n: int = 4096) -> float:
        """Enclosed area, from the polar form ``0.5 * integral r^2 dtheta``."""
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return 0.5 * np.trapezoid(
            np.append(self.radius(theta) ** 2, self.radius(theta)[:1]),
            np.append(theta, 2.0 * np.pi),
        )

    def mesh(self, n_radial: int = 14, n_angular: int = 72) -> MeshTri:
        return star_mesh(self.radius, n_radial=n_radial, n_angular=n_angular)

    def __repr__(self) -> str:
        return (
            f"FourierShape(a0={self.a0:.4f}, "
            f"cos={np.round(self.cos_coeffs, 4).tolist()}, "
            f"sin={np.round(self.sin_coeffs, 4).tolist()})"
        )


def star_mesh(
    radius: Callable[[np.ndarray], np.ndarray],
    n_radial: int = 14,
    n_angular: int = 72,
) -> MeshTri:
    """Structured triangulation of the star-shaped domain ``r <= radius(theta)``.

    The unit disc is meshed in polar coordinates and then scaled outward by
    ``radius(theta)``. Radial positions follow ``sqrt`` spacing so that the
    triangles have roughly equal area instead of bunching at the centre.

    Parameters
    ----------
    radius
        Vectorised function of ``theta`` returning the boundary radius.
    n_radial
        Number of rings between the centre and the boundary.
    n_angular
        Number of points around each ring.
    """
    if n_radial < 2 or n_angular < 3:
        raise ValueError("need n_radial >= 2 and n_angular >= 3")

    theta = np.linspace(0.0, 2.0 * np.pi, n_angular, endpoint=False)
    r_boundary = np.asarray(radius(theta), dtype=float)
    if np.any(r_boundary <= 0.0):
        raise ValueError("radius must be positive everywhere")

    # sqrt spacing equalises ring areas
    fractions = np.sqrt(np.linspace(0.0, 1.0, n_radial + 1))[1:]

    points = [np.zeros((2, 1))]
    for f in fractions:
        r = f * r_boundary
        points.append(np.vstack([r * np.cos(theta), r * np.sin(theta)]))
    p = np.hstack(points)

    triangles = []
    # centre fan: ring 1 indices are 1 .. n_angular
    for j in range(n_angular):
        triangles.append([0, 1 + j, 1 + (j + 1) % n_angular])
    # quad strips between consecutive rings, split into two triangles
    for i in range(n_radial - 1):
        inner = 1 + i * n_angular
        outer = 1 + (i + 1) * n_angular
        for j in range(n_angular):
            jn = (j + 1) % n_angular
            triangles.append([inner + j, outer + j, outer + jn])
            triangles.append([inner + j, outer + jn, inner + jn])

    return MeshTri(p, np.array(triangles, dtype=np.int64).T)


def polygon_mesh(vertices, max_area: float | None = None) -> MeshTri:
    """Unstructured triangulation of a simple polygon.

    Boundary vertices are resampled to a uniform spacing, interior points are
    laid on a staggered grid, the union is triangulated with Delaunay, and the
    triangles whose centroid falls outside the polygon are dropped. That last
    step is what makes the result respect a non-convex boundary.

    Parameters
    ----------
    vertices
        Polygon vertices, shape ``(2, n)`` or ``(n, 2)``, in order and without
        repeating the first point.
    max_area
        Target triangle area. Defaults to the polygon area divided by 1500.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2:
        raise ValueError("vertices must be 2-dimensional")
    if v.shape[0] != 2:
        v = v.T
    if v.shape[0] != 2 or v.shape[1] < 3:
        raise ValueError("need at least 3 vertices shaped (2, n) or (n, 2)")

    x, y = v
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if abs(signed_area) < 1e-14:
        raise ValueError("polygon has zero area")
    if signed_area < 0.0:  # normalise to counter-clockwise
        v = v[:, ::-1]
        x, y = v

    poly_area = abs(signed_area)
    if max_area is None:
        max_area = poly_area / 1500.0
    h = np.sqrt(4.0 * max_area / np.sqrt(3.0))  # side of an equilateral triangle

    closed = np.hstack([v, v[:, :1]])
    seg = np.diff(closed, axis=1)
    seg_len = np.hypot(seg[0], seg[1])

    boundary = []
    for i in range(v.shape[1]):
        n_sub = max(1, int(np.ceil(seg_len[i] / h)))
        t = np.linspace(0.0, 1.0, n_sub, endpoint=False)
        boundary.append(closed[:, i : i + 1] + np.outer(seg[:, i], t))
    boundary = np.hstack(boundary)

    path = Path(np.vstack([closed[0], closed[1]]).T)
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()
    nx = max(2, int(np.ceil((x1 - x0) / h)))
    ny = max(2, int(np.ceil((y1 - y0) / (h * np.sqrt(3.0) / 2.0))))

    rows = []
    for j in range(ny + 1):
        yy = y0 + j * (h * np.sqrt(3.0) / 2.0)
        offset = 0.5 * h if j % 2 else 0.0
        xx = x0 + offset + np.arange(nx + 1) * h
        rows.append(np.vstack([xx, np.full_like(xx, yy)]))
    grid = np.hstack(rows)

    # keep interior points that are comfortably inside, so they do not crowd
    # the resampled boundary
    inside = path.contains_points(grid.T, radius=-0.5 * h)
    interior = grid[:, inside]

    # Drop interior points that sit almost on top of a boundary point.
    # contains_points with a negative radius is not reliable enough on its own:
    # at fine resolutions a grid point landing within rounding distance of the
    # boundary produced exactly-zero-area triangles. Linear elements absorb
    # those silently, but any element that inverts a per-element matrix, such
    # as Morley or Argyris for plate problems, fails outright on them.
    if interior.shape[1]:
        too_close = cKDTree(boundary.T).query(interior.T)[0] < 0.55 * h
        interior = interior[:, ~too_close]

    p = np.hstack([boundary, interior])
    tri = Delaunay(p.T)

    centroids = p[:, tri.simplices].mean(axis=2)
    keep = path.contains_points(centroids.T)
    t = tri.simplices[keep].T

    # Delaunay can still emit slivers along a concave boundary. Measure and
    # discard them rather than hand a degenerate element to the solver.
    if t.shape[1]:
        tx, ty = p[0, t], p[1, t]
        twice_area = np.abs(
            (tx[1] - tx[0]) * (ty[2] - ty[0]) - (tx[2] - tx[0]) * (ty[1] - ty[0])
        )
        t = t[:, twice_area > 1e-9 * h * h]

    if t.shape[1] == 0:
        raise ValueError("triangulation produced no interior elements")

    # drop points no triangle references, and reindex
    used = np.unique(t)
    remap = np.full(p.shape[1], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return MeshTri(p[:, used], remap[t])


def resample_closed_path(points, n: int = 160):
    """Even out a freehand path and close it.

    A mouse drags out points clumped wherever the hand slowed down. Delaunay
    copes badly with that, so the path is walked at constant arclength.

    Parameters
    ----------
    points
        Shape ``(2, m)`` or ``(m, 2)``. The path is treated as closed whether
        or not the last point repeats the first.
    n
        Number of output vertices.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2:
        raise ValueError("points must be 2-dimensional")
    if p.shape[0] != 2:
        p = p.T
    if p.shape[1] < 3:
        raise ValueError("need at least 3 points")

    # drop a repeated closing point, then drop consecutive duplicates
    if np.allclose(p[:, 0], p[:, -1]):
        p = p[:, :-1]
    keep = np.hypot(*np.diff(np.hstack([p, p[:, :1]]), axis=1)) > 1e-12
    p = p[:, keep]
    if p.shape[1] < 3:
        raise ValueError("path collapses to fewer than 3 distinct points")

    closed = np.hstack([p, p[:, :1]])
    steps = np.hypot(*np.diff(closed, axis=1))
    distance = np.concatenate([[0.0], np.cumsum(steps)])
    if distance[-1] <= 0:
        raise ValueError("path has zero length")

    wanted = np.linspace(0.0, distance[-1], n, endpoint=False)
    return np.vstack(
        [np.interp(wanted, distance, closed[0]), np.interp(wanted, distance, closed[1])]
    )


def is_simple_polygon(points) -> bool:
    """True if no two non-adjacent edges of the closed path cross.

    A figure-eight has no well defined interior, so meshing it would produce
    nonsense rather than an error. This catches it first.
    """
    p = np.asarray(points, dtype=float)
    if p.shape[0] != 2:
        p = p.T
    n = p.shape[1]
    if n < 3:
        return False

    def crosses(a, b, c, d):
        def side(u, v, w):
            return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])

        d1, d2 = side(c, d, a), side(c, d, b)
        d3, d4 = side(a, b, c), side(a, b, d)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    for i in range(n):
        a, b = p[:, i], p[:, (i + 1) % n]
        # start two edges along, and stop before wrapping onto edge i again
        for j in range(i + 2, n - 1 if i == 0 else n):
            c, d = p[:, j % n], p[:, (j + 1) % n]
            if crosses(a, b, c, d):
                return False
    return True
