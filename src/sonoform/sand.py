"""How sand finds the nodal set.

The obvious model is wrong. Grains do not roll downhill toward the nodes, and
there is no average force pushing them there at all. Abramian, Protiere,
Lazarus and Devauchelle, Phys. Rev. Research 7, L032001 (2025), show the
mechanism is transport: a grain sitting where the plate moves is thrown up
repeatedly and lands somewhere random, while a grain on a quiet patch stays
put. Averaged over many bounces that is a random walk whose diffusivity
follows the local amplitude,

``dx = sqrt(2 D(x) dt) xi``,    ``D(x) = D_floor + scale * |A(x)|``

with no drift term whatsoever. The steady state of that walk is

``rho(x) proportional to 1 / D(x)``

so the grains accumulate exactly where the plate is still, which is the nodal
set. The pattern is a consequence of the plate being quiet there, not of
anything pushing the sand.

Two things must not be changed without reading the test file. ``D`` is sampled
at the pre-step position, which is the Ito convention and is what produces
``rho ~ 1/D``. Adding the ``grad(D)`` term that much of the SDE literature
calls a spurious-drift correction switches the result to anti-Ito, whose steady
state is uniform: the plate comes out blank, with no crash and no warning.
``tests/test_sand_transport.py`` pins both.

The nodal set is also drawn on its own, by marching the zero contour of ``A``
across the triangles, so the sand can be checked against the contour rather
than standing in for it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["diffusivity", "nodal_segments", "sand_step"]

# Defaults for a mode normalised to |A| <= 1. The floor keeps grains on a node
# from freezing solid, which reads as dead pixels rather than sand.
DIFFUSION_FLOOR = 0.02
DIFFUSION_SCALE = 1.0


def diffusivity(
    amplitude,
    floor: float = DIFFUSION_FLOOR,
    scale: float = DIFFUSION_SCALE,
):
    """Local diffusivity of the bouncing grains.

    Low where the plate is still, high where it moves. The shape of this
    function sets how sharp the lines come out: the steady density is its
    reciprocal, so a larger ``scale`` relative to ``floor`` gives thinner,
    higher-contrast figures, which is also what driving a real plate harder
    does.
    """
    return floor + scale * np.abs(np.asarray(amplitude, dtype=float))


def sand_step(
    positions,
    amplitude,
    gradient=None,
    diffusion: float = 1.0,
    dt: float = 1.0,
    rng=None,
    contains=None,
    floor: float = DIFFUSION_FLOOR,
):
    """Advance grains one step and keep them on the plate.

    Parameters
    ----------
    positions
        Shape ``(2, n)``.
    amplitude
        Mode amplitude at each grain, shape ``(n,)``. Only its magnitude
        matters, since a grain does not care which way the plate is bent.
    gradient
        Accepted and ignored, and kept in the signature to make the omission
        explicit rather than invisible. The transport law has no drift term,
        so no gradient is needed. See ``test_sand_step_has_no_drift_term``.
    diffusion
        Overall diffusivity scale, multiplying :func:`diffusivity`.
    dt
        Time step.
    rng
        NumPy random generator.
    contains
        ``contains(points) -> bool array`` for points shaped ``(2, m)``.
        Points on the boundary count as inside.
    floor
        Diffusivity where the plate is perfectly still.
    """
    positions = np.asarray(positions, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if diffusion < 0.0:
        raise ValueError("diffusion must be non-negative")
    if rng is None:
        rng = np.random.default_rng()

    if diffusion == 0.0 or dt == 0.0:
        return positions.copy()

    # D sampled here, at the position the grain is leaving. See the module
    # docstring: moving this sample is the silent failure.
    local = diffusion * diffusivity(amplitude, floor=floor, scale=DIFFUSION_SCALE)
    sigma = np.sqrt(2.0 * local * dt)
    proposed = positions + sigma * rng.standard_normal(positions.shape)

    if contains is None:
        return proposed
    return _reflect(positions, proposed, contains)


def nodal_segments(points, triangles, values) -> np.ndarray:
    """Zero contour of a vertex field, one segment per crossed triangle.

    Returns an array of shape ``(n, 2, 2)``. Item ``s`` is the segment
    ``[[x0, y0], [x1, y1]]``. A triangle whose vertex values do not change
    sign contributes nothing; a zero that lands exactly on a vertex is left
    to the neighbouring edges, which still cross.
    """
    pts = np.asarray(points, dtype=float)
    tris = np.asarray(triangles)
    vals = np.asarray(values, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be 2-dimensional")
    if pts.shape[0] != 2:
        pts = pts.T
    if tris.ndim != 2:
        raise ValueError("triangles must be 2-dimensional")
    if tris.shape[0] != 3:
        tris = tris.T

    segments = []
    for tri in tris.T:
        hits = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            va = float(vals[tri[a]])
            vb = float(vals[tri[b]])
            if va * vb < 0.0:
                weight = va / (va - vb)
                hit = (1.0 - weight) * pts[:, tri[a]] + weight * pts[:, tri[b]]
                hits.append(hit)
        if len(hits) >= 2:
            segments.append([hits[0].tolist(), hits[1].tolist()])
    if not segments:
        return np.zeros((0, 2, 2))
    return np.asarray(segments, dtype=float)


def _reflect(start, proposed, contains):
    """Bounce steps that leave the plate back in along the same line."""
    proposed = np.array(proposed, dtype=float, copy=True)
    inside = np.asarray(contains(proposed), dtype=bool)
    if inside.shape != (proposed.shape[1],):
        raise ValueError("contains must return one boolean per point")
    if np.all(inside):
        return proposed

    for i in np.flatnonzero(~inside):
        origin = start[:, i]
        step = proposed[:, i] - origin
        if not np.any(contains(origin.reshape(2, 1))):
            proposed[:, i] = origin
            continue
        lo, hi = 0.0, 1.0
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            point = (origin + mid * step).reshape(2, 1)
            if bool(np.asarray(contains(point)).reshape(-1)[0]):
                lo = mid
            else:
                hi = mid
        hit = origin + lo * step
        bounced = hit - (1.0 - lo) * step
        if bool(np.asarray(contains(bounced.reshape(2, 1))).reshape(-1)[0]):
            proposed[:, i] = bounced
        else:
            proposed[:, i] = hit
    return proposed
