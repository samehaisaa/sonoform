"""The plates the app offers.

Single source of truth for the preset outlines, so the interactive app and the
published site cannot drift apart. Nothing here is special: every preset is a
closed outline meshed at the same physical span, and only the square keeps its
structured mesh, because it is the one checked against Leissa's table.

Each plate carries one line of text for the page. Those lines make claims, so
they are kept to what the symmetry of the outline guarantees.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PRESETS", "preset_points", "preset_request"]

# The outlines live on a unit radius and are scaled to this fraction of the
# span when they are meshed.
_SCALE = 0.09


def _ring(n: int, radius) -> list[list[float]]:
    theta = 2.0 * np.pi * np.arange(n) / n
    r = radius(theta)
    return np.vstack([r * np.cos(theta), r * np.sin(theta)]).T.tolist()


def _regular(sides: int, floor: float):
    """Radius of a regular polygon inscribed in the unit circle."""
    wedge = 2.0 * np.pi / sides

    def radius(theta):
        folded = np.mod(theta, wedge) - 0.5 * wedge
        return 1.0 / np.maximum(floor, np.cos(folded))

    return radius


def _triangle(theta):
    wedge = 2.0 * np.pi / 3.0
    folded = np.mod(theta + 0.5 * np.pi, wedge) - np.pi / 3.0
    return 1.0 / np.maximum(0.35, np.cos(folded))


def _ellipse(theta, aspect: float = 0.62):
    return 1.0 / np.hypot(np.cos(theta), np.sin(theta) / aspect)


def _petals(n: int = 160, lobes: int = 5, depth: float = 0.2):
    """A smooth five-fold rose, one petal pointing up."""
    theta = 2.0 * np.pi * np.arange(n) / n
    r = (1.0 + depth * np.cos(lobes * theta)) / (1.0 + depth)
    angle = theta + 0.5 * np.pi
    return np.vstack([r * np.cos(angle), r * np.sin(angle)]).T.tolist()


def _stadium(n: int = 160, straight: float = 1.0):
    """Bunimovich's stadium: two half discs joined by straight sides.

    Sampled at constant arclength, with the straight part as long as the
    diameter, the proportion Bunimovich used.
    """
    radius = 1.0
    half = straight * radius
    perimeter = 4.0 * half + 2.0 * np.pi * radius
    points = []
    for d in perimeter * np.arange(n) / n:
        if d < 2.0 * half:
            points.append((-half + d, -radius))
            continue
        d -= 2.0 * half
        if d < np.pi * radius:
            angle = -0.5 * np.pi + d / radius
            points.append((half + radius * np.cos(angle), radius * np.sin(angle)))
            continue
        d -= np.pi * radius
        if d < 2.0 * half:
            points.append((half - d, radius))
            continue
        d -= 2.0 * half
        angle = 0.5 * np.pi + d / radius
        points.append((-half + radius * np.cos(angle), radius * np.sin(angle)))
    out = np.asarray(points) / (half + radius)
    return out.tolist()


def _catmull_rom(control, n: int) -> np.ndarray:
    """Closed centripetal Catmull-Rom spline, resampled at constant arclength.

    Centripetal parametrisation cannot form cusps or self-intersections
    between control points, which is what keeps a hand-placed outline simple.
    """
    ctrl = np.asarray(control, dtype=float)
    m = len(ctrl)
    dense = []
    for i in range(m):
        p0, p1, p2, p3 = (ctrl[(i + j) % m] for j in (-1, 0, 1, 2))
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5
        for t in np.linspace(t1, t2, 64, endpoint=False):
            a1 = ((t1 - t) * p0 + (t - t0) * p1) / (t1 - t0)
            a2 = ((t2 - t) * p1 + (t - t1) * p2) / (t2 - t1)
            a3 = ((t3 - t) * p2 + (t - t2) * p3) / (t3 - t2)
            b1 = ((t2 - t) * a1 + (t - t0) * a2) / (t2 - t0)
            b2 = ((t3 - t) * a2 + (t - t1) * a3) / (t3 - t1)
            dense.append(((t2 - t) * b1 + (t - t1) * b2) / (t2 - t1))
    dense = np.asarray(dense)
    closed = np.vstack([dense, dense[:1]])
    step = np.hypot(*np.diff(closed, axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(step)])
    wanted = np.linspace(0.0, along[-1], n, endpoint=False)
    return np.vstack(
        [np.interp(wanted, along, closed[:, 0]), np.interp(wanted, along, closed[:, 1])]
    ).T


def _mirrored(right_half, n: int = 160) -> list[list[float]]:
    """Close a right half traced bottom to top into a symmetric outline.

    The result is centred on its own vertex mean and fitted to the unit
    circle's height, so it meshes like the other presets.
    """
    right = np.asarray(right_half, dtype=float)
    left = right[-2:0:-1] * np.array([-1.0, 1.0])
    points = _catmull_rom(np.vstack([right, left]), n)
    points -= points.mean(axis=0)
    points /= np.abs(points).max()
    return points.tolist()


# Traced from the proportions of a full-size violin: 356 mm long, 208 mm
# across the lower bout, 112 mm at the waist and 168 mm across the upper bout,
# with the four corners that close the C-bouts.
_VIOLIN = [
    (0.0, -1.0), (0.22, -0.975), (0.40, -0.895), (0.52, -0.77), (0.58, -0.62),
    (0.59, -0.47), (0.568, -0.35), (0.535, -0.275), (0.50, -0.225),
    (0.475, -0.21), (0.41, -0.18), (0.355, -0.10), (0.332, 0.0), (0.34, 0.09),
    (0.378, 0.16), (0.435, 0.198), (0.458, 0.205), (0.48, 0.25), (0.492, 0.35),
    (0.492, 0.48), (0.466, 0.61), (0.41, 0.74), (0.32, 0.86), (0.19, 0.95),
    (0.0, 0.985),
]

# A classical guitar body: 490 mm long, 370 mm across the lower bout, 235 mm
# at the waist and 285 mm across the upper bout.
_GUITAR = [
    (0.0, -1.0), (0.30, -0.972), (0.54, -0.875), (0.695, -0.715), (0.755, -0.50),
    (0.735, -0.28), (0.655, -0.08), (0.54, 0.09), (0.49, 0.20), (0.505, 0.31),
    (0.565, 0.45), (0.585, 0.59), (0.555, 0.74), (0.455, 0.875), (0.26, 0.962),
    (0.0, 0.985),
]


# label and a line for the page, then either "square" or the outline
PRESETS: dict[str, dict] = {
    "square": {
        "label": "square",
        "shape": "square",
        "note": "The classic. Leissa tabulated its notes in 1969, and this "
        "solver is checked against his table.",
    },
    "circle": {
        "label": "circle",
        "points": _ring(160, np.ones_like),
        "note": "Most of its notes belong to two figures at once, so where "
        "you bow decides which way the lines turn.",
    },
    "ellipse": {
        "label": "ellipse",
        "points": _ring(160, _ellipse),
        "note": "Squash the circle and every shared note splits in two.",
    },
    "triangle": {
        "label": "triangle",
        "points": _ring(150, _triangle),
        "note": "Three-fold symmetry. Some notes come alone, the rest in "
        "pairs that turn with the bow.",
    },
    "hexagon": {
        "label": "hexagon",
        "points": _ring(150, _regular(6, 0.5)),
        "note": "To its lowest notes, very nearly a circle. Higher up, the six "
        "corners start to matter.",
    },
    "flower": {
        "label": "flower",
        "points": _petals(),
        "note": "Five petals. Each figure keeps the five-fold symmetry, or "
        "shares its note with a partner and turns.",
    },
    "stadium": {
        "label": "stadium",
        "points": _stadium(),
        "note": "Bunimovich's stadium, where a bouncing ball moves "
        "chaotically. Its higher figures stop looking like patterns.",
    },
    "guitar": {
        "label": "guitar",
        "points": _mirrored(_GUITAR),
        "note": "A guitar top without its soundhole. Builders scatter tea "
        "leaves on real ones to see figures like these.",
    },
    "violin": {
        "label": "violin",
        "points": _mirrored(_VIOLIN),
        "note": "A violin outline in flat brass. Luthiers tune real tops, "
        "arched spruce, by figures like these.",
    },
}


def preset_points(key: str) -> list[list[float]] | None:
    """Outline of a preset at its meshing scale, or None for the square."""
    plate = PRESETS[key]
    if "points" in plate:
        return [[x * _SCALE, y * _SCALE] for x, y in plate["points"]]
    return None


def preset_request(key: str, modes: int = 6) -> dict:
    """The solve request for a preset, in the same form a drawn shape uses."""
    points = preset_points(key)
    if points is None:
        return {"shape": "square", "modes": modes}
    return {"points": points, "modes": modes}
