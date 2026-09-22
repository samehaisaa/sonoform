"""The plates the app offers before you draw one.

Single source of truth for the preset outlines, so the interactive app and the
precomputed demo cannot drift apart. Nothing here is special: every preset is
the same closed outline a drawn shape produces, at the same physical span.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PRESETS", "preset_points", "preset_request"]

# The outlines live on a unit radius and are scaled to this fraction of the
# span when they are meshed, matching what the drawing tool produces.
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


# label, then either "square" or the unit-radius outline
PRESETS: dict[str, dict] = {
    "square": {"label": "square", "shape": "square"},
    "circle": {"label": "circle", "points": _ring(160, np.ones_like)},
    "triangle": {"label": "triangle", "points": _ring(150, _triangle)},
    "ellipse": {"label": "ellipse", "points": _ring(160, _ellipse)},
    "hexagon": {"label": "hexagon", "points": _ring(150, _regular(6, 0.5))},
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
