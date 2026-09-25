"""Plates solved ahead of time, in the form the browser draws them.

The page never solves anything. It evaluates. An Argyris mode is a quintic
polynomial on every triangle, continuous with its first derivatives across
every edge, so a mode can travel as 21 monomial coefficients per element and
be evaluated exactly wherever it is needed: per pixel for the nodal lines and
the hologram fringes, on a grid for the sand, at quadrature points for the
sound. Nothing on the page is interpolated between samples of the solution.

Coordinates are in plate units, the physical span divided out, and every
solve uses unit rigidity and unit areal density. What comes back is the
frequency parameter

``Ω = ω L² √(ρh / D)``

which depends on nothing but the outline and Poisson's ratio. The page turns
it into hertz for whichever sheet, thickness and size is on screen,

``f = Ω √(D / ρh) / (2π L²)``,    ``D = E h³ / 12 (1 - ν²)``

and that step is exact rather than approximate, because Kirchhoff's equation
has no other length or stiffness in it. Poisson's ratio is the one constant
that does not scale out, so each distinct ν gets a solve of its own.
"""

from __future__ import annotations

import base64

import numpy as np
from skfem import Basis, ElementTriArgyris

from sonoform import __version__
from sonoform.audio import AIR_DENSITY, AIR_SPEED
from sonoform.materials import MATERIALS
from sonoform.plate import (
    SPAN,
    PlateMaterial,
    PlateSpectrum,
    outline_mesh,
    solve_plate,
    square_mesh,
)
from sonoform.presets import PRESETS, preset_points

__all__ = [
    "CLUSTER_GAP",
    "EXPONENTS",
    "MODES",
    "boundary_loop",
    "degenerate_clusters",
    "manifest",
    "payload_from_spectrum",
    "plate_file",
    "plate_mesh",
    "plate_payload",
    "poisson_ratios",
    "quintic_coefficients",
    "unit_material",
    "verification",
]

# Twenty flexible modes. Argyris resolves all of them on the default meshes:
# refining the circle from 318 to 1240 triangles moves none of the first
# twenty by more than 0.1 Hz.
MODES = 20

# Modes closer than this, relative, are treated as one note. A symmetric
# outline has exactly degenerate pairs, and a mesh that is not perfectly
# symmetric splits them, by up to 0.14 % on the triangle's sharp corners.
# That split is discretisation, not physics, and it must not survive as two
# notes: bowed, a degenerate pair responds as whichever combination the bow
# point selects, and struck, a false split would beat.
#
# The closest genuinely distinct neighbours on any preset are 0.39 % apart,
# on the hexagon at 1747 and 1754 Hz. Rotating the modes by 60 degrees tells
# the two kinds apart without reference to frequency: a mode of a
# two-dimensional representation overlaps itself by cos 60° = ±0.5, a
# one-dimensional one by ±1, and those two come out at -0.51 and +0.97.
CLUSTER_GAP = 2.5e-3

# Monomials x^a y^b of total degree at most five, grouped by degree. The page
# evaluates them in exactly this order.
EXPONENTS = [(d - b, b) for d in range(6) for b in range(d + 1)]

# The degree-five principal lattice of the reference triangle. Unisolvent for
# quintics, so the fit through it is exact rather than least squares.
_LATTICE = np.array(
    [(i / 5.0, j / 5.0) for i in range(6) for j in range(6 - i)], dtype=float
).T

# The square in Leissa's dimensionless form at ν = 0.3, five figures, as
# reproduced by Narita (2022). tests/test_plate.py rebuilds this table with an
# independent Rayleigh-Ritz solve before trusting it.
_LEISSA_SQUARE = (13.468, 19.596, 24.270, 34.801, 34.801, 61.093)

# A free disc at ν = 0.33: roots of Kirchhoff's frequency equation for the
# modes with two nodal diameters, one nodal circle, and three diameters,
# as Ω on the radius. Checked against the Bessel residual in the tests.
_KIRCHHOFF_DISC = ((2, 0, 5.262037), (0, 1, 9.068899), (3, 0, 12.243894))


def unit_material(poisson: float) -> PlateMaterial:
    """A sheet with D = 1 and ρh = 1, so that ω on a unit plate is Ω."""
    return PlateMaterial(
        young=12.0 * (1.0 - poisson * poisson),
        poisson=poisson,
        density=1.0,
        thickness=1.0,
        loss_factor=0.0,
    )


def poisson_ratios() -> list[float]:
    """Every distinct ν among the materials, in the order they first appear."""
    seen: list[float] = []
    for entry in MATERIALS.values():
        nu = round(float(entry["material"].poisson), 4)
        if nu not in seen:
            seen.append(nu)
    return seen


def plate_file(key: str, poisson: float) -> str:
    """Where a plate's payload lives, relative to the data directory."""
    return f"plates/{key}-{poisson:.2f}.json"


def plate_mesh(key: str):
    """The preset's mesh in plate units, its longest side one."""
    if key not in PRESETS:
        raise ValueError(f"no preset called {key!r}")
    points = preset_points(key)
    if points is None:
        return square_mesh(span=1.0)
    mesh, _outline = outline_mesh(points, span=1.0)
    return mesh


def quintic_coefficients(spec: PlateSpectrum):
    """Every mode as a quintic per element, in a frame centred on the element.

    Returns ``(frames, coefficients, peaks)``. ``frames`` is ``(nt, 3)``, holding the
    centroid and the radius ``s`` of each element, and a mode's value at
    ``(x, y)`` inside element ``e`` is

    ``Σ c[e, m] ξ^a η^b``,   ``ξ = (x - cx) / s``,   ``η = (y - cy) / s``

    over the monomials of :data:`EXPONENTS`. Scaling by the element's own
    radius keeps every ``ξ`` inside the unit disc, which keeps the fit well
    conditioned on small triangles.

    ``coefficients`` is ``(k, nt, 21)``. Each mode is oriented and scaled the
    way :meth:`PlateSpectrum.vertex_values` does it, largest magnitude equal to
    one and positive, but taken over all 21 lattice points of every element
    rather than the vertices alone. ``peaks`` holds the signed value each mode
    was divided by, so ``peaks[i] * field`` is mode ``i`` mass-normalised again.
    """
    mesh = spec.mesh
    weights = np.full(_LATTICE.shape[1], 1.0 / _LATTICE.shape[1])
    # A custom quadrature puts the lattice inside every element directly, so
    # nothing has to be located and no point on an edge can be refused.
    basis = Basis(mesh, ElementTriArgyris(), quadrature=(_LATTICE, weights))
    xy = np.asarray(basis.global_coordinates())  # (2, nt, 21)

    corners = np.asarray(mesh.p[:, mesh.t], dtype=float)  # (2, 3, nt)
    centre = corners.mean(axis=1)
    radius = np.sqrt(((corners - centre[:, None, :]) ** 2).sum(axis=0)).max(axis=0)
    frames = np.vstack([centre, radius]).T  # (nt, 3)

    xi = (xy[0] - centre[0][:, None]) / radius[:, None]
    eta = (xy[1] - centre[1][:, None]) / radius[:, None]
    vandermonde = np.stack([xi**a * eta**b for a, b in EXPONENTS], axis=-1)

    samples = np.stack(
        [np.asarray(basis.interpolate(vector)) for vector in spec.eigenvectors.T],
        axis=-1,
    )  # (nt, 21, k)
    coeffs = np.linalg.solve(vandermonde, samples)  # (nt, 21, k)

    flat = samples.reshape(-1, samples.shape[-1])
    peak_at = np.argmax(np.abs(flat), axis=0)
    peak = flat[peak_at, np.arange(flat.shape[1])]
    peak = np.where(peak == 0.0, 1.0, peak)
    coeffs = coeffs / peak
    return frames, np.ascontiguousarray(np.transpose(coeffs, (2, 0, 1))), peak


def boundary_loop(mesh) -> np.ndarray:
    """Vertex indices of the outer boundary, in order, counter-clockwise."""
    facets = mesh.facets[:, mesh.boundary_facets()]
    following: dict[int, list[int]] = {}
    for a, b in facets.T:
        following.setdefault(int(a), []).append(int(b))
        following.setdefault(int(b), []).append(int(a))
    start = int(facets[0, 0])
    loop, previous, current = [start], None, start
    while True:
        a, b = following[current]
        step = b if a == previous else a
        if step == start:
            break
        loop.append(step)
        previous, current = current, step
        if len(loop) > facets.shape[1]:
            raise RuntimeError("the boundary is not a single loop")
    if len(loop) != facets.shape[1]:
        raise RuntimeError("the boundary is not a single loop")
    x, y = mesh.p[:, loop]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0.0:
        loop.reverse()
    return np.asarray(loop, dtype=np.int64)


def degenerate_clusters(omega, gap: float = CLUSTER_GAP) -> list[list[int]]:
    """Group ascending frequencies whose neighbours sit within ``gap``."""
    omega = np.asarray(omega, dtype=float)
    clusters = [[0]]
    for i in range(1, omega.size):
        if omega[i] - omega[i - 1] <= gap * omega[i - 1]:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def _b64(array, dtype) -> str:
    raw = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    return base64.b64encode(raw.astype(raw.dtype.newbyteorder("<")).tobytes()).decode(
        "ascii"
    )


def plate_payload(key: str, poisson: float, modes: int = MODES) -> dict:
    """Everything the page needs to draw, bow, strike and hear one plate.

    Solves every time it is called. Callers that serve these repeatedly keep
    the serialised result rather than the spectrum, which carries the whole
    Argyris basis and runs to megabytes.
    """
    poisson = round(float(poisson), 4)
    spec = solve_plate(plate_mesh(key), unit_material(poisson), k=modes)
    return payload_from_spectrum(key, spec, span=1.0)


def payload_from_spectrum(key: str, spec: PlateSpectrum, span: float) -> dict:
    """The payload for a spectrum solved on a mesh whose longest side is ``span``.

    Any sheet will do: coordinates are divided by the span and frequencies
    turned into Ω, so a plate solved in metres at brass gives the same payload
    as the unit solve, up to the sign of each mode. The tests lean on that to
    check the page against scikit-fem, and against :mod:`sonoform.audio`, on
    the very plate the app has always shown.
    """
    mesh = spec.mesh
    material = spec.material
    scale = np.sqrt(material.areal_density / material.flexural_rigidity)
    omega = 2.0 * np.pi * spec.frequencies * span**2 * scale
    frames, coeffs, _peaks = quintic_coefficients(spec)
    frames = frames / span
    index = np.uint16 if mesh.p.shape[1] < 65536 else np.uint32
    return {
        "format": 1,
        "key": key,
        "poisson": round(float(material.poisson), 4),
        "omega": [round(float(w), 6) for w in omega],
        "clusters": degenerate_clusters(omega),
        "elements": int(mesh.t.shape[1]),
        "vertices": int(mesh.p.shape[1]),
        "dofs": int(spec.eigenvectors.shape[0]),
        "kernel": float(np.abs(spec.kernel).max() / spec.eigenvalues[0]),
        "index": "u16" if index is np.uint16 else "u32",
        "points": _b64(mesh.p.T / span, np.float32),
        "triangles": _b64(mesh.t.T, index),
        "boundary": _b64(boundary_loop(mesh), index),
        "frames": _b64(frames, np.float32),
        "coefficients": _b64(coeffs, np.float32),
    }


def _icon_outline(key: str, n: int = 72) -> list[list[float]]:
    """A light copy of the outline, in plate units, for drawing buttons."""
    points = preset_points(key)
    if points is None:
        path = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    else:
        path = np.asarray(points, dtype=float)
        path = path - path.mean(axis=0)
        path = path / np.ptp(path, axis=0).max()
        step = max(1, len(path) // n)
        path = path[::step]
    return [[round(float(x), 4), round(float(y), 4)] for x, y in path]


def manifest() -> dict:
    """What exists, without solving anything."""
    ratios = poisson_ratios()
    return {
        "format": 1,
        "version": __version__,
        "span": SPAN,
        "thickness": MATERIALS["brass"]["material"].thickness,
        "modes": MODES,
        "clusterGap": CLUSTER_GAP,
        "air": {"density": AIR_DENSITY, "speed": AIR_SPEED},
        "materials": [
            {
                "key": key,
                "label": entry["label"],
                "note": entry["note"],
                "young": entry["material"].young,
                "poisson": entry["material"].poisson,
                "density": entry["material"].density,
                "lossFactor": entry["material"].loss_factor,
            }
            for key, entry in MATERIALS.items()
        ],
        "plates": [
            {
                "key": key,
                "label": plate["label"],
                "note": plate["note"],
                "outline": _icon_outline(key),
                "files": {f"{nu:.2f}": plate_file(key, nu) for nu in ratios},
            }
            for key, plate in PRESETS.items()
        ],
    }


def verification() -> dict:
    """The benchmarks, solved again at build time, for the page to show.

    These are the same comparisons the test suite makes, run by the same code
    that produced every plate on the site, so the numbers a visitor reads are
    the numbers this build actually got.
    """
    square = solve_plate(square_mesh(span=1.0), unit_material(0.3), k=6)
    got = 2.0 * np.pi * square.frequencies

    disc_mesh, _ = outline_mesh(
        np.c_[np.cos(np.linspace(0, 2 * np.pi, 160, endpoint=False)),
              np.sin(np.linspace(0, 2 * np.pi, 160, endpoint=False))].tolist(),
        span=2.0,
        max_triangles=600,
    )
    disc = solve_plate(disc_mesh, unit_material(0.33), k=4)
    # On a unit radius Ω is ω itself. Modes 0 and 1 are the degenerate pair
    # with two nodal diameters, 2 is the first ring, 3 the three-diameter pair.
    disc_omega = 2.0 * np.pi * disc.frequencies
    disc_rows = []
    for (diameters, circles, reference), index in zip(
        _KIRCHHOFF_DISC, (0, 2, 3), strict=True
    ):
        disc_rows.append(
            {
                "diameters": diameters,
                "circles": circles,
                "computed": round(float(disc_omega[index]), 5),
                "reference": reference,
            }
        )
    return {
        "square": {
            "poisson": 0.3,
            "source": "Leissa, Vibration of Plates, NASA SP-160 (1969)",
            "computed": [round(float(v), 4) for v in got],
            "reference": list(_LEISSA_SQUARE),
            "kernel": float(np.abs(square.kernel).max() / square.eigenvalues[0]),
            "elements": int(square.mesh.t.shape[1]),
        },
        "disc": {
            "poisson": 0.33,
            "source": "Kirchhoff's frequency equation for the free disc",
            "rows": disc_rows,
            "elements": int(disc_mesh.t.shape[1]),
        },
    }
