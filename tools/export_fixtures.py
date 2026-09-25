"""Reference numbers for the page's JavaScript, from the Python package.

    python tools/export_fixtures.py tests/js/fixtures

The page evaluates plates, synthesises strikes and radiates sound in
JavaScript, and none of that is trusted until it agrees with the package. So
this writes the plates the app has always shown, solved in metres at brass,
next to what scikit-fem and sonoform.audio say about them at the same points.
tests/test_web.py runs it and then the JavaScript tests against its output.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from sonoform.audio import AIR_SPEED, radiation_decay, rayleigh_integral, render_plate
from sonoform.export import (
    EXPONENTS,
    MODES,
    payload_from_spectrum,
    quintic_coefficients,
)
from sonoform.plate import BRASS, SPAN, outline_mesh, solve_plate, square_mesh
from sonoform.presets import preset_points

# Off the axis, so the odd modes, which cancel on it, are heard too.
LISTENER = (0.031, -0.022, 0.35)
STRIKE = (0.041, 0.017)


def fixture(key: str) -> dict:
    points = preset_points(key)
    mesh = square_mesh() if points is None else outline_mesh(points)[0]
    spec = solve_plate(mesh, BRASS, k=MODES)
    payload = payload_from_spectrum(key, spec, SPAN)
    _frames, _coeffs, peaks = quintic_coefficients(spec)
    # The page's modes are the package's divided by a signed peak, so its
    # mass-normalised modes carry that sign. Phasors are matched to it.
    sign = np.sign(peaks)

    rng = np.random.default_rng(7)
    tri = rng.integers(0, mesh.t.shape[1], 48)
    bary = rng.dirichlet([2.0, 2.0, 2.0], 48).T
    probe = np.einsum("dct,ct->dt", mesh.p[:, mesh.t[:, tri]], bary)
    values = np.asarray(spec.basis.probes(probe) @ spec.eigenvectors) / peaks

    omega = np.sqrt(spec.eigenvalues)
    rayleigh = []
    for i in range(len(spec)):
        z = rayleigh_integral(
            spec, spec.eigenvectors[:, i], LISTENER, omega[i] / AIR_SPEED
        )
        rayleigh.append([float(z.real * sign[i]), float(z.imag * sign[i])])

    _signal, struck = render_plate(spec, STRIKE, duration=0.05)
    centroid = np.asarray(mesh.p, dtype=float).mean(axis=1)
    for i, mode in enumerate(struck):
        mode["velocity"] *= float(sign[i])

    return {
        "key": key,
        "span": SPAN,
        "material": {
            "young": BRASS.young,
            "poisson": BRASS.poisson,
            "density": BRASS.density,
            "lossFactor": BRASS.loss_factor,
        },
        "thickness": BRASS.thickness,
        "exponents": EXPONENTS,
        "payload": payload,
        "hz": [float(f) for f in spec.frequencies],
        "probe": {
            "points": (probe.T / SPAN).tolist(),
            "values": values.tolist(),
        },
        "norms": [float(v) for v in 1.0 / (BRASS.areal_density * peaks**2 * SPAN**2)],
        "radiation": [float(radiation_decay(spec, i)) for i in range(len(spec))],
        "listener": list(LISTENER),
        "rayleigh": rayleigh,
        "strike": {
            "point": [STRIKE[0] / SPAN, STRIKE[1] / SPAN],
            "listener": [float(centroid[0]), float(centroid[1]), 0.35],
            "modes": struck,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=pathlib.Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for key in ("square", "circle", "triangle"):
        path = args.out / f"{key}.json"
        path.write_text(json.dumps(fixture(key)), encoding="utf-8")
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
