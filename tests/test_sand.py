"""Sand gathers on the nodal set. The nodal set is the zero contour.

The transport law itself lives in test_sand_transport.py. These are the
behaviours the renderer depends on: grains stay on the plate, and the contour
they gather on can be drawn independently so the two can be compared.
"""

import numpy as np

from sonoform.sand import nodal_segments, sand_step


def _disc(points):
    points = np.asarray(points, dtype=float)
    return np.hypot(points[0], points[1]) <= 1.0


def test_a_cloud_concentrates_on_the_nodal_set():
    """A = x, so the node is the line x = 0 and the rim is the loudest part."""
    rng = np.random.default_rng(0)
    angles = rng.uniform(0.0, 2.0 * np.pi, 3000)
    radii = np.sqrt(rng.uniform(0.0, 1.0, angles.size))
    positions = np.vstack([radii * np.cos(angles), radii * np.sin(angles)])
    start = float(np.median(np.abs(positions[0])))

    # Settling is gradual, and the floor in diffusivity() stops it collapsing
    # to a line. Measured: 0.41 at rest, 0.26 by step 600, 0.18 by 1200,
    # levelling off near 0.11.
    for _ in range(1200):
        positions = sand_step(
            positions,
            amplitude=positions[0].copy(),
            diffusion=3.0e-4,
            dt=1.0,
            rng=rng,
            contains=_disc,
        )
        assert np.all(np.hypot(positions[0], positions[1]) <= 1.0 + 1e-8)

    end = float(np.median(np.abs(positions[0])))
    assert end < 0.5 * start, f"median |x| went {start:.3f} -> {end:.3f}"


def test_a_step_off_the_plate_is_reflected_back():
    rng = np.random.default_rng(3)
    positions = np.full((2, 400), 0.0)
    positions[0] = 0.98
    for _ in range(40):
        positions = sand_step(
            positions,
            amplitude=np.full(positions.shape[1], 1.0),
            diffusion=0.02,
            dt=1.0,
            rng=rng,
            contains=_disc,
        )
    assert np.all(np.hypot(positions[0], positions[1]) <= 1.0 + 1e-8)


def test_zero_diffusion_leaves_the_grains_alone():
    positions = np.array([[0.4, -0.2], [0.1, 0.3]])
    updated = sand_step(
        positions,
        amplitude=np.array([0.4, 0.9]),
        diffusion=0.0,
        dt=0.1,
        rng=np.random.default_rng(0),
        contains=_disc,
    )
    np.testing.assert_allclose(updated, positions)


def test_nodal_segments_follow_the_sign_change():
    points = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    triangles = np.array([[0], [1], [2]])
    segments = nodal_segments(points, triangles, [1.0, -1.0, -1.0])
    assert segments.shape == (1, 2, 2)
    got = segments[0]
    expected = np.array([[0.5, 0.0], [0.0, 0.5]])
    assert np.allclose(got, expected) or np.allclose(got, expected[::-1])


def test_a_field_that_does_not_change_sign_has_no_nodal_set():
    points = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    triangles = np.array([[0, 0], [1, 1], [2, 2]])
    assert nodal_segments(points, triangles, [1.0, 0.2, 0.4]).shape[0] == 0
