"""The transport law, tested before the sand is allowed to look pretty.

Abramian, Protiere, Lazarus and Devauchelle, Phys. Rev. Research 7, L032001
(2025): grains on a vibrating plate are not pushed toward the nodes. There is
no average force at all. They bounce, and the bouncing is violent where the
plate moves and gentle where it does not, so the diffusivity D(x) is small near
a node. The steady state of a diffusion with no drift and space-dependent D is

    rho(x) proportional to 1 / D(x)

which piles the grains up exactly where the plate is quiet.

The reason this file exists, and exists first, is that the wrong convention
does not crash. Sampling D at the midpoint or adding the grad(D) "spurious
drift correction" that much of the SDE literature recommends gives the
anti-Ito interpretation, whose steady state is uniform. The plate comes out
blank, the simulation runs happily, and nothing reports an error.
"""

import numpy as np
import pytest

from sonoform.sand import diffusivity, sand_step


def _all_inside(points):
    return np.ones(np.asarray(points).shape[1], dtype=bool)


def _periodic_density(x, bins=48):
    counts, _ = np.histogram(np.mod(x, 1.0), bins=bins, range=(0.0, 1.0))
    return counts / counts.mean()


def test_ito_convention_gives_density_inversely_proportional_to_D():
    """The load-bearing test. rho ~ 1/D, on a periodic line.

    D(x) = 0.05 + sin^2(2 pi x), so D is smallest at x = 0 and x = 1/2 and the
    walkers must pile up there.
    """
    rng = np.random.default_rng(0)
    n, steps, dt = 80_000, 4000, 2.0e-4
    x = rng.uniform(0.0, 1.0, n)

    def D(pos):
        return 0.05 + np.sin(2.0 * np.pi * pos) ** 2

    for _ in range(steps):
        # sample D at the PRE-step position; this is what makes it Ito
        x = x + np.sqrt(2.0 * D(x) * dt) * rng.standard_normal(n)

    centres = (np.arange(48) + 0.5) / 48.0
    rho = _periodic_density(x)
    predicted = 1.0 / D(centres)
    predicted = predicted / predicted.mean()
    corr = float(np.corrcoef(rho, predicted)[0, 1])
    assert corr > 0.99, f"rho vs 1/D correlation was {corr:.4f}"
    assert rho.max() / rho.min() > 3.0, "the walkers did not concentrate"


def test_anti_ito_would_flatten_the_plate():
    """Guard against the plausible-looking fix.

    Adding the grad(D) drift gives the anti-Ito convention, whose steady state
    is uniform. This test records that failure mode so nobody reintroduces it
    believing they are correcting a bug.
    """
    rng = np.random.default_rng(1)
    n, steps, dt = 40_000, 3000, 2.0e-4
    x = rng.uniform(0.0, 1.0, n)

    def D(pos):
        return 0.05 + np.sin(2.0 * np.pi * pos) ** 2

    def dD(pos):
        return 2.0 * np.sin(2.0 * np.pi * pos) * np.cos(2.0 * np.pi * pos) * 2.0 * np.pi

    for _ in range(steps):
        x = x + dD(x) * dt + np.sqrt(2.0 * D(x) * dt) * rng.standard_normal(n)

    rho = _periodic_density(x)
    assert rho.max() / rho.min() < 1.6, (
        "anti-Ito is supposed to come out flat; if this now concentrates, the "
        "convention in sand_step has probably changed"
    )


def test_diffusivity_is_small_where_the_plate_is_still():
    amplitude = np.array([0.0, 0.25, 1.0])
    d = diffusivity(amplitude, floor=0.02, scale=1.0)
    assert d[0] == pytest.approx(0.02)
    assert d[0] < d[1] < d[2]


def test_sand_step_has_no_drift_term():
    """With zero diffusion nothing may move, whatever the gradient says.

    Drifting along -mu grad(A^2) is the intuitive model and it is the wrong
    one. If any drift term is present, a grain on a steep slope moves here
    even with the noise switched off, so this pins the absence of one.
    """
    positions = np.array([[0.4, -0.2], [0.1, 0.3]])
    updated = sand_step(
        positions,
        amplitude=np.array([0.4, 0.9]),
        gradient=np.array([[1.0, -2.0], [0.5, 3.0]]),
        diffusion=0.0,
        dt=0.1,
        rng=np.random.default_rng(0),
        contains=_all_inside,
    )
    np.testing.assert_allclose(updated, positions)


def test_grains_concentrate_on_the_nodal_line_of_a_real_mode():
    """End to end in 2-D: A(x, y) = x, so the node is the line x = 0."""
    rng = np.random.default_rng(2)
    n = 6000
    pos = np.vstack([rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n)])

    def inside(points):
        points = np.asarray(points)
        return (np.abs(points[0]) <= 1.0) & (np.abs(points[1]) <= 1.0)

    start = float(np.median(np.abs(pos[0])))
    for _ in range(900):
        amplitude = pos[0]
        gradient = np.vstack([np.ones(n), np.zeros(n)])
        pos = sand_step(
            pos,
            amplitude=amplitude,
            gradient=gradient,
            diffusion=4.0e-4,
            dt=1.0,
            rng=rng,
            contains=inside,
        )
    end = float(np.median(np.abs(pos[0])))
    assert end < 0.5 * start, f"median |x| went {start:.3f} -> {end:.3f}"
    assert np.all(inside(pos)), "grains escaped the plate"
