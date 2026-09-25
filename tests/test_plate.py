"""Free-plate spectra against the classical frequency parameters."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from sonoform.audio import rayleigh_integral
from sonoform.geometry import star_mesh
from sonoform.plate import (
    BRASS,
    SPAN,
    PlateMaterial,
    outline_mesh,
    solve_plate,
    square_mesh,
)
from sonoform.presets import preset_points

# Five significant figures, ν = 0.3, Ω = ω a² √(ρ h / D).
# Narita, EPI International Journal of Engineering 5 (2022), reproducing
# Leissa's completely free square.
LEISSA_SQUARE = np.array([13.468, 19.596, 24.270, 34.801, 34.801, 61.093])

# λ² = Ω for a free circular plate at ν = 0.33, modes (2,0), (0,1), (3,0).
# Roots of Kirchhoff's frequency equation; (2,0) and (3,0) are degenerate.
LEISSA_DISC = {
    2: 5.262037,
    0: 9.068899,
    3: 12.243894,
}

UNIT = PlateMaterial(
    young=1.0, poisson=0.3, density=1.0, thickness=1.0, loss_factor=0.0
)
DISC_MATERIAL = PlateMaterial(
    young=1.0, poisson=0.33, density=1.0, thickness=1.0, loss_factor=0.0
)


def _parameter(spec, span=1.0):
    omega = np.sqrt(spec.eigenvalues)
    scale = np.sqrt(spec.material.areal_density / spec.material.flexural_rigidity)
    return omega * (span**2) * scale


def _frequency_residual(lam, order, poisson):
    from scipy.special import iv, ivp, jv, jvp

    bessel_j = jv(order, lam)
    bessel_i = iv(order, lam)
    dj = jvp(order, lam)
    di = ivp(order, lam)
    one = 1.0 - poisson
    moment_j = lam**2 * bessel_j + one * (lam * dj - order**2 * bessel_j)
    moment_i = lam**2 * bessel_i - one * (lam * di - order**2 * bessel_i)
    shear_j = lam**3 * dj + one * order**2 * (lam * dj - bessel_j)
    shear_i = lam**3 * di - one * order**2 * (lam * di - bessel_i)
    return moment_j * shear_i - moment_i * shear_j


@pytest.fixture(scope="module")
def square():
    return solve_plate(square_mesh(span=1.0, nrefs=3), UNIT, k=6)


def test_rigid_motions_are_the_kernel(square):
    assert square.kernel.size == 3
    assert np.abs(square.kernel).max() < 1e-8 * square.eigenvalues[0]
    assert square.eigenvalues[0] > 1.0


def _ritz_free_square(deg_max=9, poisson=0.3, nq=32):
    """Free square plate by Rayleigh-Ritz on a tensor Legendre basis.

    Shares nothing with solve_plate: different discretisation, different
    quadrature, no scikit-fem. A free plate has no essential boundary
    conditions, so there is nothing to impose and the raw generalised
    eigenproblem is the answer.

    Returns Omega = omega a^2 sqrt(rho h / D) on [-1, 1]^2, so a = 2 with
    D = rho h = 1, with the three rigid-body zeros still at the front.
    """
    from numpy.polynomial import legendre as leg
    from scipy.linalg import eigh

    xq, wq = np.polynomial.legendre.leggauss(nq)
    n1 = deg_max + 1
    val, d1, d2 = [], [], []
    for n in range(n1):
        c = np.zeros(n + 1)
        c[n] = 1.0
        val.append(leg.legval(xq, c))
        d1.append(leg.legval(xq, leg.legder(c, 1)) if n >= 1 else np.zeros_like(xq))
        d2.append(leg.legval(xq, leg.legder(c, 2)) if n >= 2 else np.zeros_like(xq))

    weight = np.outer(wq, wq).ravel()[None, :]
    size = n1 * n1
    u = np.empty((size, nq * nq))
    uxx, uyy, uxy = (np.empty_like(u) for _ in range(3))
    for i in range(n1):
        for j in range(n1):
            k = i * n1 + j
            u[k] = np.outer(val[i], val[j]).ravel()
            uxx[k] = np.outer(d2[i], val[j]).ravel()
            uyy[k] = np.outer(val[i], d2[j]).ravel()
            uxy[k] = np.outer(d1[i], d1[j]).ravel()

    # nu (lap u)(lap v) + (1 - nu)(u_xx v_xx + 2 u_xy v_xy + u_yy v_yy)
    lap = uxx + uyy
    stiffness = poisson * (lap * weight) @ lap.T + (1.0 - poisson) * (
        (uxx * weight) @ uxx.T
        + 2.0 * (uxy * weight) @ uxy.T
        + (uyy * weight) @ uyy.T
    )
    inertia = (u * weight) @ u.T
    omega2 = eigh(
        0.5 * (stiffness + stiffness.T),
        0.5 * (inertia + inertia.T),
        eigvals_only=True,
    )
    return np.sqrt(np.maximum(omega2, 0.0)) * 4.0


def test_the_square_reference_is_verified_not_trusted():
    """LEISSA_SQUARE has to be earned, like LEISSA_DISC is.

    The disc has a closed form, so test_free_disc_matches_kirchhoff can check
    its reference against the frequency equation before using it. The square
    has no closed form, and a table of constants asserted against our own
    output is not a benchmark: it locks in whatever this solver happens to
    produce and can never reveal that the solver is wrong.

    So the reference is reproduced here by a second method. Converged to six
    figures and flat from degree 9 to degree 17.
    """
    omega = _ritz_free_square()
    assert np.abs(omega[:3]).max() < 1e-3, "rigid motions must come out at zero"
    np.testing.assert_allclose(omega[3:9], LEISSA_SQUARE, rtol=1e-4)


def test_the_solve_does_not_depend_on_the_units_of_the_mesh():
    """The same plate meshed in metres and in plate units rings the same.

    The shift that makes K + αM invertible used to be a fixed fraction of
    D/ρh. That let the rigid modes outweigh the flexible ones in the
    shift-inverted operator by six to ten orders of magnitude, set by the
    length unit alone, and ARPACK misconverged without a warning: the
    triangle, whose lowest three modes lie within 2.3 % of each other, came
    back 21 cents low at a span of one metre while a dense solve of the very
    same matrices said otherwise.
    """
    points = preset_points("triangle")
    near, _ = outline_mesh(points, span=SPAN)
    far, _ = outline_mesh(points, span=1.0)
    a = solve_plate(near, BRASS, k=6).frequencies
    b = solve_plate(far, BRASS, k=6).frequencies / SPAN**2
    cents = np.abs(1200.0 * np.log2(b / a))
    assert cents.max() < 0.01, f"{cents.max():.3f} cents between the two units"


def test_free_square_matches_leissa(square):
    got = _parameter(square)
    np.testing.assert_allclose(got, LEISSA_SQUARE, rtol=1e-4)


def test_free_square_matches_the_independent_ritz_solve(square):
    """The Argyris solve against Rayleigh-Ritz, neither one trusted a priori."""
    np.testing.assert_allclose(_parameter(square), _ritz_free_square()[3:9], rtol=1e-3)


def test_hertz_follows_from_the_frequency_parameter(square):
    """f = Ω √(D / ρh) / (2 π a²), with a = 1 in this fixture."""
    stiffness = square.material.flexural_rigidity / square.material.areal_density
    expected = LEISSA_SQUARE[0] * np.sqrt(stiffness) / (2.0 * np.pi)
    assert square.frequencies[0] == pytest.approx(expected, rel=1e-4)


def test_vertex_values_match_the_basis(square):
    probe = square.basis.probes(square.mesh.p) @ square.eigenvectors[:, 0]
    stored = square._vertex_component(0, 0)
    np.testing.assert_allclose(np.asarray(probe).ravel(), stored, atol=1e-8)


def test_odd_modes_do_not_radiate_on_axis(square):
    """An on-axis listener sits on the symmetry axis, so an odd mode cancels."""
    points = square.mesh.p.T
    mirror = cKDTree(points).query(-points)[1]
    listener = (0.0, 0.0, 0.35)
    loudest = 0.0
    for index in range(len(square)):
        values = square._vertex_component(index, 0)
        odd = np.linalg.norm(values + values[mirror]) < 1e-8 * np.linalg.norm(values)
        wave = np.sqrt(square.eigenvalues[index]) / 343.0
        gain = abs(
            rayleigh_integral(square, square.eigenvectors[:, index], listener, wave)
        )
        loudest = max(loudest, gain)
        if odd:
            assert gain < 1e-6, f"odd mode {index} radiates on axis"
    assert loudest > 1e-2


def test_free_disc_matches_kirchhoff():
    for order, root in LEISSA_DISC.items():
        assert _frequency_residual(np.sqrt(root), order, 0.33) == pytest.approx(
            0.0, abs=1e-4
        )
    mesh = star_mesh(lambda theta: np.ones_like(theta), n_radial=8, n_angular=64)
    spec = solve_plate(mesh, DISC_MATERIAL, k=4)
    got = _parameter(spec)
    assert got[0] == pytest.approx(LEISSA_DISC[2], rel=3e-3)
    assert got[1] == pytest.approx(got[0], rel=1e-3)
    assert got[2] == pytest.approx(LEISSA_DISC[0], rel=3e-3)


def test_outline_is_scaled_to_the_span_and_rejects_a_crossing():
    _mesh, path = outline_mesh([[0.0, 0.0], [2.0, 0.0], [2.0, 0.4], [0.0, 0.4]])
    assert np.ptp(path, axis=1).max() == pytest.approx(SPAN)
    with pytest.raises(ValueError, match="crosses"):
        outline_mesh(
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.2, 0.5], [0.8, 0.5]]
        )


def test_display_field_is_the_real_solution_not_interpolation():
    """Subdividing must resample the quintic, not lerp between vertices.

    If this ever degrades to interpolation the surface will look smooth and be
    wrong, which is the failure that is hard to see.

    The tolerance is 1e-5 rather than machine epsilon because probing happens
    a millionth of an element inside each triangle; lattice points on an edge
    are otherwise rejected as outside the mesh on a curved outline. Measured
    cost of that inset is 2.6e-07 on values that live in [-1, 1].
    """
    from scipy.spatial import cKDTree

    from sonoform.plate import display_field

    spec = solve_plate(square_mesh(), k=3)
    points, triangles, values, gradients = display_field(spec, level=3)

    # every original vertex survives, carrying its original value
    tree = cKDTree(points.T)
    distance, index = tree.query(spec.mesh.p.T)
    assert distance.max() < 1e-12
    for mode in range(3):
        np.testing.assert_allclose(
            values[mode][index], spec.vertex_values(mode), atol=1e-5
        )

    # and the new points agree with an independent probe of the basis
    probe = spec.basis.probes(points)
    raw = np.asarray(probe @ spec.eigenvectors[:, 0])
    peak = np.abs(raw).max()
    sign = 1.0 if raw[np.argmax(np.abs(raw))] >= 0.0 else -1.0
    np.testing.assert_allclose(values[0], raw * sign / peak, atol=1e-5)

    assert triangles.shape[1] == 9 * spec.mesh.t.shape[1]
    # and the inset must stay negligible, not quietly grow
    assert np.abs(values[0][index] - spec.vertex_values(0)).max() < 1e-5
    assert values.shape == (3, points.shape[1])
    assert gradients.shape == (3, 2, points.shape[1])


def test_display_field_merges_shared_vertices():
    """Duplicated seam vertices would tear the nodal contour."""
    from sonoform.plate import display_field

    spec = solve_plate(square_mesh(), k=1)
    points, triangles, _, _ = display_field(spec, level=2)
    per_triangle = 6  # the level-2 barycentric lattice
    assert points.shape[1] < per_triangle * spec.mesh.t.shape[1]
    assert triangles.max() == points.shape[1] - 1


def test_display_field_rejects_a_silly_level():
    from sonoform.plate import display_field

    spec = solve_plate(square_mesh(), k=1)
    with pytest.raises(ValueError, match="level must be at least 1"):
        display_field(spec, level=0)


def test_display_field_survives_a_curved_outline():
    """A subdivision point sitting exactly on a curved boundary can read as
    outside the mesh and get rejected.

    The square has straight edges aligned with the mesh, so it never exercises
    this; every drawn or preset shape does, and the failure mode is silent, a
    plate that stays square instead of reporting anything.
    """
    from sonoform.plate import display_field, outline_mesh

    angle = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)
    circle = np.vstack([np.cos(angle), np.sin(angle)]).T * 0.09
    mesh, _ = outline_mesh(circle.tolist())
    spec = solve_plate(mesh, k=4)
    points, triangles, values, _ = display_field(spec, level=3)
    assert triangles.shape[1] == 9 * mesh.t.shape[1]
    assert np.all(np.isfinite(values))
