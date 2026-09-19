"""Accuracy tests against domains whose spectra are known in closed form."""

import numpy as np
import pytest
from scipy.special import jn_zeros
from skfem import MeshTri

from sonoform.geometry import FourierShape, polygon_mesh, star_mesh
from sonoform.spectrum import solve_spectrum


def square_exact(k):
    vals = sorted(np.pi**2 * (m**2 + n**2) for m in range(1, 9) for n in range(1, 9))
    return np.array(vals[:k])


def disc_exact(k, radius=1.0):
    vals = []
    for order in range(6):
        zeros = jn_zeros(order, 6)
        vals.extend(zeros**2)
        if order > 0:  # non-radial modes are doubly degenerate
            vals.extend(zeros**2)
    return np.array(sorted(vals)[:k]) / radius**2


def test_unit_square_matches_closed_form():
    spec = solve_spectrum(MeshTri().refined(6), k=8, order=2)
    rel = np.abs(spec.eigenvalues - square_exact(8)) / square_exact(8)
    assert rel.max() < 1e-5, f"max relative error {rel.max():.2e}"


def test_square_degeneracies_are_resolved():
    """Modes 2 and 3 are both 5 pi^2, modes 5 and 6 both 10 pi^2."""
    spec = solve_spectrum(MeshTri().refined(6), k=6, order=2)
    lam = spec.eigenvalues
    assert lam[1] == pytest.approx(lam[2], rel=1e-5)
    assert lam[4] == pytest.approx(lam[5], rel=1e-5)


def test_star_mesh_of_constant_radius_is_a_disc():
    mesh = star_mesh(lambda t: np.ones_like(t), n_radial=24, n_angular=128)
    spec = solve_spectrum(mesh, k=6, order=2)
    rel = np.abs(spec.eigenvalues - disc_exact(6)) / disc_exact(6)
    # bounded by the inscribed-polygon area deficit, not by solver error
    assert rel.max() < 5e-3, f"max relative error {rel.max():.2e}"


def test_polygon_mesh_reproduces_the_square():
    mesh = polygon_mesh([[0, 0], [1, 0], [1, 1], [0, 1]], max_area=2e-4)
    spec = solve_spectrum(mesh, k=4, order=2)
    rel = np.abs(spec.eigenvalues - square_exact(4)) / square_exact(4)
    assert rel.max() < 1e-3, f"max relative error {rel.max():.2e}"


def test_polygon_mesh_handles_a_non_convex_shape():
    """An L-shape. Its fundamental is known to about 9.6397238."""
    l_shape = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]
    mesh = polygon_mesh(l_shape, max_area=1e-3)
    spec = solve_spectrum(mesh, k=1, order=2)
    assert spec.eigenvalues[0] == pytest.approx(9.6397238, rel=2e-2)


def test_eigenvalues_scale_inversely_with_area():
    """Scaling a domain by c divides every eigenvalue by c squared."""
    base = solve_spectrum(star_mesh(lambda t: np.ones_like(t)), k=4, order=1)
    scaled = solve_spectrum(star_mesh(lambda t: 2.0 * np.ones_like(t)), k=4, order=1)
    np.testing.assert_allclose(scaled.eigenvalues * 4.0, base.eigenvalues, rtol=1e-9)


def test_ratios_are_scale_invariant():
    small = solve_spectrum(star_mesh(lambda t: 0.3 * np.ones_like(t)), k=5, order=1)
    large = solve_spectrum(star_mesh(lambda t: 3.0 * np.ones_like(t)), k=5, order=1)
    np.testing.assert_allclose(small.ratios, large.ratios, rtol=1e-9)


def test_fourier_shape_roundtrips_through_a_vector():
    shape = FourierShape(1.0, [0.1, -0.05], [0.0, 0.02])
    restored = FourierShape.from_vector(shape.to_vector())
    np.testing.assert_allclose(restored.to_vector(), shape.to_vector())


def test_fourier_shape_area_matches_the_disc():
    assert FourierShape(1.0).area() == pytest.approx(np.pi, rel=1e-6)


def test_fourier_shape_rejects_a_boundary_through_the_origin():
    assert FourierShape(1.0, [0.2]).is_valid()
    assert not FourierShape(1.0, [1.5]).is_valid()


def test_from_vector_rejects_mismatched_coefficients():
    with pytest.raises(ValueError, match="odd number of coefficients"):
        FourierShape.from_vector([1.0, 0.1])


def test_solve_spectrum_rejects_too_many_modes():
    with pytest.raises(ValueError, match="interior degrees of freedom"):
        solve_spectrum(MeshTri().refined(1), k=500, order=1)


def test_solve_spectrum_is_reproducible():
    """ARPACK randomises its start vector unless told otherwise.

    Without a pinned v0 the same call returns eigenvalues differing in the
    last digits, which was enough to make solve_inverse take a different path
    from an identical seed.
    """
    from sonoform.geometry import FourierShape

    mesh = FourierShape(0.95, [0.12, -0.06], [0.09, 0.03]).mesh(8, 160)
    runs = [solve_spectrum(mesh, k=4, order=2).eigenvalues for _ in range(3)]
    for other in runs[1:]:
        np.testing.assert_array_equal(runs[0], other)


def test_resample_closed_path_evens_out_spacing():
    """A mouse drag clumps points wherever the hand slowed down."""
    from sonoform.geometry import resample_closed_path

    clumped = np.array([[0, 0], [0.01, 0], [0.02, 0], [1, 0], [1, 1], [0, 1]])
    out = resample_closed_path(clumped, n=60)
    steps = np.hypot(*np.diff(np.hstack([out, out[:, :1]]), axis=1))
    assert out.shape == (2, 60)
    assert steps.std() / steps.mean() < 1e-6


def test_resample_handles_a_repeated_closing_point():
    from sonoform.geometry import resample_closed_path

    out = resample_closed_path([[0, 0], [1, 0], [1, 1], [0, 0]], n=24)
    assert out.shape == (2, 24)


def test_resample_rejects_degenerate_input():
    from sonoform.geometry import resample_closed_path

    with pytest.raises(ValueError, match="at least 3 points"):
        resample_closed_path([[0, 0], [1, 1]])
    with pytest.raises(ValueError, match="fewer than 3 distinct"):
        resample_closed_path([[0, 0], [0, 0], [0, 0], [0, 0]])


def test_self_intersecting_outlines_are_caught():
    """A figure eight has no well defined interior, so meshing it is nonsense."""
    from sonoform.geometry import is_simple_polygon

    assert is_simple_polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert not is_simple_polygon([[0, 0], [1, 1], [1, 0], [0, 1]])
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    blob = np.vstack([np.cos(theta), np.sin(theta)]) * (1 + 0.3 * np.sin(3 * theta))
    assert is_simple_polygon(blob)
