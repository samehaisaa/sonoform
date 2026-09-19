"""Tests for the inverse problem and the bound that constrains it."""

import numpy as np
import pytest

from sonoform.geometry import FourierShape, star_mesh
from sonoform.inverse import (
    CHORDS,
    PPW_FREQUENCY_BOUND,
    PPW_RATIO_BOUND,
    THIRD_PARTIAL_CEILING,
    check_feasibility,
    fit_scale,
    solve_inverse,
    verify,
)
from sonoform.spectrum import solve_spectrum


def test_the_disc_attains_the_ppw_bound():
    """Ashbaugh-Benguria says the disc is the unique maximiser of lambda2/lambda1.

    This checks the theorem numerically, which also checks that our bound
    constant is the right one.
    """
    spec = solve_spectrum(
        star_mesh(lambda t: np.ones_like(t), n_radial=26, n_angular=160),
        k=2,
        order=2,
    )
    ratio = spec.eigenvalues[1] / spec.eigenvalues[0]
    assert ratio == pytest.approx(PPW_RATIO_BOUND, rel=3e-3)


def test_no_shape_beats_the_bound():
    """Sample the search space and confirm nothing exceeds the ceiling."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(25):
        v = np.concatenate([[1.0], rng.normal(scale=0.15, size=8)])
        shape = FourierShape.from_vector(v)
        if not shape.is_valid():
            continue
        spec = solve_spectrum(shape.mesh(n_radial=12, n_angular=64), k=2, order=1)
        worst = max(worst, spec.eigenvalues[1] / spec.eigenvalues[0])
    # a little slack for discretisation, but nowhere near enough to reach 4.0
    assert worst < PPW_RATIO_BOUND * 1.02


def test_octave_drum_is_rejected_as_impossible():
    f = check_feasibility(CHORDS["octaves"])
    assert not f.achievable
    assert "Ashbaugh-Benguria" in f.reason
    assert f.nearest[1] == pytest.approx(PPW_FREQUENCY_BOUND)


def test_feasible_chords_pass_the_gate():
    # "fifths" is deliberately absent: its third partial is out of reach and
    # the gate is right to refuse it.
    for name in ["major", "minor", "diminished", "augmented", "minor7"]:
        assert check_feasibility(CHORDS[name]).achievable, name


def test_feasibility_object_is_truthy():
    assert check_feasibility([1.0, 1.25, 1.5])
    assert not check_feasibility([1.0, 2.0])


def test_solve_inverse_refuses_an_impossible_target():
    with pytest.raises(ValueError, match="no planar drum can exceed"):
        solve_inverse("octaves")


def test_solve_inverse_validates_its_target():
    with pytest.raises(ValueError, match="first target ratio must be 1.0"):
        solve_inverse([1.2, 1.5])
    with pytest.raises(ValueError, match="strictly increasing"):
        solve_inverse([1.0, 1.5, 1.2])
    with pytest.raises(ValueError, match="unknown chord"):
        solve_inverse("diminished9")


def test_fit_scale_places_the_fundamental():
    shape = FourierShape(1.0)
    s = fit_scale(shape, fundamental_hz=220.0)
    scaled = FourierShape(1.0 * s)
    spec = solve_spectrum(scaled.mesh(), k=1, order=2)
    hz = 100.0 * np.sqrt(spec.eigenvalues[0]) / (2.0 * np.pi)
    assert hz == pytest.approx(220.0, rel=1e-6)


@pytest.mark.slow
def test_major_triad_is_solved_to_within_a_few_cents():
    """The headline claim, at the settings a user actually gets.

    Deliberately left at the defaults rather than pinning n_harmonics. Four
    harmonics only reach about 10 cents on a major triad, so a pinned low
    count would assert a configuration nobody runs.
    """
    result = solve_inverse("major", seed=1, n_restarts=2)
    assert result.success, f"reported {result.max_cents:.1f} cents"
    independent = 1200.0 * np.log2(verify(result) / result.target)
    assert np.abs(independent).max() < 5.0, (
        f"independent check gave {np.abs(independent).max():.1f} cents"
    )


@pytest.mark.slow
def test_four_harmonics_is_not_enough_for_a_triad():
    """Why the default is five.

    Kept so that lowering the default silently is caught.
    """
    assert not solve_inverse("major", n_harmonics=4, seed=1, n_restarts=2).success


@pytest.mark.slow
def test_solution_shape_is_geometrically_valid():
    result = solve_inverse("minor", n_harmonics=3, seed=2, n_restarts=1)
    assert result.shape.is_valid()
    assert result.shape.area() > 0.0
    boundary = result.shape.boundary()
    assert np.all(np.isfinite(boundary))


def test_fifths_is_rejected_on_the_third_partial():
    """Regression: this chord silently returned 402 cents of nonsense.

    The gate only looked at the second partial, so a target whose third
    partial is unreachable sailed through and the optimiser returned whatever
    it could manage.
    """
    f = check_feasibility(CHORDS["fifths"])
    assert not f.achievable
    assert "third partial" in f.reason
    assert f.nearest[2] == pytest.approx(THIRD_PARTIAL_CEILING)


def test_every_named_chord_either_solves_or_is_refused():
    """No chord in the table may quietly produce a bad answer."""
    for name, ratios in CHORDS.items():
        feasible = check_feasibility(ratios)
        if not feasible:
            with pytest.raises(ValueError):
                solve_inverse(name)


def test_third_partial_ceiling_is_consistent_with_the_proven_one():
    assert THIRD_PARTIAL_CEILING > PPW_FREQUENCY_BOUND


@pytest.mark.slow
def test_major_seventh_reports_its_own_failure():
    """It passes the feasibility gate but does not converge.

    The contract is that it says so rather than returning a confident wrong
    answer.
    """
    result = solve_inverse("major7", n_harmonics=5, seed=3, n_restarts=2)
    assert not result.success
    assert result.max_cents > 5.0


@pytest.mark.slow
def test_same_seed_gives_the_same_shape():
    a = solve_inverse("major", seed=1, n_restarts=2).shape.to_vector()
    b = solve_inverse("major", seed=1, n_restarts=2).shape.to_vector()
    np.testing.assert_array_equal(a, b)
