"""The inverse problem: find a shape whose spectrum matches a target.

Mark Kac asked in 1966 whether you can hear the shape of a drum. You cannot,
quite: there are distinct shapes that sound identical. But you can often go the
other way and *design* a shape to sound roughly how you want, and that is what
this module does.

The search space is :class:`~sonoform.geometry.FourierShape`, a boundary radius
written as a truncated Fourier series. That keeps the problem to a handful of
coefficients instead of thousands of boundary nodes, and every candidate is a
valid simple closed curve as long as the radius stays positive. The cost is
that only star-shaped domains are reachable, so no holes and no deep
crescents.

What is matched is the vector of *ratios* ``sqrt(lambda_i / lambda_1)``, not
the raw eigenvalues. Scaling a drum moves every partial together, so ratios are
the part of the spectrum that shape alone controls. The overall size is then
free, and :func:`fit_scale` sets it to place the fundamental at a chosen pitch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.special import jn_zeros

from sonoform.geometry import FourierShape
from sonoform.spectrum import boundary_normal_derivative, solve_spectrum

__all__ = [
    "PPW_FREQUENCY_BOUND",
    "THIRD_PARTIAL_CEILING",
    "eigenvalue_jacobian",
    "Feasibility",
    "InverseResult",
    "check_feasibility",
    "fit_scale",
    "solve_inverse",
]

# Ashbaugh-Benguria, proving the Payne-Polya-Weinberger conjecture: for every
# bounded domain in the plane, lambda_2 / lambda_1 <= (j_1,1 / j_0,1)^2, with
# equality only for the disc. In frequency terms the second partial can never
# exceed the square root of that. This is why you cannot build a drum that
# plays an octave, and it is a theorem rather than a limit of this solver.
PPW_RATIO_BOUND = float(jn_zeros(1, 1)[0] ** 2 / jn_zeros(0, 1)[0] ** 2)
PPW_FREQUENCY_BOUND = float(np.sqrt(PPW_RATIO_BOUND))

# There is no comparable theorem for the third partial, but it is plainly
# bounded too. Maximising lambda_3 / lambda_1 over this solver's shape family
# from many random starts, at four, six and eight harmonics, converges to
# 1.78990, a lambda ratio of 3.20375, and will not go past it.
#
# That procedure is trustworthy because it reproduces the case we can check:
# run on the second partial it returns 1.59338 against a proven 1.59334, and
# the maximising shape it reports is a disc to within a three percent
# perturbation, which is exactly the uniqueness half of Ashbaugh-Benguria.
#
# Still, measured is not proved, so a target is given one percent of slack
# before being refused.
THIRD_PARTIAL_CEILING = 1.78990

# Frequency ratios of some chords, as multiples of the root, in just
# intonation. These are what a user actually asks for.
CHORDS: dict[str, tuple[float, ...]] = {
    "major": (1.0, 5 / 4, 3 / 2),
    "minor": (1.0, 6 / 5, 3 / 2),
    "diminished": (1.0, 6 / 5, 7 / 5),
    "augmented": (1.0, 5 / 4, 8 / 5),
    "major7": (1.0, 5 / 4, 3 / 2, 15 / 8),
    "minor7": (1.0, 6 / 5, 3 / 2, 9 / 5),
    "octaves": (1.0, 2.0, 3.0, 4.0),
    "fifths": (1.0, 3 / 2, 9 / 4),
}


@dataclass(frozen=True)
class Feasibility:
    """Whether a target spectrum is achievable by any shape at all.

    Attributes
    ----------
    achievable
        False when a target violates a proven bound, in which case no amount
        of searching will help.
    reason
        Plain-language explanation, empty when achievable.
    nearest
        The closest achievable target, obtained by clamping the offending
        partials to the bound.
    """

    achievable: bool
    reason: str
    nearest: np.ndarray

    def __bool__(self) -> bool:
        return self.achievable


def check_feasibility(target) -> Feasibility:
    """Test a target against the Payne-Polya-Weinberger bound.

    The second partial of any planar drum is capped at
    :data:`PPW_FREQUENCY_BOUND`, about 1.5933, which the disc alone attains.
    A target asking for more is impossible, not merely hard.

    Examples
    --------
    >>> check_feasibility([1.0, 2.0, 3.0]).achievable
    False
    >>> check_feasibility([1.0, 1.25, 1.5]).achievable
    True
    """
    t = np.asarray(target, dtype=float)

    if t.size >= 2 and t[1] > PPW_FREQUENCY_BOUND:
        semitones = 12.0 * np.log2(t[1] / PPW_FREQUENCY_BOUND)
        nearest = t.copy()
        nearest[1] = PPW_FREQUENCY_BOUND
        nearest[2:] = np.maximum(nearest[2:], nearest[1])
        return Feasibility(
            achievable=False,
            reason=(
                f"the second partial asks for {t[1]:.4f} but no planar drum "
                f"can exceed {PPW_FREQUENCY_BOUND:.4f}, which only the disc "
                f"attains. The request is {semitones:.1f} semitones too wide. "
                f"This is the Ashbaugh-Benguria theorem, not a solver limit."
            ),
            nearest=nearest,
        )

    if t.size >= 3 and t[2] > THIRD_PARTIAL_CEILING * 1.01:
        semitones = 12.0 * np.log2(t[2] / THIRD_PARTIAL_CEILING)
        nearest = t.copy()
        nearest[2] = THIRD_PARTIAL_CEILING
        nearest[3:] = np.maximum(nearest[3:], nearest[2])
        return Feasibility(
            achievable=False,
            reason=(
                f"the third partial asks for {t[2]:.4f}, and searching this "
                f"shape family from many starts never gets past "
                f"{THIRD_PARTIAL_CEILING:.4f}. The request is "
                f"{semitones:.1f} semitones too wide. Unlike the second-partial "
                f"bound this ceiling is measured rather than proved, so treat "
                f"it as a strong indication rather than a theorem."
            ),
            nearest=nearest,
        )

    return Feasibility(achievable=True, reason="", nearest=t.copy())


@dataclass
class InverseResult:
    """Outcome of an inverse solve.

    Attributes
    ----------
    shape
        Best shape found.
    achieved
        Frequency ratios its spectrum actually produces, measured on a mesh
        far finer than the one used during the search so the number is not
        the optimiser marking its own homework.
    target
        Ratios that were asked for.
    residual
        Root-mean-square relative error between the two, in cents per the
        :attr:`cents` property.
    success
        Whether the optimiser converged and the shape is valid.
    n_evaluations
        Number of forward solves used.
    history
        Residual after each accepted step, for plotting convergence.
    """

    shape: FourierShape
    achieved: np.ndarray
    target: np.ndarray
    residual: float
    success: bool
    n_evaluations: int
    history: list[float] = field(default_factory=list)

    @property
    def cents(self) -> np.ndarray:
        """Per-partial error in cents. 100 cents is one semitone.

        Musically, under about 5 cents is inaudible to most listeners and
        under 15 is acceptable; anything past 50 is a different note.
        """
        return 1200.0 * np.log2(self.achieved / self.target)

    @property
    def max_cents(self) -> float:
        return float(np.abs(self.cents).max())

    def __repr__(self) -> str:
        return (
            f"InverseResult(success={self.success}, "
            f"max_error={self.max_cents:.1f} cents, "
            f"evaluations={self.n_evaluations})"
        )


def _ratios(shape: FourierShape, n: int, n_radial: int, n_angular: int, order: int):
    """Frequency ratios of a candidate shape, or None if it is degenerate."""
    if not shape.is_valid():
        return None
    try:
        spec = solve_spectrum(
            shape.mesh(n_radial=n_radial, n_angular=n_angular), k=n, order=order
        )
    except (ValueError, RuntimeError):
        return None
    return spec.ratios


def solve_inverse(
    target,
    n_harmonics: int = 5,
    n_radial: int = 7,
    n_angular: int = 80,
    order: int = 1,
    max_nfev: int = 260,
    seed: int = 0,
    n_restarts: int = 3,
    polish_radial: int = 8,
    polish_angular: int = 200,
    max_polish_nfev: int = 90,
) -> InverseResult:
    """Find a star-shaped domain whose partials match ``target``.

    Parameters
    ----------
    target
        Either a chord name from :data:`CHORDS`, or a sequence of frequency
        ratios beginning with 1.0, for example ``(1, 1.25, 1.5)`` for a major
        triad.
    n_harmonics
        Number of Fourier harmonics in the boundary. More harmonics fit better
        but produce wigglier shapes and a harder optimisation.

        Three partials are comfortably matched with four harmonics. Four
        partials want five or six: a minor seventh lands within about two
        cents at six harmonics with three restarts, where four harmonics
        leaves it near four.
    n_radial, n_angular
        Mesh resolution for the coarse search. Discretisation error here only
        has to be small enough to put the optimiser in the right basin, since
        the polish stage re-solves properly.
    order
        Element order inside the loop. Order 1 is about four times faster and
        the ratios are accurate enough to optimise against.
    max_nfev
        Cap on forward solves per restart.
    seed
        Seeds the random restarts, so a run is reproducible.
    n_restarts
        Number of random starting shapes. The objective is non-convex and a
        single start lands in a poor local minimum often enough to matter.
    polish_radial, polish_angular, max_polish_nfev
        Resolution and budget for the final high-fidelity refinement. See the
        note in the body about why this stage is not optional.

        The defaults are deliberately lopsided. On a curved boundary the
        dominant error is that the mesh is an inscribed polygon, which is
        governed by ``n_angular`` alone: holding the radial count at 8 and
        raising the angular count from 128 to 200 takes the error from 1.67 to
        0.86 cents, while raising the radial count from 8 to 16 at fixed
        angular buys 0.26 cents for two and a half times the cost. Spend the
        budget on the boundary.

    Returns
    -------
    InverseResult

    Notes
    -----
    Not every feasible-looking chord is reachable. A major seventh,
    ``(1, 5/4, 3/2, 15/8)``, stalls around fifteen cents no matter how many
    harmonics or restarts it is given, even though each of its partials is
    individually attainable: maximised on its own the fourth partial reaches
    2.14, well past the 1.875 it asks for. The obstruction is joint rather
    than per-partial, and this solver does not currently detect that case in
    advance the way :func:`check_feasibility` detects a violated PPW bound.
    Check :attr:`InverseResult.max_cents` rather than assuming success.
    """
    if isinstance(target, str):
        if target not in CHORDS:
            raise ValueError(f"unknown chord {target!r}; known: {sorted(CHORDS)}")
        target_ratios = np.asarray(CHORDS[target], dtype=float)
    else:
        target_ratios = np.asarray(target, dtype=float)

    if target_ratios.ndim != 1 or target_ratios.size < 2:
        raise ValueError("target must be a sequence of at least two ratios")
    if not np.isclose(target_ratios[0], 1.0):
        raise ValueError("the first target ratio must be 1.0, the fundamental")
    if np.any(np.diff(target_ratios) <= 0):
        raise ValueError("target ratios must be strictly increasing")

    feasible = check_feasibility(target_ratios)
    if not feasible:
        raise ValueError(feasible.reason)

    n_modes = target_ratios.size
    rng = np.random.default_rng(seed)
    counter = {"n": 0}
    history: list[float] = []

    def residuals(v):
        counter["n"] += 1
        got = _ratios(FourierShape.from_vector(v), n_modes, n_radial, n_angular, order)
        if got is None:
            return np.full(n_modes - 1, 10.0)
        # work in log space so the error is musical rather than arithmetic
        res = np.log(got[1:] / target_ratios[1:])
        history.append(float(np.sqrt(np.mean(res**2))))
        return res

    best = None
    for restart in range(max(1, n_restarts)):
        v0 = np.zeros(2 * n_harmonics + 1)
        v0[0] = 1.0
        if restart > 0:
            # perturb the harmonics, leaving the mean radius alone
            v0[1:] = rng.normal(scale=0.12, size=2 * n_harmonics)

        # keep the mean radius near 1, bound harmonics so the radius stays > 0
        lo = np.concatenate([[0.5], np.full(2 * n_harmonics, -0.45)])
        hi = np.concatenate([[1.5], np.full(2 * n_harmonics, 0.45)])

        out = least_squares(
            residuals,
            v0,
            bounds=(lo, hi),
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            diff_step=1e-3,
        )
        rms = float(np.sqrt(np.mean(out.fun**2)))
        if best is None or rms < best[0]:
            best = (rms, out.x)

    rms, v_best = best

    # Polish at the resolution the answer will actually be judged at.
    #
    # The loop above runs on a coarse mesh with linear elements for speed, and
    # left there the optimiser drives the residual to zero by exploiting
    # discretisation error rather than by finding a genuinely correct shape.
    # Measured on a major triad that self-deception was worth about 38 cents,
    # which is a third of a semitone and plainly audible. Re-solving from the
    # coarse optimum against quadratic elements on a finer mesh removes it.
    def fine_residuals(v):
        counter["n"] += 1
        got = _ratios(
            FourierShape.from_vector(v),
            n_modes,
            polish_radial,
            polish_angular,
            2,
        )
        if got is None:
            return np.full(n_modes - 1, 10.0)
        return np.log(got[1:] / target_ratios[1:])

    def fine_jacobian(v):
        """d(residual)/d(coefficients), analytically where that is valid.

        residual_i = log(sqrt(lambda_i / lambda_0)) - log(target_i), so
        d residual_i / d c = (dlambda_i / lambda_i - dlambda_0 / lambda_0) / 2.
        """
        shape = FourierShape.from_vector(v)
        if shape.is_valid():
            try:
                lam, jac, trustworthy = eigenvalue_jacobian(
                    shape, n_modes, polish_radial, polish_angular, order=2
                )
            except (ValueError, RuntimeError):
                trustworthy = False
            if trustworthy:
                counter["n"] += 1
                d_log = jac / lam[:, None]
                return 0.5 * (d_log[1:] - d_log[0][None, :])
        # Near a degeneracy the shape derivative is undefined, so pay for
        # finite differences on this step rather than return nonsense.
        return _finite_difference_jacobian(fine_residuals, v, n_modes - 1)

    lo = np.concatenate([[0.5], np.full(2 * n_harmonics, -0.45)])
    hi = np.concatenate([[1.5], np.full(2 * n_harmonics, 0.45)])
    polished = least_squares(
        fine_residuals,
        np.clip(v_best, lo, hi),
        jac=fine_jacobian,
        bounds=(lo, hi),
        max_nfev=max_polish_nfev,
        xtol=1e-12,
        ftol=1e-12,
    )
    rms = float(np.sqrt(np.mean(polished.fun**2)))
    history.append(rms)

    shape = FourierShape.from_vector(polished.x)

    # Report the answer measured on a mesh far finer than the one it was
    # optimised against. Reading it off the polish mesh gives 0.0 cents on a
    # major triad, which is the optimiser grading its own work: it drove the
    # residual to zero there, so of course it looks perfect there.
    achieved = _ratios(shape, n_modes, _CHECK_RADIAL, _CHECK_ANGULAR, 2)
    if achieved is None:
        achieved = np.full(n_modes, np.nan)
    honest = float(np.sqrt(np.mean(np.log(achieved[1:] / target_ratios[1:]) ** 2)))

    return InverseResult(
        shape=shape,
        achieved=achieved,
        target=target_ratios,
        residual=honest,
        success=bool(np.isfinite(honest) and honest < 0.003 and shape.is_valid()),
        n_evaluations=counter["n"],
        history=history,
    )


_CHECK_RADIAL = 20
_CHECK_ANGULAR = 400


def verify(
    result: InverseResult,
    n_radial: int = _CHECK_RADIAL,
    n_angular: int = _CHECK_ANGULAR,
):
    """Recompute a result's ratios on a finer mesh with quadratic elements.

    The optimiser runs at low resolution for speed. This confirms the answer
    survives a properly resolved solve rather than being an artefact of a
    coarse mesh.

    The angular count here is deliberately far above anything the solver uses.
    A check that shares the solver's own discretisation shares its bias too,
    and will happily confirm an answer that is wrong in the same direction.
    """
    spec = solve_spectrum(
        result.shape.mesh(n_radial=n_radial, n_angular=n_angular),
        k=result.target.size,
        order=2,
    )
    return spec.ratios


def fit_scale(shape: FourierShape, fundamental_hz: float, wave_speed: float = 100.0):
    """Scale factor putting the fundamental at ``fundamental_hz``.

    For an ideal membrane ``f = c sqrt(lambda) / (2 pi)`` where ``c`` is the
    wave speed set by tension and density. Scaling the shape by ``s`` divides
    every eigenvalue by ``s**2``, so the fundamental moves as ``1 / s``.
    """
    spec = solve_spectrum(shape.mesh(), k=1, order=2)
    current = wave_speed * np.sqrt(spec.eigenvalues[0]) / (2.0 * np.pi)
    return float(current / fundamental_hz)


def _finite_difference_jacobian(residual_fn, v, n_res, step=1e-5):
    """Central-difference fallback for when the analytic formula does not hold."""
    jac = np.zeros((n_res, v.size))
    for j in range(v.size):
        vp, vm = v.copy(), v.copy()
        vp[j] += step
        vm[j] -= step
        jac[:, j] = (residual_fn(vp) - residual_fn(vm)) / (2.0 * step)
    return jac


def _radius_derivative_basis(shape: FourierShape, theta):
    """``d r / d c`` at ``theta`` for each coefficient, in vector order."""
    rows = [np.ones_like(theta)]
    for k in range(1, shape.cos_coeffs.size + 1):
        rows.append(np.cos(k * theta))
    for k in range(1, shape.sin_coeffs.size + 1):
        rows.append(np.sin(k * theta))
    return rows


def _dr_dtheta(shape: FourierShape, theta):
    out = np.zeros_like(theta)
    for k, a in enumerate(shape.cos_coeffs, start=1):
        out = out - k * a * np.sin(k * theta)
    for k, b in enumerate(shape.sin_coeffs, start=1):
        out = out + k * b * np.cos(k * theta)
    return out


def eigenvalue_jacobian(shape: FourierShape, n_modes, n_radial, n_angular, order=2):
    """``d lambda_i / d c_j`` by the Hadamard shape derivative.

    One forward solve yields the whole Jacobian, where finite differences need
    one solve per coefficient. For the nine-coefficient default that is close
    to a tenfold saving on the dominant cost.

    For a polar boundary the geometry collapses pleasantly: a radial
    perturbation ``dr`` has normal component ``dr * r / sqrt(r^2 + r'^2)``
    while the arclength element carries ``sqrt(r^2 + r'^2) dtheta``, so the
    two factors cancel and ``V_n ds = dr * r dtheta``.

    Returns ``(eigenvalues, jacobian, trustworthy)``. The last flag is False
    when any of the requested eigenvalues is close to a neighbour, because the
    formula assumes a simple eigenvalue and gives nonsense at a crossing.
    """
    spec = solve_spectrum(
        shape.mesh(n_radial=n_radial, n_angular=n_angular), k=n_modes, order=order
    )
    trustworthy = bool(np.all(spec.separations() > 0.02))

    jac = np.zeros((n_modes, shape.to_vector().size))
    for i in range(n_modes):
        dudn2, weights, theta = boundary_normal_derivative(spec, i)
        r = shape.radius(theta)
        normal_factor = r / np.hypot(r, _dr_dtheta(shape, theta))
        common = dudn2 * weights * normal_factor
        for j, dr in enumerate(_radius_derivative_basis(shape, theta)):
            jac[i, j] = -np.sum(common * dr)
    return spec.eigenvalues, jac, trustworthy
