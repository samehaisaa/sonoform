"""Dirichlet Laplacian eigenproblem on a 2-D domain.

Solves ``-laplace(u) = lambda u`` with ``u = 0`` on the boundary. The
eigenvalues are what the domain rings at; the zero sets of the eigenfunctions
are the Chladni figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skfem import (
    Basis,
    ElementTriP1,
    ElementTriP2,
    FacetBasis,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import dot, grad
from skfem.models.poisson import laplace, mass
from skfem.utils import solver_eigen_scipy_sym

__all__ = ["Spectrum", "boundary_normal_derivative", "solve_spectrum"]

_ELEMENTS = {1: ElementTriP1, 2: ElementTriP2}


@dataclass(frozen=True)
class Spectrum:
    """Eigenvalues and eigenfunctions of a domain.

    Attributes
    ----------
    eigenvalues
        Ascending array of ``lambda_i``, length ``k``.
    eigenvectors
        Column ``i`` is ``u_i`` sampled at the basis degrees of freedom.
    mesh
        The mesh the problem was solved on.
    order
        Polynomial order of the element used.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    mesh: MeshTri
    order: int

    @property
    def frequencies(self) -> np.ndarray:
        """Wavenumbers ``sqrt(lambda)``, in units where the wave speed is 1.

        Not hertz, and not angular frequency either. For an ideal membrane the
        angular frequency is ``c * sqrt(lambda)`` and the audible frequency is
        ``c * sqrt(lambda) / (2 * pi)``, where ``c = sqrt(T / rho)`` is set by
        tension and density. Use :func:`~sonoform.inverse.fit_scale` to place a
        spectrum at an absolute pitch.
        """
        return np.sqrt(self.eigenvalues)

    @property
    def ratios(self) -> np.ndarray:
        """Partials relative to the fundamental.

        This is the scale-free fingerprint of the shape. Making a drum bigger
        or smaller moves every frequency together and leaves these untouched,
        so it is what you compare against a target chord.
        """
        return self.frequencies / self.frequencies[0]

    def nodal_values(self, index: int) -> np.ndarray:
        """Eigenfunction ``index`` sampled at mesh vertices, scaled to [-1, 1].

        P2 solutions carry edge degrees of freedom too; only the vertex values
        are returned, which is what a triangulation-based contour plot needs.
        """
        u = self.eigenvectors[: self.mesh.p.shape[1], index]
        peak = np.abs(u).max()
        return u / peak if peak > 0 else u

    def __len__(self) -> int:
        return self.eigenvalues.size

    def separations(self) -> np.ndarray:
        """Relative gap from each eigenvalue to its nearest neighbour.

        A shape derivative is only defined where an eigenvalue is simple. At a
        degeneracy the eigenvalue is continuous but not differentiable, and
        the Hadamard formula silently returns nonsense. Callers use this to
        decide whether to trust an analytic gradient.
        """
        lam = self.eigenvalues
        gaps = np.full(lam.size, np.inf)
        if lam.size > 1:
            gaps[:-1] = np.minimum(gaps[:-1], np.diff(lam))
            gaps[1:] = np.minimum(gaps[1:], np.diff(lam))
        return gaps / np.maximum(np.abs(lam), 1e-30)


def solve_spectrum(mesh: MeshTri, k: int = 8, order: int = 2) -> Spectrum:
    """Compute the first ``k`` Dirichlet eigenpairs of ``mesh``.

    Parameters
    ----------
    mesh
        Triangulation of the domain.
    k
        Number of eigenpairs, ascending from the fundamental.
    order
        Element order, 1 or 2. Order 2 is roughly two digits more accurate for
        the same mesh and is the default; order 1 is faster and is enough
        inside an optimisation loop.

    Notes
    -----
    On a polygon the result is limited only by discretisation, around 3e-06
    relative on the first eight modes at the default resolution. On a curved
    boundary there is an additional bias because the triangulated domain is
    slightly smaller than the true one; that error equals the relative area
    deficit and falls as ``O(h^2)``.
    """
    if order not in _ELEMENTS:
        raise ValueError(f"order must be 1 or 2, got {order}")
    if k < 1:
        raise ValueError("k must be at least 1")

    basis = Basis(mesh, _ELEMENTS[order]())
    boundary_dofs = basis.get_dofs()
    n_interior = basis.N - boundary_dofs.flatten().size
    if k >= n_interior:
        raise ValueError(
            f"asked for {k} eigenvalues but the mesh has only {n_interior} "
            f"interior degrees of freedom; refine the mesh or lower k"
        )

    stiffness = laplace.assemble(basis)
    mass_matrix = mass.assemble(basis)

    # ARPACK picks a random starting vector when none is given, so two
    # identical calls can return eigenvectors that differ in sign and
    # eigenvalues that differ in the last few digits. Inside an optimiser that
    # is enough to send the same seed down a different path, so pin it.
    condensed = condense(stiffness, mass_matrix, D=boundary_dofs)
    # A constant vector is reproducible but a poor Krylov start: it is nearly
    # orthogonal to the antisymmetric modes, and using one cost a factor of
    # two and a half in wall time. A fixed pseudo-random vector is just as
    # reproducible and converges like the random start ARPACK would have
    # chosen for itself.
    v0 = np.random.default_rng(0).standard_normal(condensed[0].shape[0])
    lams, vecs = solve(
        *condensed,
        solver=solver_eigen_scipy_sym(k=k, sigma=0.0, v0=v0),
    )

    order_idx = np.argsort(lams)
    return Spectrum(
        eigenvalues=np.asarray(lams)[order_idx],
        eigenvectors=np.asarray(vecs)[:, order_idx],
        mesh=mesh,
        order=order,
    )


def boundary_normal_derivative(spec: Spectrum, index: int):
    """Squared normal derivative of an eigenfunction on the boundary.

    Returns ``(dudn_squared, weights, theta)`` sampled at the boundary
    quadrature points, where ``weights`` already carries the arclength
    measure. This is the kernel of the Hadamard shape derivative

    ``dlambda = -integral (du/dn)^2 V_n ds``

    which holds for a simple eigenvalue with an eigenfunction normalised so
    that ``integral u^2 = 1``. The generalised solver returns mass-orthonormal
    vectors, which is exactly that normalisation.
    """
    mesh = spec.mesh
    element = _ELEMENTS[spec.order]()
    fb = FacetBasis(mesh, element, facets=mesh.boundary_facets())
    u = spec.eigenvectors[:, index]
    gradient = fb.interpolate(u)
    dudn = dot(grad(gradient), fb.normals)
    xq = np.asarray(fb.global_coordinates())
    return np.asarray(dudn) ** 2, np.asarray(fb.dx), np.arctan2(xq[1], xq[0])
