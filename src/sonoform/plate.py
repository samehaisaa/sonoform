"""Free Kirchhoff plate.

A Chladni plate is not a membrane. The mid-surface deflection satisfies

``D ∇⁴ w = ρ h ω² w``

with the bending moment and the Kirchhoff shear left free on the edge.
Those are the natural boundary conditions of the bending energy, so a
conforming element carries them without constraints on the boundary.

The element is the Argyris triangle, which is C¹. A completely free plate
has a three-dimensional kernel of rigid motions, heave and two rotations.
Those eigenvalues sit at zero and are discarded.

Frequencies scale with thickness and with the inverse square of the size.
Poisson's ratio does not: it stays inside the bilinear form, so the
partials of a plate depend on shape and on ``ν``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import eigsh
from skfem import Basis, BilinearForm, ElementTriArgyris, MeshTri
from skfem.helpers import dd, ddot, trace

from sonoform.geometry import is_simple_polygon, polygon_mesh, resample_closed_path

__all__ = [
    "BRASS",
    "SPAN",
    "PlateMaterial",
    "PlateSpectrum",
    "display_field",
    "outline_mesh",
    "solve_plate",
    "square_mesh",
]

# A hand-sized brass sheet. Thickness over span is about 1/90, which is
# the thin-plate regime this model is the theory of.
SPAN = 0.18
_NREFS = 3


@dataclass(frozen=True)
class PlateMaterial:
    """Homogeneous isotropic sheet.

    ``loss_factor`` is the structural loss factor ``η`` in the amplitude
    envelope ``exp(-π η f t)``. It is a material constant, not a tuning knob
    for brightness.
    """

    young: float
    poisson: float
    density: float
    thickness: float
    loss_factor: float

    @property
    def flexural_rigidity(self) -> float:
        nu = self.poisson
        return self.young * self.thickness**3 / (12.0 * (1.0 - nu * nu))

    @property
    def areal_density(self) -> float:
        return self.density * self.thickness


BRASS = PlateMaterial(
    young=100.0e9,
    poisson=0.34,
    density=8500.0,
    thickness=0.002,
    loss_factor=1.0e-3,
)


@dataclass
class PlateSpectrum:
    """Flexible modes of a free plate.

    ``eigenvalues`` are ``ω²`` in radians squared per second squared, ascending,
    with the three rigid-body zeros removed. Column ``i`` of ``eigenvectors``
    is the Argyris coefficient vector of mode ``i``, mass-normalised so that
    ``∫ ρ h φ² = 1``.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    kernel: np.ndarray
    mesh: MeshTri
    material: PlateMaterial
    basis: Basis

    @property
    def frequencies(self) -> np.ndarray:
        """Cyclic frequencies in hertz."""
        return np.sqrt(np.maximum(self.eigenvalues, 0.0)) / (2.0 * np.pi)

    def vertex_values(self, index: int) -> np.ndarray:
        """Deflection at mesh vertices, sign-fixed and scaled to [-1, 1]."""
        values = self._vertex_component(index, 0)
        return _orient(values)

    def vertex_gradients(self, index: int) -> np.ndarray:
        """``(2, nvertices)`` gradient of the same oriented mode.

        The orientation matches :meth:`vertex_values`, so a browser can
        rebuild ``∇(A²) = 2 A ∇A`` without a second sign convention.
        """
        values = self._vertex_component(index, 0)
        grad = np.vstack(
            [self._vertex_component(index, 1), self._vertex_component(index, 2)]
        )
        peak = np.abs(values).max()
        sign = 1.0 if peak == 0.0 or values[np.argmax(np.abs(values))] >= 0.0 else -1.0
        scale = sign / peak if peak > 0.0 else 1.0
        return grad * scale

    def _vertex_component(self, index: int, slot: int) -> np.ndarray:
        # Argyris nodal dofs are stored per vertex, Fortran order:
        # vertex i holds (u, u_x, u_y, u_xx, u_xy, u_yy) at 6*i + slot.
        n_vertices = self.mesh.p.shape[1]
        nodal = self.eigenvectors[: 6 * n_vertices, index]
        return np.asarray(nodal.reshape((6, n_vertices), order="F")[slot])

    def __len__(self) -> int:
        return int(self.eigenvalues.size)


def _orient(values: np.ndarray) -> np.ndarray:
    peak = np.abs(values).max()
    if peak == 0.0:
        return values.copy()
    signed = values if values[np.argmax(np.abs(values))] >= 0.0 else -values
    return signed / peak


def square_mesh(span: float = SPAN, nrefs: int = _NREFS) -> MeshTri:
    """A square of side ``span``, centred at the origin."""
    return MeshTri().refined(nrefs).scaled(span).translated([-0.5 * span, -0.5 * span])


def outline_mesh(points, span: float = SPAN, max_triangles: int = 220):
    """Resample a closed outline and mesh it at a fixed physical span.

    The longest side of the bounding box becomes ``span``. Returns
    ``(mesh, outline)`` with ``outline`` shaped ``(2, n)``.
    """
    path = resample_closed_path(points, n=140)
    if not is_simple_polygon(path):
        raise ValueError("the outline crosses itself")
    path = path - path.mean(axis=1, keepdims=True)
    extent = float(np.ptp(path, axis=1).max())
    if extent <= 0.0:
        raise ValueError("the outline has no extent")
    path = path * (span / extent)
    # Argyris is quintic, so a few hundred triangles already resolve the
    # first modes. Finer than this spends the wait on the eigensolve.
    area = 0.5 * abs(
        np.sum(path[0] * np.roll(path[1], -1) - np.roll(path[0], -1) * path[1])
    )
    mesh = polygon_mesh(path, max_area=area / max_triangles)
    return mesh, path


def solve_plate(
    mesh: MeshTri,
    material: PlateMaterial = BRASS,
    k: int = 8,
) -> PlateSpectrum:
    """The first ``k`` flexible modes of a completely free plate.

    Parameters
    ----------
    mesh
        Mid-surface triangulation, in metres.
    material
        Thickness, stiffness, density, Poisson ratio, loss factor.
    k
        How many flexible modes to keep, after the rigid-body kernel.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not 0.0 <= material.poisson < 0.5:
        raise ValueError("poisson ratio must lie in [0, 0.5)")

    basis = Basis(mesh, ElementTriArgyris(), intorder=6)
    stiffness, mass = _assemble(basis, material)
    n_requested = k + 3
    if n_requested >= stiffness.shape[0]:
        raise ValueError(
            f"asked for {k} flexible modes but the mesh only has "
            f"{stiffness.shape[0]} degrees of freedom"
        )

    # K is singular: the three rigid motions cost no bending energy.
    # K + α M is positive definite, its eigenvalues are ω² + α, and
    # subtracting α recovers the plate frequencies including the zeros.
    #
    # α has to sit near the bottom of the flexible spectrum rather than far
    # below it. Shift-invert maps the rigid modes to 1/α and the first
    # flexible mode to 1/(ω₁² + α). A fixed small fraction of D/ρh, which is
    # what this used to be, lets the rigid modes outweigh the flexible ones in
    # the inverted operator by six to ten orders of magnitude, set by nothing
    # but the units of the mesh, and ARPACK then returns flexible modes a
    # percent out without reporting anything: the triangle meshed at a span
    # of one metre came back at 332.5 Hz for a mode whose discrete value, by
    # a dense solve, is 336.7. Kirchhoff's equation puts ω₁² at
    # (D / ρh)(Ω / L²)² with Ω of order ten for any free plate, so that is
    # the shift, and the solve no longer depends on the units.
    scale = float(material.flexural_rigidity / material.areal_density)
    extent = float(np.ptp(mesh.p, axis=1).max())
    alpha = scale * (10.0 / extent**2) ** 2
    shifted = (stiffness + alpha * mass).tocsc()
    v0 = np.random.default_rng(0).standard_normal(shifted.shape[0])
    mus, vecs = eigsh(
        shifted,
        k=n_requested,
        M=mass.tocsc(),
        sigma=0.0,
        which="LM",
        v0=v0,
        tol=1e-9,
    )
    omega2 = np.asarray(mus, dtype=float) - alpha
    order = np.argsort(omega2)
    omega2 = omega2[order]
    vecs = np.asarray(vecs, dtype=float)[:, order]

    kernel = omega2[:3]
    flexible = omega2[3:]
    if flexible.size == 0 or kernel.max() > 1e-4 * flexible[0]:
        raise RuntimeError(
            "the rigid-body kernel did not separate from the flexible modes"
        )
    # Mass-orthonormal from eigsh, up to a sign. Nothing else to do.
    return PlateSpectrum(
        eigenvalues=flexible[:k].copy(),
        eigenvectors=vecs[:, 3 : 3 + k].copy(),
        kernel=kernel.copy(),
        mesh=mesh,
        material=material,
        basis=basis,
    )


def _assemble(basis: Basis, material: PlateMaterial):
    rigidity = material.flexural_rigidity
    nu = material.poisson
    areal = material.areal_density

    @BilinearForm
    def bending(u, v, w):
        hu, hv = dd(u), dd(v)
        return rigidity * (nu * trace(hu) * trace(hv) + (1.0 - nu) * ddot(hu, hv))

    @BilinearForm
    def inertia(u, v, w):
        return areal * u * v

    return bending.assemble(basis), inertia.assemble(basis)


def display_field(spec: PlateSpectrum, level: int = 3):
    """Resample the modes onto a finer mesh, for drawing only.

    Argyris is a quintic, so the solution is far smoother than the triangles
    it was solved on. Drawing straight from the solve mesh throws that away:
    128 triangles give a visibly faceted sheet and a nodal contour made of
    obvious straight segments. Subdividing each triangle in barycentric
    coordinates and evaluating the basis at the new points recovers the
    smoothness that was already computed, with no second eigensolve.

    Vertices shared between neighbouring triangles are merged, so the contour
    does not break at the seams and the payload stays small.

    Returns ``(points, triangles, values, gradients)`` where ``points`` is
    ``(2, n)``, ``triangles`` is ``(3, m)``, ``values`` is ``(k, n)`` and
    ``gradients`` is ``(k, 2, n)``, all oriented the same way as
    :meth:`PlateSpectrum.vertex_values`.
    """
    if level < 1:
        raise ValueError("level must be at least 1")

    mesh = spec.mesh
    corners = mesh.p[:, mesh.t]  # (2, 3, ntri)

    # barycentric lattice on the reference triangle
    lattice = [
        (i / level, j / level, (level - i - j) / level)
        for i in range(level + 1)
        for j in range(level + 1 - i)
    ]
    index_of = {(i, j): n for n, (i, j) in enumerate(
        [(i, j) for i in range(level + 1) for j in range(level + 1 - i)]
    )}

    sub_tris = []
    for i in range(level):
        for j in range(level - i):
            a = index_of[(i, j)]
            b = index_of[(i + 1, j)]
            c = index_of[(i, j + 1)]
            sub_tris.append((a, b, c))
            if i + j < level - 1:
                d = index_of[(i + 1, j + 1)]
                sub_tris.append((b, d, c))

    bary = np.array(lattice, dtype=float).T  # (3, npts)
    # (2, npts, ntri): every lattice point of every triangle
    pts = np.einsum("dct,cp->dpt", corners, bary)

    # Lattice points on an element edge sit exactly on the boundary, and on a
    # curved outline rounding can put them a hair outside the mesh, which
    # makes skfem's point location refuse them outright. Probing is therefore
    # done at points pulled a millionth of an element toward their own
    # centroid, which is strictly interior. The shift is far below anything
    # the quintic resolves, and the merged coordinates stay exact so the
    # output mesh and the dedupe are unaffected.
    inset = 1e-6
    bary_in = (1.0 - inset) * bary + inset / 3.0
    pts_in = np.einsum("dct,cp->dpt", corners, bary_in)

    n_tri = mesh.t.shape[1]
    flat = pts.reshape(2, -1, order="F")
    flat_in = pts_in.reshape(2, -1, order="F")

    # merge coincident points; the lattice is exact on shared edges, so
    # rounding at a scale far below the element size is safe
    scale = float(np.ptp(mesh.p, axis=1).max())
    keys = np.round(flat / (1e-9 * scale)).astype(np.int64)
    _, first, inverse = np.unique(keys.T, axis=0, return_index=True,
                                  return_inverse=True)
    points = flat[:, first]
    inverse = np.asarray(inverse).ravel()

    per_tri = bary.shape[1]
    triangles = []
    for t in range(n_tri):
        base = t * per_tri
        for a, b, c in sub_tris:
            triangles.append(
                (inverse[base + a], inverse[base + b], inverse[base + c])
            )
    triangles = np.array(triangles, dtype=np.int64).T

    probe = spec.basis.probes(flat_in[:, first])
    values, gradients = [], []
    for index in range(len(spec)):
        coeffs = spec.eigenvectors[:, index]
        raw = np.asarray(probe @ coeffs)
        peak = np.abs(raw).max()
        # The sign comes from the vertices, by the same rule vertex_values
        # uses, so the two always agree. Taking it from the finer points
        # instead fails on an antisymmetric mode, whose largest values come in
        # equal and opposite pairs and leave the choice of sign to rounding.
        vertex = spec._vertex_component(index, 0)
        sign = 1.0 if vertex[np.argmax(np.abs(vertex))] >= 0.0 else -1.0
        scale_v = sign / peak if peak > 0.0 else 1.0
        values.append(raw * scale_v)
        # gradients come from the coarse nodal dofs, interpolated the same way
        coarse_grad = spec.vertex_gradients(index)
        gradients.append(_interpolate_gradient(mesh, coarse_grad, bary, inverse,
                                               points.shape[1], per_tri))
    return points, triangles, np.array(values), np.array(gradients)


def _interpolate_gradient(mesh, coarse, bary, inverse, n_points, per_tri):
    """Linear interpolation of a per-vertex gradient onto the finer points."""
    out = np.zeros((2, n_points))
    seen = np.zeros(n_points, dtype=bool)
    per_corner = coarse[:, mesh.t]  # (2, 3, ntri)
    values = np.einsum("dct,cp->dpt", per_corner, bary)
    flat = values.reshape(2, -1, order="F")
    for slot in range(flat.shape[1]):
        target = inverse[slot]
        if not seen[target]:
            out[:, target] = flat[:, slot]
            seen[target] = True
    return out
