"""What the page receives: the solved plates, as exact quintics per element."""

import base64

import numpy as np
import pytest

from sonoform.export import (
    CLUSTER_GAP,
    EXPONENTS,
    boundary_loop,
    degenerate_clusters,
    manifest,
    payload_from_spectrum,
    plate_mesh,
    plate_payload,
    poisson_ratios,
    quintic_coefficients,
    unit_material,
    verification,
)
from sonoform.materials import MATERIALS
from sonoform.plate import BRASS, SPAN, solve_plate, square_mesh
from sonoform.presets import PRESETS


def _floats(text):
    return np.frombuffer(base64.b64decode(text), dtype="<f4")


def _index(payload, name):
    kind = "<u2" if payload["index"] == "u16" else "<u4"
    return np.frombuffer(base64.b64decode(payload[name]), dtype=kind)


def _evaluate(frames, coeffs, element, points):
    cx, cy, s = frames[element].T
    xi = (points[0] - cx) / s
    eta = (points[1] - cy) / s
    mono = np.stack([xi**a * eta**b for a, b in EXPONENTS], axis=-1)
    return np.einsum("pm,kpm->pk", mono, coeffs[:, element, :])


@pytest.mark.parametrize("key", ["square", "circle"])
def test_the_quintics_are_the_solution_not_a_fit_to_it(key):
    """Evaluated anywhere inside an element, they give what the basis gives.

    A fit that merely passed through its 21 lattice points could wander
    between them. The Argyris field is a quintic on each triangle, so the fit
    through the degree-five lattice is exact and agrees everywhere.
    """
    spec = solve_plate(plate_mesh(key), unit_material(0.34), k=8)
    frames, coeffs, peaks = quintic_coefficients(spec)
    mesh = spec.mesh
    rng = np.random.default_rng(3)
    element = rng.integers(0, mesh.t.shape[1], 300)
    bary = rng.dirichlet([1.0, 1.0, 1.0], 300).T
    points = np.einsum("dct,ct->dt", mesh.p[:, mesh.t[:, element]], bary)
    want = np.asarray(spec.basis.probes(points) @ spec.eigenvectors) / peaks
    got = _evaluate(frames, coeffs, element, points)
    assert np.abs(got - want).max() < 1e-6


def test_modes_peak_at_one():
    spec = solve_plate(square_mesh(span=1.0), unit_material(0.34), k=6)
    _frames, coeffs, _peaks = quintic_coefficients(spec)
    # the constant term is the value at each centroid, bounded by the peak
    assert np.abs(coeffs[:, :, 0]).max() <= 1.0 + 1e-9


def test_a_payload_decodes_to_one_consistent_plate():
    payload = plate_payload("hexagon", 0.34)
    points = _floats(payload["points"]).reshape(-1, 2)
    triangles = _index(payload, "triangles").reshape(-1, 3)
    boundary = _index(payload, "boundary")
    frames = _floats(payload["frames"]).reshape(-1, 3)
    modes = len(payload["omega"])
    coeffs = _floats(payload["coefficients"]).reshape(modes, -1, 21)
    assert points.shape[0] == payload["vertices"]
    assert triangles.shape[0] == payload["elements"] == frames.shape[0]
    assert coeffs.shape[1] == payload["elements"]
    assert triangles.max() < points.shape[0]
    # plate units: the longest side is one
    assert np.ptp(points, axis=0).max() == pytest.approx(1.0, abs=1e-6)
    # the boundary runs counter-clockwise around the whole plate
    x, y = points[boundary].T
    enclosed = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    corners = points[triangles]
    u, v = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    meshed = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum()
    assert enclosed == pytest.approx(meshed, rel=1e-5)
    assert np.all(np.diff(payload["omega"]) >= 0)
    assert payload["kernel"] < 1e-8


def test_the_boundary_loop_visits_every_boundary_vertex_once():
    mesh = plate_mesh("violin")
    loop = boundary_loop(mesh)
    assert len(set(loop.tolist())) == len(loop) == mesh.boundary_facets().size


def test_metres_and_plate_units_give_the_same_frequencies():
    """Ω is dimensionless, so the unit solve and the app's brass plate agree."""
    metres = payload_from_spectrum(
        "square", solve_plate(square_mesh(), BRASS, k=8), SPAN
    )
    units = plate_payload("square", BRASS.poisson, modes=8)
    np.testing.assert_allclose(metres["omega"], units["omega"], rtol=1e-6)


def test_omega_turns_back_into_the_pinned_hertz():
    from tests.test_presets import EXPECTED

    stiff = np.sqrt(BRASS.flexural_rigidity / BRASS.areal_density)
    omega = np.asarray(plate_payload("square", BRASS.poisson, modes=6)["omega"])
    hz = omega * stiff / (2.0 * np.pi * SPAN**2)
    np.testing.assert_allclose(hz, EXPECTED["square"], atol=0.2)


def test_clusters_group_only_what_is_closer_than_the_gap():
    assert degenerate_clusters([1.0, 1.0, 2.0]) == [[0, 1], [2]]
    close, far = 1.0 + 0.5 * CLUSTER_GAP, 1.0 + 3.0 * CLUSTER_GAP
    assert degenerate_clusters([1.0, close, far]) == [[0, 1], [2]]
    square = plate_payload("square", 0.34, modes=6)
    assert square["clusters"] == [[0], [1], [2], [3, 4], [5]]


def test_the_ellipse_has_no_shared_notes():
    """Its symmetry group has only one-dimensional representations."""
    clusters = plate_payload("ellipse", 0.34)["clusters"]
    assert all(len(c) == 1 for c in clusters)


def test_the_manifest_lists_every_plate_and_sheet_for_every_poisson_ratio():
    m = manifest()
    assert [p["key"] for p in m["plates"]] == list(PRESETS)
    assert [s["key"] for s in m["materials"]] == list(MATERIALS)
    ratios = poisson_ratios()
    distinct = {round(e["material"].poisson, 4) for e in MATERIALS.values()}
    assert len(ratios) == len(distinct)
    for plate in m["plates"]:
        assert set(plate["files"]) == {f"{nu:.2f}" for nu in ratios}
        assert len(plate["outline"]) >= 4


def test_materials_are_physical():
    for key, entry in MATERIALS.items():
        mat = entry["material"]
        assert 0.0 <= mat.poisson < 0.5, key
        assert mat.young > 1e9 and mat.density > 1000.0, key
        assert 0.0 < mat.loss_factor < 0.01, key
        assert entry["note"], key


def test_the_benchmarks_the_page_shows_are_right():
    v = verification()
    np.testing.assert_allclose(
        v["square"]["computed"], v["square"]["reference"], rtol=2e-4
    )
    for row in v["disc"]["rows"]:
        assert row["computed"] == pytest.approx(row["reference"], rel=2e-3)
