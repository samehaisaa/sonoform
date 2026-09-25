"""The preset outlines, and the frequencies they are supposed to produce.

These numbers are what the app has always shown for its five plates. They are
pinned here because the presets are now the single source of truth for both the
interactive app and the precomputed demo: if an outline is edited, the demo
silently goes stale rather than failing, so the drift has to be caught here.
"""

import numpy as np
import pytest

from sonoform.plate import BRASS, outline_mesh, solve_plate, square_mesh
from sonoform.presets import PRESETS, preset_points, preset_request

# Hz, at BRASS, for the six lowest flexible modes of each preset.
#
# The first five rows are what the app has shown since the presets existed.
# The last four arrived with the solver's unit-independent shift, and were
# checked against a dense solve of the same matrices when they were pinned.
EXPECTED = {
    "square": [135.4, 198.0, 252.8, 352.8, 352.8, 629.8],
    "circle": [216.5, 216.5, 376.3, 504.1, 504.1, 849.8],
    "triangle": [336.7, 344.2, 344.3, 827.4, 828.5, 874.8],
    "ellipse": [266.6, 345.9, 667.7, 753.3, 778.7, 1225.6],
    "hexagon": [257.7, 257.8, 446.2, 551.3, 652.5, 986.4],
    "flower": [239.1, 239.1, 400.0, 613.0, 613.0, 738.3],
    "stadium": [255.3, 340.0, 668.2, 761.8, 991.8, 1285.9],
    "guitar": [234.2, 240.8, 481.5, 569.9, 691.0, 914.9],
    "violin": [234.2, 279.9, 605.0, 766.2, 834.0, 1170.3],
}


def test_every_preset_has_a_label_and_a_request():
    assert set(PRESETS) == set(EXPECTED)
    for key in PRESETS:
        assert PRESETS[key]["label"]
        request = preset_request(key)
        assert request["modes"] == 6
        assert "points" in request or request.get("shape") == "square"


def test_the_square_is_the_only_one_without_an_outline():
    assert preset_points("square") is None
    for key in PRESETS:
        if key == "square":
            continue
        points = np.asarray(preset_points(key))
        assert len(points) >= 150, key
        assert np.isfinite(points).all(), key
        radius = np.hypot(*points.T)
        assert radius.min() > 0.0, key
        # the polygon presets reach 1/floor at their vertices, not 1, so the
        # only thing worth asserting is that nothing has blown up
        assert radius.max() < 1.0, key


def test_the_scale_of_a_preset_outline_does_not_matter():
    """outline_mesh normalises, so the preset scale is presentational only.

    Worth pinning because it is easy to assume the 0.09 in presets sets the
    physical size. It does not: the longest side of the bounding box becomes
    SPAN, so drawing the same shape at any size gives the same pitch.
    """
    points = np.asarray(preset_points("hexagon"))
    a = outline_mesh(points.tolist())[0]
    b = outline_mesh((points * 7.3).tolist())[0]
    fa = solve_plate(a, BRASS, k=2).frequencies
    fb = solve_plate(b, BRASS, k=2).frequencies
    # In cents, because that is the unit the claim is made in. The residual is
    # the resample and it moves with the scipy version, from 0.0001 cents to
    # about 0.02, so a relative tolerance here pins the library rather than the
    # physics. Half a cent is an order of magnitude below anything audible.
    cents = np.abs(1200.0 * np.log2(fb / fa))
    assert cents.max() < 0.5, f"{cents.max():.4f} cents apart at 7.3x the size"


def test_outlines_are_simple_and_centred():
    for key in PRESETS:
        points = preset_points(key)
        if points is None:
            continue
        arr = np.asarray(points)
        assert np.allclose(arr.mean(axis=0), 0.0, atol=2e-3), key


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_preset_frequencies_have_not_drifted(key):
    points = preset_points(key)
    mesh = square_mesh() if points is None else outline_mesh(points)[0]
    spec = solve_plate(mesh, BRASS, k=6)
    got = [round(float(f), 1) for f in spec.frequencies]
    np.testing.assert_allclose(got, EXPECTED[key], atol=0.2)
