"""Tests for the local web app's data layer."""

import numpy as np
import pytest

from sonoform.geometry import FourierShape
from sonoform.server import _LIVE_ANGULAR, _LIVE_RADIAL, _nearest_chord, _shape_from
from sonoform.server import _spectrum_payload as payload


def test_shape_round_trips_through_the_wire_format():
    shape = _shape_from({"a0": 0.9, "cos": [0.1, -0.2], "sin": [0.05, 0.0]})
    assert shape.a0 == pytest.approx(0.9)
    np.testing.assert_allclose(shape.cos_coeffs, [0.1, -0.2])
    np.testing.assert_allclose(shape.sin_coeffs, [0.05, 0.0])


def test_live_resolution_is_accurate_enough_to_display():
    """A correct shape must not display as a wrong one.

    The disc's second partial is 1.5933. A coarse live mesh puts it near
    1.6060, which is 135 cents out and enough to make a shape solved to within
    a cent look broken. Whatever the live mesh is, it has to stay well inside
    what a listener could hear.
    """
    out = payload(FourierShape(1.0), 4)
    second = out["ratios"][1]
    cents = abs(1200.0 * np.log2(second / 1.5933402))
    assert cents < 5.0, f"live display is {cents:.0f} cents off"


def test_payload_grid_matches_the_declared_mesh():
    """The browser rebuilds positions from nRadial and nAngular alone."""
    out = payload(FourierShape(1.0, [0.1], [0.05]), 3)
    expected = 1 + out["nRadial"] * out["nAngular"]
    assert out["nRadial"] == _LIVE_RADIAL
    assert out["nAngular"] == _LIVE_ANGULAR
    for mode in out["modes"]:
        assert len(mode) == expected


def test_modes_are_normalised_and_oriented():
    out = payload(FourierShape(1.0, [0.12], [0.04]), 4)
    for mode in out["modes"]:
        values = np.asarray(mode)
        assert np.abs(values).max() == pytest.approx(1.0, abs=1e-3)
        # the largest magnitude is positive, so colours do not flip per frame
        assert values[np.argmax(np.abs(values))] > 0


def test_boundary_samples_line_up():
    out = payload(FourierShape(0.9, [0.2], [0.1]), 2)
    assert len(out["theta"]) == len(out["radius"])
    assert min(out["radius"]) > 0


def test_nearest_chord_recognises_an_exact_match():
    assert _nearest_chord([1.0, 1.25, 1.5]).startswith("major")
    assert "0 cents" in _nearest_chord([1.0, 1.25, 1.5])


def test_nearest_chord_reports_distance_for_a_disc():
    label = _nearest_chord([1.0, 1.5933, 1.5933])
    assert "cents off" in label


def test_plate_payload_lists_audible_partials():
    from sonoform.server import _plate_for, _plate_payload

    spec, outline = _plate_for({"shape": "square", "modes": 4})
    out = _plate_payload(spec, outline)
    assert out["valid"] is True
    assert len(out["modes"]) == 4
    assert out["modes"][0]["hz"] > 50.0
    assert len(out["modes"][0]["values"]) == len(out["points"])
    assert len(out["modes"][0]["grad"]) == len(out["points"])
    assert len(out["triangles"][0]) == 3
    cached, _ = _plate_for({"shape": "square", "modes": 4})
    assert cached is spec


def test_plate_http_solves_and_returns_a_wav(tmp_path):
    import base64
    import json
    import threading
    import urllib.request
    import wave
    from http.server import ThreadingHTTPServer

    from sonoform.server import _Handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.tmpdir = str(tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read()
        assert b"draw a closed loop" in page
        body = json.dumps({"shape": "square", "modes": 4}).encode()
        plate = json.load(
            urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{port}/plate",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
            )
        )
        assert plate["valid"] is True
        assert len(plate["modes"]) == 4
        audio_body = json.dumps(
            {"shape": "square", "modes": 4, "strike": [0.04, 0.02], "duration": 0.3}
        ).encode()
        audio = json.load(
            urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{port}/plateAudio",
                    data=audio_body,
                    headers={"Content-Type": "application/json"},
                )
            )
        )
        wav_path = tmp_path / "heard.wav"
        wav_path.write_bytes(base64.b64decode(audio["wav"]))
        with wave.open(str(wav_path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == 44100
            assert handle.getnframes() > 1000
        heard = audio["modes"][0]["hz"]
        shown = plate["modes"][0]["hz"]
        assert heard == pytest.approx(shown, abs=0.05)
    finally:
        server.shutdown()
