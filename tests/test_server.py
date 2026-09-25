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
        assert b"sonoform" in page
        assert b'src="js/main.js"' in page
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


def test_serve_binds_loopback_unless_told_otherwise():
    """The container has to bind 0.0.0.0; nothing else should.

    Guards the default rather than the container, since the failure that
    matters is a local run quietly listening on every interface.
    """
    import inspect

    from sonoform.server import serve

    assert inspect.signature(serve).parameters["host"].default == "127.0.0.1"


def test_play_passes_the_host_through(monkeypatch):
    from sonoform import cli

    seen = {}
    monkeypatch.setattr(
        "sonoform.server.serve", lambda **kw: seen.update(kw) or None
    )
    cli.main(["play", "--host", "0.0.0.0", "--port", "9123", "--no-browser"])
    assert seen == {"host": "0.0.0.0", "port": 9123, "open_browser": False}


@pytest.fixture
def running(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer

    from sonoform.server import _Handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.tmpdir = str(tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def _get(url):
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url) as res:
            return res.status, res.headers.get("Content-Type"), res.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers.get("Content-Type"), err.read()


def test_the_page_and_its_modules_are_served_with_their_types(running):
    status, kind, body = _get(f"{running}/app.css")
    assert status == 200 and kind.startswith("text/css") and body
    status, kind, body = _get(f"{running}/js/main.js")
    assert status == 200 and kind.startswith("text/javascript")
    assert b"import" in body
    status, kind, _ = _get(f"{running}/icon.svg")
    assert status == 200 and kind == "image/svg+xml"


@pytest.mark.parametrize(
    "path",
    [
        "/../pyproject.toml",
        "/js/../../server.py",
        "/%2e%2e/%2e%2e/pyproject.toml",
        "/..%2f..%2fpyproject.toml",
        "/js/",
        "/nothing-here.html",
        "/server.py",
    ],
)
def test_nothing_outside_the_page_is_served(running, path):
    """The static route resolves inside web/ or refuses: no source, no config."""
    status, _kind, body = _get(f"{running}{path}")
    assert status == 404
    assert b"import numpy" not in body and b"[project]" not in body


def test_the_data_tree_is_served_as_the_site_carries_it(running):
    import json

    from sonoform.export import plate_file

    status, _kind, body = _get(f"{running}/data/manifest.json")
    assert status == 200
    manifest = json.loads(body)
    square = next(p for p in manifest["plates"] if p["key"] == "square")
    brass = next(m for m in manifest["materials"] if m["key"] == "brass")
    path = square["files"][f"{brass['poisson']:.2f}"]
    assert path == plate_file("square", brass["poisson"])
    status, _kind, body = _get(f"{running}/data/{path}")
    assert status == 200
    payload = json.loads(body)
    assert payload["key"] == "square"
    assert len(payload["omega"]) == manifest["modes"]


@pytest.mark.parametrize(
    "path",
    [
        "/data/plates/nope-0.34.json",
        "/data/plates/square-0.99.json",
        "/data/other.json",
    ],
)
def test_unknown_data_is_not_found(running, path):
    status, _kind, _body = _get(f"{running}{path}")
    assert status == 404
