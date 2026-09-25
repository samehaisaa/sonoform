"""A small local web app.

``sonoform play`` serves the page in ``web/`` and the data it reads. The data
is the same set of files the published site carries, ``data/manifest.json``
and one payload per plate and Poisson ratio, produced on request by
:mod:`sonoform.export` rather than read from disk, so a checkout always serves
what its own solver says.

The POST routes are older: a plate solve for the presets and the square, and
the membrane editor. The page no longer calls them.
"""

from __future__ import annotations

import base64
import io
import json
import re
import threading
import webbrowser
from collections import OrderedDict
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from sonoform import export
from sonoform.audio import (
    radiation_decay,
    render,
    render_plate,
    structural_decay,
    write_wav,
)
from sonoform.geometry import FourierShape
from sonoform.inverse import CHORDS, check_feasibility, solve_inverse
from sonoform.plate import (
    SPAN,
    display_field,
    outline_mesh,
    solve_plate,
    square_mesh,
)
from sonoform.presets import PRESETS, preset_request
from sonoform.spectrum import solve_spectrum

__all__ = ["serve"]

_WEB = Path(__file__).parent / "web"

_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
    ".webmanifest": "application/manifest+json",
}

# Payloads are a few hundred kilobytes each and take half a second to solve.
# Keeping the serialised bytes rather than the spectra matters: a spectrum
# carries its whole Argyris basis, megabytes apiece.
_DATA_LOCK = threading.Lock()
_DATA_CACHE: OrderedDict[str, bytes] = OrderedDict()
_DATA_KEEP = 16
_PLATE_FILE = re.compile(r"plates/([a-z]+)-(\d\.\d\d)\.json")


def _static_file(route: str) -> Path | None:
    """A file under ``web/`` that may be served, or None.

    The route is resolved and must land inside ``web/``, so ``..`` and
    encoded variants of it cannot reach anything else on the machine.
    """
    root = _WEB.resolve()
    relative = route.lstrip("/") or "index.html"
    try:
        path = (root / relative).resolve()
    except (OSError, ValueError):
        return None
    if root not in path.parents:
        return None
    if not path.is_file() or path.suffix not in _TYPES:
        return None
    return path


def _data_file(route: str) -> bytes | None:
    """One file of the data tree, built on first request and then kept."""
    name = route[len("/data/"):]
    if name == "manifest.json":
        build = export.manifest
    elif name == "verification.json":
        build = export.verification
    else:
        match = _PLATE_FILE.fullmatch(name)
        if match is None or match[1] not in PRESETS:
            return None
        poisson = float(match[2])
        if not any(abs(poisson - nu) < 1e-9 for nu in export.poisson_ratios()):
            return None

        def build(key=match[1], poisson=poisson):
            return export.plate_payload(key, poisson)

    with _DATA_LOCK:
        if name in _DATA_CACHE:
            _DATA_CACHE.move_to_end(name)
            return _DATA_CACHE[name]
        body = json.dumps(build(), separators=(",", ":")).encode("utf-8")
        _DATA_CACHE[name] = body
        while len(_DATA_CACHE) > _DATA_KEEP:
            _DATA_CACHE.popitem(last=False)
        return body


# Live editing resolution.
#
# The live mesh has to be accurate enough that a correct shape does not read
# as broken. Linear elements at 8x96 carry about seventy cents of error on a
# curved boundary, enough to display a shape solved to within a cent as 1.3016
# instead of 1.2500. Quadratic at 8x160 lands near one cent for roughly fifty
# milliseconds, which is still fast enough to drag a slider against.
#
# The angular count is what matters here; see the note in inverse.solve_inverse.
_LIVE_RADIAL = 8
_LIVE_ANGULAR = 160
_LIVE_ORDER = 2


def _shape_from(payload) -> FourierShape:
    return FourierShape(
        float(payload.get("a0", 1.0)),
        [float(v) for v in payload.get("cos", [])],
        [float(v) for v in payload.get("sin", [])],
    )


def _nearest_chord(ratios) -> str:
    """The named chord this spectrum is closest to, in cents."""
    best, best_err = None, np.inf
    for name, target in CHORDS.items():
        n = min(len(target), len(ratios))
        if n < 2:
            continue
        err = np.abs(
            1200.0 * np.log2(np.asarray(ratios[:n]) / np.asarray(target[:n]))
        ).max()
        if err < best_err:
            best, best_err = name, err
    return f"{best} ({best_err:.0f} cents off)" if best else "unknown"


def _spectrum_payload(shape: FourierShape, n_modes: int) -> dict:
    """Boundary, partials and the modes sampled on the polar grid.

    star_mesh lays its vertices out as the centre followed by rings of
    n_angular points, so the eigenvector restricted to vertices already is a
    polar grid. The browser rebuilds the positions from the boundary radius
    and needs only the values.
    """
    mesh = shape.mesh(n_radial=_LIVE_RADIAL, n_angular=_LIVE_ANGULAR)
    spec = solve_spectrum(mesh, k=n_modes, order=_LIVE_ORDER)

    theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    radius = shape.radius(theta)

    modes = []
    for i in range(len(spec)):
        values = spec.eigenvectors[: mesh.p.shape[1], i]
        peak = np.abs(values).max()
        if peak > 0:
            values = values / peak
        # orient consistently so the colours do not flip between frames
        if values[np.argmax(np.abs(values))] < 0:
            values = -values
        modes.append([round(float(v), 4) for v in values])

    return {
        "theta": [round(float(t), 5) for t in theta],
        "radius": [round(float(r), 5) for r in radius],
        "ratios": [round(float(r), 5) for r in spec.ratios],
        "order": _LIVE_ORDER,
        "nRadial": _LIVE_RADIAL,
        "nAngular": _LIVE_ANGULAR,
        "modes": modes,
        "nearest": _nearest_chord(spec.ratios),
        "valid": bool(shape.is_valid()),
    }


_PLATE_LOCK = threading.Lock()
_PLATE_CACHE: dict = {"key": None, "spec": None, "outline": None}


def _square_outline(span: float = SPAN) -> np.ndarray:
    half = 0.5 * span
    return np.array(
        [[-half, half, half, -half], [-half, -half, half, half]], dtype=float
    )


def _plate_for(payload):
    """Solve, or reuse the last plate when the outline has not changed."""
    count = int(payload.get("modes", 6))
    preset = payload.get("preset")
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"no preset called {preset!r}")
        payload = {**preset_request(preset, count), "modes": count}
    points = payload.get("points")
    if payload.get("shape") == "square" or not points:
        key = ("square", count)

        def build():
            return solve_plate(square_mesh(), k=count), _square_outline()

    else:
        raw = np.asarray(points, dtype=float).ravel()
        key = ("outline", count, tuple(np.round(raw, 4)))

        def build(points=points):
            mesh, outline = outline_mesh(points)
            return solve_plate(mesh, k=count), outline

    with _PLATE_LOCK:
        if _PLATE_CACHE["key"] == key:
            return _PLATE_CACHE["spec"], _PLATE_CACHE["outline"]
        spec, outline = build()
        _PLATE_CACHE["key"] = key
        _PLATE_CACHE["spec"] = spec
        _PLATE_CACHE["outline"] = outline
        return spec, outline


def _plate_payload(spec, outline) -> dict:
    # Draw on a subdivided mesh. Argyris is quintic, so the solve mesh is far
    # coarser than the solution it carries, and rendering straight off it gives
    # a faceted sheet and a nodal contour of obvious straight segments. The
    # subdivision costs no eigensolve; see plate.display_field.
    points, triangles, fine_values, fine_grads = display_field(spec, level=3)
    mesh = spec.mesh
    modes = []
    for index in range(len(spec)):
        values = fine_values[index]
        gradient = fine_grads[index]
        raw = spec._vertex_component(index, 0)
        frequency = float(spec.frequencies[index])
        gamma = structural_decay(frequency, spec.material.loss_factor)
        gamma += radiation_decay(spec, index)
        modes.append(
            {
                "hz": round(frequency, 3),
                "gamma": round(float(gamma), 5),
                "peak": round(float(np.abs(raw).max()), 6),
                "values": [round(float(v), 5) for v in values],
                "grad": [
                    [round(float(gradient[0, i]), 5), round(float(gradient[1, i]), 5)]
                    for i in range(gradient.shape[1])
                ],
            }
        )
    return {
        "valid": True,
        "points": [[round(float(x), 6), round(float(y), 6)] for x, y in points.T],
        "triangles": [[int(a), int(b), int(c)] for a, b, c in triangles.T],
        "outline": [[round(float(x), 6), round(float(y), 6)] for x, y in outline.T],
        "modes": modes,
        "span": SPAN,
        "thickness": spec.material.thickness,
        "elements": int(mesh.t.shape[1]),
        "drawnElements": int(triangles.shape[1]),
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the console quiet
        pass

    def handle(self):
        with suppress(ConnectionResetError, BrokenPipeError):
            super().handle()

    def _route(self) -> str:
        return self.path.split("?", 1)[0]

    def _send(self, code, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        route = self._route()
        if route.startswith("/data/"):
            try:
                body = _data_file(route)
            except Exception as exc:  # a failed solve should not take the server down
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
                return
            if body is None:
                self._json({"error": "not found"}, 404)
            else:
                self._send(200, body, "application/json")
        elif route == "/presets":
            # Labels only. The outlines stay server side so the page has no
            # second copy of them to drift from.
            self._json({key: {"label": p["label"]} for key, p in PRESETS.items()})
        elif route == "/chords":
            self._json(
                {
                    name: {
                        "ratios": list(ratios),
                        "possible": bool(check_feasibility(ratios)),
                    }
                    for name, ratios in CHORDS.items()
                }
            )
        else:
            path = _static_file(route)
            if path is None:
                self._json({"error": "not found"}, 404)
            else:
                self._send(200, path.read_bytes(), _TYPES[path.suffix])

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return

        route = self._route()
        try:
            if route == "/spectrum":
                shape = _shape_from(payload)
                if not shape.is_valid():
                    self._json({"valid": False, "error": "boundary crosses itself"})
                    return
                self._json(_spectrum_payload(shape, int(payload.get("modes", 6))))

            elif route == "/audio":
                shape = _shape_from(payload)
                spec = solve_spectrum(shape.mesh(10, 200), k=10, order=2)
                signal = render(
                    spec,
                    fundamental_hz=float(payload.get("hz", 196.0)),
                    duration=float(payload.get("duration", 2.5)),
                )
                buffer = io.BytesIO()
                tmp = Path(self.server.tmpdir) / "clip.wav"
                write_wav(tmp, signal)
                buffer.write(tmp.read_bytes())
                self._send(200, buffer.getvalue(), "audio/wav")

            elif route == "/plate":
                try:
                    spec, outline = _plate_for(payload)
                except ValueError as exc:
                    self._json({"valid": False, "error": str(exc)})
                    return
                self._json(_plate_payload(spec, outline))

            elif route == "/plateAudio":
                spec, _outline = _plate_for(payload)
                strike = payload.get("strike")
                if strike is None:
                    strike = (0.22 * SPAN, 0.06 * SPAN)
                signal, coeffs = render_plate(
                    spec,
                    (float(strike[0]), float(strike[1])),
                    duration=float(payload.get("duration", 2.5)),
                )
                tmp = Path(self.server.tmpdir) / f"plate-{threading.get_ident()}.wav"
                write_wav(tmp, signal)
                self._json(
                    {
                        "wav": base64.b64encode(tmp.read_bytes()).decode("ascii"),
                        "modes": coeffs,
                        "sampleRate": 44100,
                    }
                )

            elif route == "/solve":
                target = payload.get("target")
                ratios = CHORDS.get(target, target)
                feasible = check_feasibility(ratios)
                if not feasible:
                    self._json({"possible": False, "reason": feasible.reason})
                    return
                result = solve_inverse(
                    target,
                    n_harmonics=int(payload.get("harmonics", 5)),
                    n_restarts=int(payload.get("restarts", 2)),
                    seed=1,
                )
                self._json(
                    {
                        "possible": True,
                        "a0": result.shape.a0,
                        "cos": result.shape.cos_coeffs.tolist(),
                        "sin": result.shape.sin_coeffs.tolist(),
                        "maxCents": result.max_cents,
                        "success": result.success,
                        "achieved": result.achieved.tolist(),
                        "target": result.target.tolist(),
                    }
                )
            else:
                self._json({"error": "not found"}, 404)

        except Exception as exc:  # a bad shape should not take the server down
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def serve(
    port: int = 8731, open_browser: bool = True, host: str = "127.0.0.1"
) -> None:
    """Run the local app until interrupted.

    ``host`` stays on the loopback address by default. A container has to bind
    0.0.0.0 to be reachable from outside its own network namespace, which is
    what the image does, but nothing else should.
    """
    import tempfile

    server = ThreadingHTTPServer((host, port), _Handler)
    server.tmpdir = tempfile.mkdtemp(prefix="sonoform-")
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{shown}:{port}/"
    print(f"\n  sonoform is running at {url}")
    print("  press Ctrl+C to stop\n")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped\n")
    finally:
        server.server_close()
