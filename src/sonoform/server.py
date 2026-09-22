"""A small local web app.

``sonoform play`` serves a free brass plate. Draw a closed outline, or start
from the square. The eigenproblem is solved once; the page then flexes one
partial, lets sand diffuse onto its nodal set, and plays the pressure a strike
radiates. The membrane routes are still here for the earlier editor.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import webbrowser
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from sonoform.audio import (
    radiation_decay,
    render,
    render_plate,
    structural_decay,
    write_wav,
)
from sonoform.geometry import (
    FourierShape,
    is_simple_polygon,
    polygon_mesh,
    resample_closed_path,
)
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


def _drawn_payload(points, n_modes: int) -> dict:
    """Spectrum of a freehand outline.

    Unlike the slider shapes this is an arbitrary polygon, so there is no
    polar grid to exploit. The triangulation itself goes to the browser and
    the modes ride along as one value per vertex.
    """
    path = resample_closed_path(points, n=150)
    if not is_simple_polygon(path):
        return {"valid": False, "error": "the outline crosses itself"}

    # normalise scale so the pitch does not depend on how big you drew it
    path = path - path.mean(axis=1, keepdims=True)
    path = path / np.abs(path).max()

    mesh = polygon_mesh(path, max_area=float(np.ptp(path[0]) * np.ptp(path[1]) / 1400))
    spec = solve_spectrum(mesh, k=n_modes, order=2)

    n_vertices = mesh.p.shape[1]
    modes = []
    for i in range(len(spec)):
        values = spec.eigenvectors[:n_vertices, i]
        peak = np.abs(values).max()
        if peak > 0:
            values = values / peak
        if values[np.argmax(np.abs(values))] < 0:
            values = -values
        modes.append([round(float(v), 4) for v in values])

    return {
        "valid": True,
        "points": [[round(float(x), 5), round(float(y), 5)] for x, y in mesh.p.T],
        "triangles": [[int(a), int(b), int(c)] for a, b, c in mesh.t.T],
        "outline": [[round(float(x), 5), round(float(y), 5)] for x, y in path.T],
        "ratios": [round(float(r), 5) for r in spec.ratios],
        "modes": modes,
        "nearest": _nearest_chord(spec.ratios),
        "elements": int(mesh.t.shape[1]),
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
        if len(points) < 6:
            raise ValueError("draw a bigger loop")
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
        if route in ("/", "/index.html"):
            self._send(
                200,
                (_WEB / "index.html").read_bytes(),
                "text/html; charset=utf-8",
            )
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
            self._json({"error": "not found"}, 404)

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

            elif route == "/drawn":
                points = payload.get("points") or []
                if len(points) < 6:
                    self._json({"valid": False, "error": "draw a bigger loop"})
                    return
                self._json(_drawn_payload(points, int(payload.get("modes", 6))))

            elif route == "/drawnAudio":
                path = resample_closed_path(payload.get("points") or [], n=150)
                path = path - path.mean(axis=1, keepdims=True)
                path = path / np.abs(path).max()
                mesh = polygon_mesh(
                    path, max_area=float(np.ptp(path[0]) * np.ptp(path[1]) / 1400)
                )
                spec = solve_spectrum(mesh, k=10, order=2)
                # Where the shape is struck is the user's choice, not ours.
                # It decides which modes sound: a strike on a nodal line
                # cannot drive that mode at all.
                strike = payload.get("strike")
                if strike is None:
                    strike = (0.35 * float(mesh.p[0].max()), 0.0)
                signal = render(
                    spec,
                    fundamental_hz=float(payload.get("hz", 196.0)),
                    strike=(float(strike[0]), float(strike[1])),
                    duration=float(payload.get("duration", 2.5)),
                )
                tmp = Path(self.server.tmpdir) / "drawn.wav"
                write_wav(tmp, signal)
                self._send(200, tmp.read_bytes(), "audio/wav")

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
