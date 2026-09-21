"""Command line interface.

``sonoform tune major`` asks for a shape that rings as a major triad and shows
what came back. ``sonoform chords`` lists the chords it knows, and which of
them are impossible.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from sonoform import __version__
from sonoform.audio import render, write_wav
from sonoform.geometry import FourierShape
from sonoform.inverse import CHORDS, check_feasibility, solve_inverse, verify
from sonoform.spectrum import solve_spectrum

__all__ = ["main"]


def ascii_shape(shape: FourierShape, width: int = 44) -> list[str]:
    """A small terminal drawing of the shape.

    Each character holds two vertical pixels as a half block, which both
    doubles the vertical resolution and fixes the aspect ratio, since terminal
    cells are about twice as tall as they are wide.
    """
    bx, by = shape.boundary(1440)
    # Frame on the bounding box, not on the origin: these shapes are lopsided
    # and centring on the origin wastes half the canvas.
    cx, cy = 0.5 * (bx.max() + bx.min()), 0.5 * (by.max() + by.min())
    span = 0.53 * max(bx.max() - bx.min(), by.max() - by.min())
    rows_of_pixels = max(8, width) // 2 * 2  # even, so it pairs cleanly

    xs = cx + span * (2.0 * (np.arange(width) + 0.5) / width - 1.0)
    ys = cy + span * (1.0 - 2.0 * (np.arange(rows_of_pixels) + 0.5) / rows_of_pixels)
    gx, gy = np.meshgrid(xs, ys)
    inside = np.hypot(gx, gy) <= shape.radius(np.arctan2(gy, gx))

    glyphs = _glyph_set()
    rows = [
        "".join(
            glyphs[(bool(top), bool(bottom))] for top, bottom in zip(t, b, strict=True)
        )
        for t, b in zip(inside[0::2], inside[1::2], strict=True)
    ]
    # The frame is square to keep the aspect honest, so a wide shape leaves
    # blank rows above and below. Drop them.
    while rows and not rows[0].strip():
        rows.pop(0)
    while rows and not rows[-1].strip():
        rows.pop()
    return rows


def _glyph_set() -> dict:
    """Half blocks where the terminal can print them, plain ASCII otherwise.

    Windows consoles still default to cp1252, which cannot encode U+2588 and
    friends, so assuming UTF-8 here crashes the command on the platform.
    """
    blocks = {
        (True, True): "█",
        (True, False): "▀",
        (False, True): "▄",
        (False, False): " ",
    }
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "".join(blocks.values()).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return {
            (True, True): "#",
            (True, False): '"',
            (False, True): "_",
            (False, False): " ",
        }
    return blocks


def _format_series(shape: FourierShape, limit: int = 3) -> str:
    parts = [f"{shape.a0:.3f}"]
    for k, a in enumerate(shape.cos_coeffs[:limit], start=1):
        parts.append(f"{a:+.3f} cos {k}t")
    for k, b in enumerate(shape.sin_coeffs[:limit], start=1):
        parts.append(f"{b:+.3f} sin {k}t")
    tail = " ..." if shape.cos_coeffs.size > limit else ""
    return "r(t) = " + " ".join(parts) + tail


def _parse_target(tokens):
    """Either a chord name or a list of ratios."""
    if len(tokens) == 1 and not _looks_numeric(tokens[0]):
        return tokens[0]
    try:
        return [float(tok) for tok in tokens]
    except ValueError:
        raise SystemExit(
            f"could not read {' '.join(tokens)!r} as a chord name or as ratios"
        ) from None


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _cmd_tune(args) -> int:
    target = _parse_target(args.target)
    if isinstance(target, str) and target not in CHORDS:
        known = ", ".join(sorted(CHORDS))
        print(
            f"\n  unknown chord {target!r}\n  known chords: {known}\n",
            file=sys.stderr,
        )
        return 2
    ratios = np.asarray(CHORDS[target] if isinstance(target, str) else target)

    feasible = check_feasibility(ratios)
    if not feasible:
        label = target if isinstance(target, str) else " ".join(args.target)
        print(f"\n  {label} cannot be built.\n")
        for line in _wrap(feasible.reason, 66):
            print(f"  {line}")
        print(f"\n  nearest achievable: {_ratio_line(feasible.nearest)}\n")
        return 2

    started = time.perf_counter()
    result = solve_inverse(
        target,
        n_harmonics=args.harmonics,
        n_restarts=args.restarts,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - started
    # result.achieved is already measured on a mesh far finer than the search
    # used, so there is nothing to re-check unless asked for more still.
    achieved = (
        verify(result, n_radial=28, n_angular=560)
        if args.extra_check
        else result.achieved
    )
    cents = 1200.0 * np.log2(achieved / result.target)

    print()
    for line in ascii_shape(result.shape):
        print("  " + line)
    print()
    print(f"  target    {_ratio_line(result.target)}")
    print(f"  achieved  {_ratio_line(achieved)}")
    print(f"  error     {_cents_line(cents)}")
    print()
    print(f"  {_format_series(result.shape)}")

    worst = float(np.abs(cents).max())
    if worst < 5.0:
        print(f"  solved in {elapsed:.1f}s, worst partial off by {worst:.1f} cents")
    else:
        print(f"  did NOT converge: worst partial off by {worst:.1f} cents")
        print("  try --harmonics 6 --restarts 4, or a different chord")

    if args.wav or args.png:
        spec = solve_spectrum(result.shape.mesh(10, 200), k=args.modes, order=2)
        if args.wav:
            write_wav(args.wav, render(spec, fundamental_hz=args.hz, duration=2.5))
            print(f"  wrote {args.wav}")
        if args.png:
            _write_png(args.png, result.shape, spec)
            print(f"  wrote {args.png}")
    print()
    return 0 if worst < 5.0 else 1


def _write_png(path, shape, spec) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mesh = spec.mesh
    x, y = mesh.p
    tri = mesh.t.T
    bx, by = shape.boundary(600)
    bx, by = np.append(bx, bx[0]), np.append(by, by[0])

    n = min(4, len(spec))
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4), facecolor="white")
    for i, ax in enumerate(np.atleast_1d(axes)):
        u = spec.nodal_values(i)
        ax.tripcolor(x, y, tri, u, cmap="RdBu_r", vmin=-1, vmax=1, shading="gouraud")
        ax.tricontour(x, y, tri, u, levels=[0.0], colors="k", linewidths=1.5)
        ax.plot(bx, by, color="#222", lw=1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{spec.ratios[i]:.3f}x", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _ratio_line(values) -> str:
    return "  ".join(f"{v:7.4f}" for v in values)


def _cents_line(cents) -> str:
    return "  ".join(f"{c:+6.1f}c" for c in cents)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _cmd_play(args) -> int:
    from sonoform.server import serve

    serve(port=args.port, open_browser=args.open_browser)
    return 0


def _cmd_chords(args) -> int:
    print()
    print(f"  {'chord':<12} {'partials':<32} status")
    print(f"  {'-' * 12} {'-' * 32} {'-' * 26}")
    for name, ratios in CHORDS.items():
        feasible = check_feasibility(ratios)
        status = "buildable" if feasible else "impossible"
        shown = " ".join(f"{r:.3f}" for r in ratios)
        print(f"  {name:<12} {shown:<32} {status}")
    print()
    print("  impossible ones are refused before any search runs: the second")
    print("  partial against a proven bound, the third against a measured one.")
    print()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="sonoform",
        description="The spectrum of a shape, and the shape of a spectrum.",
    )
    parser.add_argument(
        "--version", action="version", version=f"sonoform {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tune = subparsers.add_parser(
        "tune", help="find a shape that rings as the given chord"
    )
    tune.add_argument(
        "target",
        nargs="+",
        help="a chord name such as 'major', or frequency ratios such as 1 1.25 1.5",
    )
    tune.add_argument(
        "--harmonics",
        type=int,
        default=5,
        help="Fourier harmonics in the boundary (default 5)",
    )
    tune.add_argument(
        "--restarts", type=int, default=3, help="random restarts (default 3)"
    )
    tune.add_argument("--seed", type=int, default=0, help="random seed")
    tune.add_argument(
        "--modes", type=int, default=8, help="modes to synthesise for audio and figures"
    )
    tune.add_argument(
        "--hz",
        type=float,
        default=196.0,
        help="fundamental pitch for the audio (default 196, G3)",
    )
    tune.add_argument("--wav", metavar="PATH", help="write the struck sound here")
    tune.add_argument("--png", metavar="PATH", help="write the nodal figures here")
    tune.add_argument(
        "--extra-check",
        action="store_true",
        help="re-measure the result on an even finer mesh than the default",
    )
    tune.set_defaults(func=_cmd_tune)

    chords = subparsers.add_parser("chords", help="list the known chords")
    chords.set_defaults(func=_cmd_chords)

    play = subparsers.add_parser(
        "play", help="open the plate, draw an outline, strike it"
    )
    play.add_argument("--port", type=int, default=8731)
    play.add_argument(
        "--no-browser",
        dest="open_browser",
        action="store_false",
        help="do not open a browser window automatically",
    )
    play.set_defaults(func=_cmd_play, open_browser=True)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"\n  {exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
