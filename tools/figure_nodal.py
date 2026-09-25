"""Draw docs/fig_nodal.png: one mode, its nodal set, and the sand on it.

    python tools/figure_nodal.py [mode] [out]

Everything comes from the solver. The left panel is the mode shape w(x, y)
of the free square, the middle its zero set, and the right panel is sand
sampled from the walk's steady state, rho proportional to 1 / (D0 + |w|).
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.tri import LinearTriInterpolator, Triangulation  # noqa: E402

from sonoform.plate import display_field, solve_plate, square_mesh  # noqa: E402

NIGHT, SLATE, BONE, ASH = "#161715", "#2a2b28", "#e8e2d5", "#8d897f"
OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "fig_nodal.png"


def main(index: int = 2, out: pathlib.Path = OUT) -> None:
    spec = solve_plate(square_mesh(), k=index + 4)
    points, triangles, values, _grads = display_field(spec, level=4)
    w = values[index] / np.abs(values[index]).max()
    x, y = points
    half = 0.5 * (x.max() - x.min())
    x, y = x / half, y / half
    tri = Triangulation(x, y, triangles.T)

    fig = plt.figure(figsize=(12, 4.2), dpi=160, facecolor=NIGHT)
    shade = LinearSegmentedColormap.from_list("stone", ["#4a4a44", SLATE, BONE])

    # the mode shape, as the plate bends
    ax = fig.add_subplot(1, 3, 1, projection="3d", facecolor=NIGHT)
    ax.computed_zorder = False
    ax.plot_trisurf(
        tri,
        0.55 * w,
        cmap=shade,
        linewidth=0,
        edgecolor="none",
        antialiased=False,
        shade=True,
        zorder=1,
    )
    # the nodal line on the surface itself, where the height is zero
    probe = plt.figure()
    zero = probe.add_subplot().tricontour(tri, w, levels=[0]).allsegs[0]
    plt.close(probe)
    for seg in zero:
        ax.plot(seg[:, 0], seg[:, 1], 0.01, color=BONE, linewidth=2.0, zorder=2)
    ax.set_zlim(-1.1, 1.1)
    ax.view_init(elev=42, azim=-58)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.62), zoom=1.35)

    # where it does not move
    ax = fig.add_subplot(1, 3, 2, facecolor=NIGHT)
    amplitude = LinearSegmentedColormap.from_list("amp", [NIGHT, "#55544d"])
    ax.tricontourf(tri, np.abs(w), levels=24, cmap=amplitude)
    ax.tricontour(tri, w, levels=[0], colors=[BONE], linewidths=1.6)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # sand, sampled from the steady state of the walk
    ax = fig.add_subplot(1, 3, 3, facecolor=NIGHT)
    ax.fill([-1, 1, 1, -1], [-1, -1, 1, 1], color=SLATE, lw=0)
    interp = LinearTriInterpolator(tri, np.abs(w))
    rng = np.random.default_rng(3)
    floor, grains = 0.004, []
    while sum(len(g) for g in grains) < 18000:
        px, py = rng.uniform(-1, 1, (2, 40000))
        a = np.ma.filled(interp(px, py), 1.0)
        keep = rng.uniform(size=px.size) < floor / (floor + a)
        grains.append(np.c_[px[keep], py[keep]])
    g = np.vstack(grains)[:18000]
    ax.scatter(g[:, 0], g[:, 1], s=0.5, c=BONE, linewidths=0)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_aspect("equal")
    ax.set_axis_off()

    labels = [
        "the mode  w(x, y)",
        "its nodal set  w = 0",
        "sand at rest  ρ ∝ 1 / (D₀ + |w|)",
    ]
    for i, text in enumerate(labels):
        fig.text((i + 0.5) / 3, 0.05, text, ha="center", color=ASH, fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.12, wspace=0.08)
    fig.savefig(out, facecolor=NIGHT)
    plt.close(fig)
    print(out, f"{spec.frequencies[index]:.1f} Hz")


if __name__ == "__main__":
    mode = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    main(mode, pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else OUT)
