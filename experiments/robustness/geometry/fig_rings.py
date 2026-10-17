#!/usr/bin/env python3
"""P3b geometry figure: the readout decides whether DA can find the slow mode.

Two panels, same density, two coordinate systems.

  (a) Cartesian readout x=(x1,x2). The two-shell density is rotationally
      symmetric, so every direction is equivalent by construction: the per-seed
      DA1 axes point nowhere in particular and the spectrum is degenerate.

  (b) Polar readout. The same trajectories expressed as (r, theta) put the two
      shells at two values of r, and DA1 lands almost entirely on the radial
      coordinate.

The claim is about DA only. SSA is deliberately not shown here: its verdict
flips with the choice of unimodal baseline (against a matched isotropic Gaussian
the radial readout shows no excess, because q=||x||^2 of a Gaussian is itself a
slowly-relaxing chi-squared), and most cells are window-limited.

Densities are analytic, imported from the experiment module. DA1 vectors come
from p3b_polar_da.py, which computes both readouts through one code path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/robustness/geometry"))

POLAR = ROOT / "experiments/robustness/geometry/results/p3b_rings/p3b_polar_da.json"
OUT = ROOT / "experiments/robustness/figures/p3b_rings"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRID = "#d9d8d4"
INK = "#52514e"
TEXT = "#0b0b0b"

LAG = 3000  # physical t = 6.0; the radial loading has plateaued by here


def main() -> None:
    from p3b_continuous_rings import two_shell  # noqa: E402 (path injected above)

    pol = json.loads(POLAR.read_text())
    rows = [r for r in pol["rows"] if r["lag"] == LAG]
    if not rows:
        raise SystemExit(f"no rows at lag {LAG} in {POLAR}")
    dens = two_shell(pol["m1"], pol["m2"], pol["ratio"])
    r_shell = np.sqrt([pol["m1"], pol["m2"]])
    load = float(np.median([r["polar_radial_loading"] for r in rows]))

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": TEXT,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": TEXT,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))

    # ---------------- (a) Cartesian: DA1 is an arbitrary diameter ----------
    ax = axes[0]
    lim = float(r_shell.max()) * 1.45
    g = np.linspace(-lim, lim, 420)
    X, Y = np.meshgrid(g, g)
    P = (dens.pdf_q((X ** 2 + Y ** 2).ravel()) / np.pi).reshape(X.shape)
    ax.contourf(X, Y, P, levels=12, cmap="Blues")

    ramp = plt.cm.autumn(np.linspace(0.0, 0.7, len(rows)))
    for r, col in zip(rows, ramp):
        v = np.asarray(r["ambient_da1"], dtype=float)
        u = v / np.linalg.norm(v) * lim * 0.95
        ax.plot([-u[0], u[0]], [-u[1], u[1]], lw=2.0, color=col,
                solid_capstyle="round", zorder=4)

    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("(a)", loc="left", fontweight="bold")

    # ---------------- (b) Polar: DA1 is the radial coordinate --------------
    ax = axes[1]
    rr = np.linspace(0.0, lim, 400)
    th = np.linspace(-np.pi, np.pi, 200)
    # Density of (r, theta): p(r,theta) = r * p_q(r^2) / pi, flat in theta.
    Pr = rr * dens.pdf_q(rr ** 2) / np.pi
    ax.contourf(rr, th, np.tile(Pr, (len(th), 1)), levels=12, cmap="Blues")

    # DA1 lies in the (radial, angular) plane; its radial weight is `load`.
    # Draw it as a double-headed axis whose tilt is set by that measured split.
    ang_w = np.sqrt(max(0.0, 1.0 - load))
    scale_r, scale_t = 0.42 * lim, 0.42 * np.pi
    dx, dy = np.sqrt(load) * scale_r, ang_w * scale_t
    cx = 0.5 * (r_shell[0] + r_shell[1])
    ax.plot([cx - dx, cx + dx], [-dy, dy], lw=2.4, color="#d81e5b",
            solid_capstyle="round", zorder=4)

    ax.set_xlim(0.0, lim)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_xlabel(r"$r=\|x\|$")
    ax.set_ylabel(r"$\theta$")
    ax.set_title("(b)", loc="left", fontweight="bold")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")

    print(f"wrote {OUT}.pdf / .png")
    print(f"  ratio={pol['ratio']}  m1={pol['m1']}  m2={pol['m2']}"
          f"  shells r={np.round(r_shell,3).tolist()}")
    print(f"  lag={LAG} (t={LAG*pol['config']['dt']:.1f})  N={pol['config']['N']}"
          f"  T={pol['config']['T']}  seeds={len(rows)}")
    print(f"  polar DA1 radial loading: median={load:.4f}"
          f"  range=[{min(r['polar_radial_loading'] for r in rows):.4f},"
          f" {max(r['polar_radial_loading'] for r in rows):.4f}]")
    print(f"  polar gap   median={np.median([r['polar_gap'] for r in rows]):.2f}")
    print(f"  ambient gap median={np.median([r['ambient_gap'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
