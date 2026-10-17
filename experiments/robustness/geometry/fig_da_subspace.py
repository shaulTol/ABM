#!/usr/bin/env python3
"""P3 geometry figure: the identifiable object is the slow SUBSPACE, not DA1.

Two panels for app:rebuttal_geometry.

  (a) The known slow plane span(e1,e2) of the 10D four-mode circular GMM, with
      the true mixture density drawn as contours and each seed's recovered DA1
      overlaid. The plane itself is recovered to <0.4 deg in every seed, but
      DA1 within the plane is free to rotate: the seed-to-seed spread reaches
      89.3 deg. Directions are drawn double-headed because an eigenvector's
      sign is arbitrary.

  (b) The autocovariance spectrum at the selected lag, per seed. lambda1 and
      lambda2 are degenerate to ~3%, so no unique first direction exists; the
      pair is separated from the rest of the spectrum by ~2000x, which is what
      makes the two-dimensional subspace identifiable (Davis-Kahan).

Everything is read from the full P3 run; the density contours are the analytic
mixture used to generate it (four_circular_gmm: R=3, var=0.5, fast_var=0.05).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "experiments/robustness/geometry/results/p3_scope/p3_results_full_d10_a_fc.json"
OUT = ROOT / "experiments/robustness/figures/p3_da_subspace"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRID = "#d9d8d4"
INK = "#52514e"
TEXT = "#0b0b0b"

# Generating parameters of four_circular_gmm (experiments/robustness/geometry/p3_scope_controls.py).
R, VAR = 3.0, 0.5


def plane_density(x, y):
    """True 4-mode mixture density restricted to the slow plane."""
    ang = np.linspace(0.0, 2 * np.pi, 4, endpoint=False)
    out = np.zeros_like(x)
    for a in ang:
        mx, my = R * np.cos(a), R * np.sin(a)
        out += np.exp(-((x - mx) ** 2 + (y - my) ** 2) / (2 * VAR))
    return out / 4.0


def main() -> None:
    data = json.loads(RESULT.read_text())
    res = data["results"][0]
    st = res["da_stats"]
    seeds = res["da_per_seed"]

    da1 = np.asarray([s["da1"] for s in seeds], dtype=float)  # (n_seeds, 10)
    evals = np.asarray([s["evals"] for s in seeds], dtype=float)  # (n_seeds, 10)
    da1_ang = np.asarray(st["da1_angle_deg"], dtype=float)
    sub_ang = np.asarray(st["known_plane"]["max_principal_angle_deg"], dtype=float)

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
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))

    # ---------------- (a) DA1 inside the recovered plane -------------------
    ax = axes[0]
    lim = 5.6
    g = np.linspace(-lim, lim, 400)
    X, Y = np.meshgrid(g, g)
    ax.contourf(X, Y, plane_density(X, Y), levels=9, cmap="Blues", alpha=0.55)

    ramp = plt.cm.autumn(np.linspace(0.0, 0.75, len(seeds)))
    for i, (v, col) in enumerate(zip(da1, ramp)):
        u = v[:2]
        n = np.linalg.norm(u)
        if n == 0:
            continue
        u = u / n * (lim * 0.88)
        # Sign of an eigenvector is arbitrary -> draw the axis, not an arrow.
        ax.plot(
            [-u[0], u[0]],
            [-u[1], u[1]],
            lw=1.9,
            color=col,
            solid_capstyle="round",
            zorder=3,
        )

    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$e_1$")
    ax.set_ylabel(r"$e_2$")
    ax.set_title("(a)", loc="left", fontweight="bold")
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    # ---------------- (b) spectrum: degenerate pair, huge gap --------------
    ax = axes[1]
    # Show the informative head of the spectrum. Beyond ~index 6 the eigenvalues
    # are at or below the symmetrization noise (some are <=0), which a log axis
    # cannot render honestly, so they are omitted rather than clipped to a floor.
    n_show = 6
    idx = np.arange(1, n_show + 1)
    for i, (ev, col) in enumerate(zip(evals, ramp)):
        vis = ev[:n_show].astype(float)
        ax.plot(
            idx[vis > 0],
            vis[vis > 0],
            marker="o",
            ms=4,
            lw=1.2,
            color=col,
            alpha=0.9,
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xlabel(r"eigenvalue index of $\widehat C^{\mathrm{sym}}(\tau^*)$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title("(b)", loc="left", fontweight="bold")

    lam12 = float(np.mean(evals[:, :2]))
    ax.set_ylim(top=lam12 * 6)
    ax.grid(True, color=GRID, lw=0.5, which="major")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}.pdf / .png")
    print(f"  S_hat={res['ssa']['S_hat_mean']:.3f} +/- {res['ssa']['S_hat_std']:.3f}"
          f" (window-limited={res['ssa']['window_limited']})")
    print(f"  matched Gaussian={res['matched_gaussian']['S_hat_mean']:.3f}"
          f"  ratio={res['matched_gaussian']['ssa_ratio_fourmode_over_gaussian']:.2f}x")
    print(f"  lambda1/lambda2={st['gap_ratio_mean']:.4f}"
          f"  lambda2/lambda3={st['gap_ratio_23_mean']:.1f}")
    print(f"  DA1 angle vs seed 0 (deg): {np.round(da1_ang,2).tolist()}")
    print(f"  subspace max principal angle (deg): {np.round(sub_ang,3).tolist()}")


if __name__ == "__main__":
    main()
