#!/usr/bin/env python3
"""P2 numerical robustness, as two single-panel figures.

Replaces the three-panel `p2_numerical_robustness` figure with one figure per
knob, so each carries a single claim:

  p2_robustness_N.{pdf,png}   -- precision improves with the trajectory count.
                                 The +/- SD band narrows and Spearman(S_hat, MI)
                                 rises 0.757 -> 0.943 across N = 50..400.
  p2_robustness_dt.{pdf,png}  -- the estimate is insensitive to the step size.
                                 A 4x range of dt moves the representative
                                 S_hat by ~1.7% with no ordering inversion.

Both share axes so they can be read side by side.

Note the P2 design is a *star*, not a crossed grid: 6 cells = baseline
(N=200, dt=0.01) plus 2 dt variants plus 3 N variants. No N x dt interaction is
measured, so none is implied. The baseline cell is common to both figures.

The dropped panel (b) series "vs baseline profile" is not replotted: it compared
the N=200 cell against itself and so was pinned to 1.00 by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
# First-crossing (paper) results, rebuilt from the saved rho^2 by
# compute_p2_numerical_robustness.py --from_npz ... --out_tag _fc (2026-10-17).
# The original largest-lag run is results/p2_numerical_robustness.json.
RESULT = ROOT / "experiments/robustness/numerics/results/p2_numerical_robustness_fc.json"
OUTDIR = ROOT / "experiments/robustness/figures"

GRID = "#d9d8d4"
INK = "#52514e"
TEXT = "#0b0b0b"

# Sequential single-hue ramps: the encoded variable is ordered magnitude in both
# panels, so a categorical palette would be the wrong encoding.
N_RAMP = ["#a3c8e8", "#6ba6dc", "#3d84c8", "#14508f"]
DT_RAMP = ["#f6b48f", "#ef8a50", "#c85a1b"]


def style() -> None:
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


def panel(seps, cells, ramp, legend_title, title, ylim, out):
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for (label, cell), color in zip(cells, ramp):
        m = np.asarray(cell["mean_S"], dtype=float)
        s = np.asarray(cell["sd_S"], dtype=float)
        rho = cell["spearman_S_MI_mean"]
        ax.fill_between(seps, m - s, m + s, color=color, alpha=0.30, lw=0)
        ax.plot(
            seps,
            m,
            marker="o",
            ms=4.5,
            lw=1.7,
            color=color,
            zorder=3,
            label=rf"{label}   $\rho={rho:.3f}$",
        )
    ax.set_xlabel("mode separation")
    ax.set_ylabel(r"finite-horizon $\widehat S$")
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left", labelspacing=0.3)
    leg.set_title(legend_title, prop={"size": 8})
    leg._legend_box.align = "left"
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")


def main() -> None:
    data = json.loads(RESULT.read_text())
    summary = data["summary_primary_1N"]
    seps = np.asarray(data["separations"], dtype=float)
    base = data["baseline"]

    n_cells = [
        (rf"$N={n}$", summary[k])
        for n, k in [(50, "N50_dt0p01"), (100, "N100_dt0p01"),
                     (200, "N200_dt0p01"), (400, "N400_dt0p01")]
    ]
    dt_cells = [
        (rf"$\Delta t={dt:g}$", summary[k])
        for dt, k in [(0.005, "N200_dt0p005"), (0.01, "N200_dt0p01"),
                      (0.02, "N200_dt0p02")]
    ]

    # Shared y-range so the two figures are directly comparable.
    allc = [c for _, c in n_cells + dt_cells]
    lo = min(float(np.min(np.asarray(c["mean_S"]) - np.asarray(c["sd_S"]))) for c in allc)
    hi = max(float(np.max(np.asarray(c["mean_S"]) + np.asarray(c["sd_S"]))) for c in allc)
    pad = 0.06 * (hi - lo)
    ylim = (lo - pad, hi + pad)

    style()
    panel(
        seps, n_cells, N_RAMP,
        rf"trajectories, at $\Delta t={base['dt']:g}$",
        "(a) Precision improves with $N$",
        ylim, OUTDIR / "p2_robustness_N",
    )
    panel(
        seps, dt_cells, DT_RAMP,
        rf"step size, at $N={base['N']}$",
        r"(b) Insensitive to $\Delta t$",
        ylim, OUTDIR / "p2_robustness_dt",
    )

    print(f"  d={data['D']}  T_phys={data['T_phys']}  n_seeds={data['n_seeds']}"
          f"  baseline={base['key']}")
    for label, c in n_cells + dt_cells:
        print(
            f"  {label:<18} spearman={c['spearman_S_MI_mean']:.3f}"
            f" +/- {c['spearman_S_MI_sd']:.3f}"
            f"  rep_S={c['rep_mean_S']:.4f} +/- {c['rep_sd_S']:.4f}"
        )


if __name__ == "__main__":
    main()
