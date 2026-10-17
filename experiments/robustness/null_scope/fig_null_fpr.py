#!/usr/bin/env python3
"""P10, single panel: false-positive rate of the analytic edge under score error.

Replaces the eight-panel `p10_score_noise_null_spectrum` figure. The subsection's
claim is a regime contrast, and it is two lines:

  * at the SDXL-like aspect ratio (d=768, N=200) the analytic edge is unaffected
    across the whole tested range -- zero exceedances in 100 replicates at every
    alpha;
  * at the alanine-like aspect ratio (d=40, N=2000) it holds to alpha=0.25 and
    then fails abruptly: 0.13 -> 0.98 between alpha=0.25 and 0.5.

Plotting the full alpha range is the point. Quoting only alpha<=0.25, as the
earlier text did, makes a cliff look like mild sensitivity.

Rates are exceedances of the exact-score analytic edge under an
isotropic-Gaussian null, nonconservative error family, one independent pair per
trajectory. Bands are the Wilson 95% intervals already stored in the JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "experiments/robustness/null_scope/results/score_error_null_spectrum.json"
OUT = ROOT / "experiments/robustness/figures/p10_null_fpr"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRID = "#d9d8d4"
INK = "#52514e"
TEXT = "#0b0b0b"

ESTIMATOR = "simple_Ntraj"
KEY = "vs_exact_score_analytic_edge"


def series(reg_block):
    """(alpha, rate, lo, hi) for the nonconservative family; alpha=0 is shared."""
    out = {}
    for c in reg_block["cells"]:
        if c["estimator"] != ESTIMATOR:
            continue
        if c["family"] == "nonconservative" or c["alpha"] == 0.0:
            f = c["fpr"][KEY]
            out[float(c["alpha"])] = (f["rate"], f["ci95"][0], f["ci95"][1])
    a = np.array(sorted(out))
    r = np.array([out[x][0] for x in a])
    lo = np.array([out[x][1] for x in a])
    hi = np.array([out[x][2] for x in a])
    return a, r, lo, hi


def main() -> None:
    d = json.loads(RESULT.read_text())
    regs = [("alanine-like", BLUE), ("sdxl-like", ORANGE)]

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
    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    xs = None
    for name, col in regs:
        blk = d["results"][name]
        a, r, lo, hi = series(blk)
        xs = np.arange(len(a))
        cfg = blk["config"]
        ax.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(xs, r, marker="o", ms=5, lw=1.8, color=col, zorder=3,
                label=rf"{name}  ($d={cfg['d']}$, $N={cfg['n_traj']}$)")
        print(f"  {name}: alphas={a.tolist()}")
        print(f"    rate={r.tolist()}")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in series(d["results"]["alanine-like"])[0]])
    ax.set_xlabel(r"injected score error  $\alpha$")
    ax.set_ylabel("rate of exceeding the analytic edge")
    ax.set_ylim(-0.04, 1.06)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}.pdf / .png")
    print(f"  n_rep={d['results']['alanine-like']['config']['n_rep']}"
          f"  estimator={ESTIMATOR}  threshold={KEY}  smoke={d['smoke']}")


if __name__ == "__main__":
    main()
