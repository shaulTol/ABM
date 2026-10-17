#!/usr/bin/env python3
"""P1 score-error figure: what an approximate score costs SSA and DA.

Two panels, following the package convention that panels carry no prose --
titles are bare "(a)"/"(b)" and every number lives in the caption.

  (a) SSA -> MI rank correlation vs injected score error. Answers the reviewer
      question directly: does the ranking survive an approximate score, and at
      what error level does it stop surviving.

  (b) DA1 angular deviation from the UNPERTURBED direction, median over the
      five representative mixtures (top-5 by true MI) with a min-max band.
      The reference is each mixture's own alpha=0 run -- same code path, same
      seed, same trajectories -- so alpha=0 is exactly 0 by construction and
      the curve isolates the effect of the perturbation alone. (The earlier
      `angle_from_base_deg` field used a reference from a different code path,
      da_convergence_profile(), and carried a spurious 0.5-3 deg offset.)

Absolute finite-horizon SSA value error (the P9 analysis) is reported in the
appendix prose and in p9_absolute_sensitivity.json; it is not plotted here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RES_DIR = ROOT / "experiments/robustness/score_error/results"
RESULT = RES_DIR / "score_robustness_full_fc.json"
DA_RESULT = RES_DIR / "score_robustness_full_fc_da.json"
OUT = ROOT / "experiments/robustness/figures/p1_score_error_ranking"

BLUE = "#2a78d6"
GRID = "#d9d8d4"
INK = "#52514e"
TEXT = "#0b0b0b"

# Sequential ramp: the five mixtures are ordered by true MI.
DA_RAMP = ["#c2d9f2", "#93bde8", "#5f9bd8", "#3376c0", "#14508f"]


def main() -> None:
    data = json.loads(RESULT.read_text())
    cfg = data["config"]
    sp = data["additive"]["spearman"]

    # JSON keys are the repr of the alpha floats ("0.0", "0.25", ...); index by
    # float value rather than by a reformatted string.
    by_alpha = {float(k): v for k, v in sp.items()}
    alphas = np.asarray(sorted(by_alpha), dtype=float)
    mean = np.asarray([by_alpha[a]["mean"] for a in alphas], dtype=float)
    sd = np.asarray([by_alpha[a]["sd"] for a in alphas], dtype=float)
    assert set(alphas) == set(cfg["additive_alphas"]), "alpha grid mismatch vs config"

    da = json.loads(DA_RESULT.read_text())
    ang = []
    for g in da["da"]:
        lv = {l["level"]: l["angle_from_unperturbed_deg"] for l in g["levels"]
              if l["family"] == "additive"}
        ang.append([lv[a] for a in alphas])
    ang = np.asarray(ang, dtype=float)          # (n_gmm, n_alpha)
    assert np.allclose(ang[:, 0], 0.0, atol=1e-3), \
        "alpha=0 must be exactly 0 under the paired reference"
    da_med, da_lo, da_hi = np.median(ang, 0), ang.min(0), ang.max(0)

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
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))

    # ---------------- (a) ranking ------------------------------------------
    ax = axes[0]
    ax.fill_between(alphas, mean - sd, mean + sd, color=BLUE, alpha=0.18, lw=0)
    ax.plot(alphas, mean, marker="o", ms=5, lw=1.8, color=BLUE, zorder=3)
    ax.axhline(mean[0], color=INK, ls="--", lw=1.0, zorder=1)
    ax.set_xlabel(r"injected score error  $\alpha$")
    ax.set_ylabel(r"Spearman$(\widehat S,\ \mathrm{MI})$")
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_ylim(0.70, 0.92)
    ax.set_title("(a)", loc="left", fontweight="bold")

    # ---------------- (b) DA direction -------------------------------------
    ax = axes[1]
    ax.fill_between(alphas, da_lo, da_hi, color=BLUE, alpha=0.18, lw=0)
    ax.plot(alphas, da_med, marker="o", ms=5, lw=1.8, color=BLUE, zorder=3)
    ax.set_xlabel(r"injected score error  $\alpha$")
    ax.set_ylabel(r"DA$_1$ rotation (deg)")
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_ylim(bottom=0.0)
    ax.set_title("(b)", loc="left", fontweight="bold")

    for ax in axes:
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")

    print(f"wrote {OUT}.pdf / .png")
    print(f"  (a) n_gmm={cfg['n_gmm']}  seeds={cfg['seeds']}  d={cfg['d']}  N={cfg['N']}")
    for a, m, s in zip(alphas, mean, sd):
        print(f"      alpha={a:<5g} spearman={m:.4f} +/- {s:.4f}")
    print(f"  (b) {len(ang)} representative mixtures ({da['representative_rule']}),"
          f" ids={da['representative_gmm_ids']}; paired reference = own alpha=0 run")
    for j, a in enumerate(alphas):
        print(f"      alpha={a:<5g} median={da_med[j]:5.2f} deg"
              f"  range=[{da_lo[j]:.2f}, {da_hi[j]:.2f}] deg")


if __name__ == "__main__":
    main()
