"""
SDXL guidance-sweep SSA tables (Table 1, the SE table and the full 14-prompt table).

Reads the per-cell trajectory-bootstrap results written by
`traj_averaged_pipeline.py --mode bootstrap` (cluster/run_bootstrap_ssa.sh):
results/bootstrap_pooled/bootstrap_<cell>.json, each holding the pooled Shat
(eq:c_hat_def, 1/N stopping rule), its bootstrap SE and the B replicates.

Prints Markdown tables and, per prompt, the bootstrap probability that the
maximizing guidance scale is intermediate (w in {3, 7.5}) and that it equals
the point-estimate peak. Prompts whose peak Shat is below CUTOFF are reported
separately (excluded from Table 1).

Usage:
  python table_sdxl_ssa.py [--results results/bootstrap_pooled] [--cutoff 1.1]
"""
import argparse
import glob
import json
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORDER = ["person", "elderly", "cat", "food", "car", "bat", "crane", "mouse",
         "seal", "jaguar", "palm", "mountains", "flowers", "street"]
WS = ["1", "3", "7.5", "15"]


def load(results_dir):
    out = {}
    for f in glob.glob(os.path.join(results_dir, "bootstrap_*.json")):
        d = json.load(open(f))
        for name, r in d.items():
            base = name.replace("ext_", "").replace("traj_", "")
            prompt = base.split("_gs")[0].split("_")[-1]
            w = base.split("_gs")[1].replace("_dt0p005", "").replace("p0", "").replace("p5", ".5")
            out.setdefault(prompt, {})[w] = r
    return out


def peak(cells):
    return max(WS, key=lambda w: cells[w]["ssa_traj"])


def table(R, prompts, with_se):
    lines = ["| w | " + " | ".join(prompts) + " |", "|---" * (len(prompts) + 1) + "|"]
    for w in WS:
        row = []
        for p in prompts:
            s = f"{R[p][w]['ssa_traj']:.2f}"
            if with_se:
                s += f" ({R[p][w]['se']:.2f})"
            row.append(f"**{s}**" if peak(R[p]) == w else s)
        lines.append(f"| {w} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(SCRIPT_DIR, "results", "bootstrap_pooled"))
    ap.add_argument("--cutoff", type=float, default=1.1)
    a = ap.parse_args()

    R = load(a.results)
    prompts = [p for p in ORDER if p in R and all(w in R[p] for w in WS)]
    body = [p for p in prompts if R[p][peak(R[p])]["ssa_traj"] >= a.cutoff]
    excluded = [p for p in prompts if p not in body]

    n_cells = sum(len(R[p]) for p in prompts)
    n_conv = sum(bool(R[p][w].get("converged_1overN")) for p in prompts for w in WS)
    print(f"{len(prompts)} prompts, {n_cells} cells; {n_conv} reach the 1/N floor; "
          f"max SE {max(R[p][w]['se'] for p in prompts for w in WS):.3f}")

    print(f"\n## Table 1: peak Shat >= {a.cutoff} ({len(body)} prompts)\n")
    print(table(R, body, with_se=False))
    print("\n## With bootstrap SEs\n")
    print(table(R, body, with_se=True))
    print(f"\n## Excluded (peak Shat < {a.cutoff}): {', '.join(excluded)}\n")
    print(table(R, excluded, with_se=True))

    print("\n## Bootstrap peak location (cells independent; replicates compared index-wise)\n")
    print("| prompt | peak w | Pr[intermediate peak] | Pr[same peak w] |")
    print("|---|---|---|---|")
    for p in prompts:
        B = np.array([R[p][w]["boot"] for w in WS])
        am = B.argmax(0)
        pk = WS.index(peak(R[p]))
        print(f"| {p} | {WS[pk]} | {np.mean((am == 1) | (am == 2)):.3f} | {np.mean(am == pk):.3f} |")


if __name__ == "__main__":
    main()
