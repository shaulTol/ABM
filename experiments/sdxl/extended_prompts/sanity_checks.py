"""
Post-processing sanity checks for extended prompts sweep.
Run after all 3 phases complete.

Produces:
  - sanity_check_report.md
  - extended_sweep_results.json
  - table_extended_sweep.tex
  - table_peak_probs.tex
  - table_convergence.tex

Usage:
    python sanity_checks.py [--results_dir .]
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path


PROMPTS = [
    "a photo of a bat",
    "a photo of a crane",
    "a photo of a mouse",
    "a photo of a seal",
    "a photo of a jaguar",
    "a photo of a palm",
]
LABELS = ["bat", "crane", "mouse", "seal", "jaguar", "palm"]
GS_VALUES = [1.0, 3.0, 7.5, 15.0]
GS_LABELS = ["gs1p0", "gs3p0", "gs7p5", "gs15p0"]
N_TRAJ = 200
NOISE_FLOOR = 1.0 / N_TRAJ  # 0.005


def load_ssa_json(results_dir, pi, label, gs_label):
    """Load SSA JSON for a cell."""
    name = f"clip_da_ext_p{pi}_{label}_{gs_label}_dt0p005.json"
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_entropy_json(results_dir, pi, label, gs_label):
    """Load entropy JSON for a cell."""
    name = f"gen_ext_p{pi}_{label}_{gs_label}_entropy.json"
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def bootstrap_ssa(rho_sq, dt, B=10000, seed=42):
    """Paired lag bootstrap for SSA SE and paired comparisons."""
    rng = np.random.default_rng(seed)
    L = len(rho_sq)
    boot_ssa = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, L, size=L)
        boot_ssa[b] = dt * np.sum(rho_sq[idx])
    return boot_ssa


def check_monotonic_decay(rho_sq, window=20):
    """Check if rho_sq decays monotonically on average (smoothed)."""
    if len(rho_sq) < 2 * window:
        return True, 0
    smoothed = np.convolve(rho_sq, np.ones(window) / window, mode='valid')
    violations = np.sum(np.diff(smoothed) > 0)
    return violations == 0, int(violations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=".")
    args = parser.parse_args()
    rd = args.results_dir

    dt = 0.005
    results = {}  # label -> gs -> {ssa, entropy, rho_sq_T, ...}
    report_lines = ["# Sanity Check Report\n"]
    missing = []

    # ─── Load all data ────────────────────────────────────────────────────
    for pi, label in enumerate(LABELS):
        results[label] = {}
        for gi, (gs, gs_label) in enumerate(zip(GS_VALUES, GS_LABELS)):
            ssa_data = load_ssa_json(rd, pi, label, gs_label)
            ent_data = load_entropy_json(rd, pi, label, gs_label)

            if ssa_data is None:
                missing.append(f"SSA: ext_p{pi}_{label}_{gs_label}_dt0p005.json")
                results[label][gs] = None
                continue

            rho_sq_all = np.array(ssa_data.get("rho_sq_all", []))
            ssa_fixed20 = ssa_data.get("ssa_fixed20")
            entropy_k5_d50 = None
            if ent_data:
                entropy_k5_d50 = ent_data["entropy"]["knn"]["k5"]["50"]

            # rho^2(T=2000) = last element of rho_sq_all
            rho_sq_T = rho_sq_all[-1] if len(rho_sq_all) > 0 else None

            results[label][gs] = {
                "ssa_fixed20": ssa_fixed20,
                "entropy_k5_d50": entropy_k5_d50,
                "rho_sq_T": float(rho_sq_T) if rho_sq_T is not None else None,
                "rho_sq_all": rho_sq_all.tolist(),
                "hit_floor": ssa_data.get("hit_floor_fixed20", None),
            }

    if missing:
        report_lines.append("## Missing Files\n")
        for m in missing:
            report_lines.append(f"- {m}\n")
        report_lines.append("\n")

    # ─── Check 1: Monotonic decay ────────────────────────────────────────
    report_lines.append("## Check 1: Monotonic ρ²(τ) decay\n")
    report_lines.append("| Prompt | w | Monotonic? | Violations (smoothed) |\n")
    report_lines.append("|--------|---|-----------|----------------------|\n")
    for label in LABELS:
        for gs in GS_VALUES:
            cell = results[label].get(gs)
            if cell is None:
                report_lines.append(f"| {label} | {gs} | MISSING | — |\n")
                continue
            rho_sq = np.array(cell["rho_sq_all"])
            mono, viol = check_monotonic_decay(rho_sq)
            flag = "✓" if mono else f"**FLAG** ({viol} violations)"
            report_lines.append(f"| {label} | {gs} | {flag} | {viol} |\n")
    report_lines.append("\n")

    # ─── Check 2: Convergence (ρ²(T) < 1/N) ─────────────────────────────
    report_lines.append("## Check 2: Convergence — ρ²(T=2000) vs 1/N=0.005\n")
    report_lines.append("| Prompt | w | ρ²(T) | Converged? |\n")
    report_lines.append("|--------|---|-------|----------|\n")
    convergence_table = {}
    for label in LABELS:
        convergence_table[label] = {}
        for gs in GS_VALUES:
            cell = results[label].get(gs)
            if cell is None:
                report_lines.append(f"| {label} | {gs} | MISSING | — |\n")
                convergence_table[label][gs] = None
                continue
            rho_sq_T = cell["rho_sq_T"]
            converged = rho_sq_T < NOISE_FLOOR if rho_sq_T is not None else None
            flag = "✓" if converged else "†"
            report_lines.append(f"| {label} | {gs} | {rho_sq_T:.6f} | {flag} |\n")
            convergence_table[label][gs] = rho_sq_T
    report_lines.append("\n")

    # ─── Check 3: Paired bootstrap ───────────────────────────────────────
    report_lines.append("## Check 3: Paired bootstrap Pr[Ŝ(7.5) > Ŝ(w)]\n")
    report_lines.append("| Prompt | Pr[Ŝ₇.₅ > Ŝ₁] | Pr[Ŝ₇.₅ > Ŝ₃] | Pr[Ŝ₇.₅ > Ŝ₁₅] |\n")
    report_lines.append("|--------|---------------|---------------|----------------|\n")
    peak_probs = {}
    for label in LABELS:
        cell_7p5 = results[label].get(7.5)
        if cell_7p5 is None or len(cell_7p5["rho_sq_all"]) == 0:
            report_lines.append(f"| {label} | MISSING | MISSING | MISSING |\n")
            continue

        rho_sq_7p5 = np.array(cell_7p5["rho_sq_all"])
        boot_7p5 = bootstrap_ssa(rho_sq_7p5, dt, B=10000)
        probs = {}
        for w_compare in [1.0, 3.0, 15.0]:
            cell_w = results[label].get(w_compare)
            if cell_w is None or len(cell_w["rho_sq_all"]) == 0:
                probs[w_compare] = None
                continue
            rho_sq_w = np.array(cell_w["rho_sq_all"])
            boot_w = bootstrap_ssa(rho_sq_w, dt, B=10000)
            prob = float(np.mean(boot_7p5 > boot_w))
            probs[w_compare] = prob

        peak_probs[label] = probs
        p1 = f"{probs[1.0]:.3f}" if probs[1.0] is not None else "—"
        p3 = f"{probs[3.0]:.3f}" if probs[3.0] is not None else "—"
        p15 = f"{probs[15.0]:.3f}" if probs[15.0] is not None else "—"
        report_lines.append(f"| {label} | {p1} | {p3} | {p15} |\n")
    report_lines.append("\n")

    # ─── Check 4: Peak SSA > 1 ───────────────────────────────────────────
    report_lines.append("## Check 4: Peak SSA vs Gaussian baseline (>1)\n")
    report_lines.append("| Prompt | Peak SSA | Peak w | Above 1? |\n")
    report_lines.append("|--------|----------|--------|----------|\n")
    for label in LABELS:
        ssas = []
        for gs in GS_VALUES:
            cell = results[label].get(gs)
            if cell and cell["ssa_fixed20"] is not None:
                ssas.append(cell["ssa_fixed20"])
            else:
                ssas.append(-np.inf)
        peak_val = max(ssas)
        peak_gs = GS_VALUES[np.argmax(ssas)]
        above = "✓" if peak_val > 1.0 else "**below 1**"
        report_lines.append(f"| {label} | {peak_val:.3f} | {peak_gs} | {above} |\n")
    report_lines.append("\n")

    # ─── Check 5: Entropy monotonically decreases ────────────────────────
    report_lines.append("## Check 5: Entropy monotonically decreases with w\n")
    report_lines.append("| Prompt | H(1) | H(3) | H(7.5) | H(15) | Monotone? |\n")
    report_lines.append("|--------|------|------|--------|-------|-----------|\n")
    for label in LABELS:
        hs = []
        for gs in GS_VALUES:
            cell = results[label].get(gs)
            if cell and cell["entropy_k5_d50"] is not None:
                hs.append(cell["entropy_k5_d50"])
            else:
                hs.append(None)
        if None in hs:
            report_lines.append(f"| {label} | — | — | — | — | MISSING |\n")
            continue
        mono = all(hs[i] >= hs[i + 1] for i in range(len(hs) - 1))
        flag = "✓" if mono else "**FLAG**"
        report_lines.append(
            f"| {label} | {hs[0]:.1f} | {hs[1]:.1f} | {hs[2]:.1f} | {hs[3]:.1f} | {flag} |\n"
        )
    report_lines.append("\n")

    # ─── Save report ─────────────────────────────────────────────────────
    report_path = os.path.join(rd, "sanity_check_report.md")
    with open(report_path, "w") as f:
        f.writelines(report_lines)
    print(f"Saved: {report_path}")

    # ─── Save extended_sweep_results.json ────────────────────────────────
    # Strip rho_sq_all (too large for JSON readability)
    results_slim = {}
    for label in LABELS:
        results_slim[label] = {}
        for gs in GS_VALUES:
            cell = results[label].get(gs)
            if cell is None:
                results_slim[label][str(gs)] = None
                continue
            results_slim[label][str(gs)] = {
                "ssa_fixed20": cell["ssa_fixed20"],
                "entropy_k5_d50": cell["entropy_k5_d50"],
                "rho_sq_T": cell["rho_sq_T"],
                "converged": cell["rho_sq_T"] < NOISE_FLOOR if cell["rho_sq_T"] else None,
            }

    json_path = os.path.join(rd, "extended_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(results_slim, f, indent=2)
    print(f"Saved: {json_path}")

    # ─── Generate LaTeX tables ────────────────────────────────────────────
    # Table: SSA with bootstrap SE
    tex_lines = []
    tex_lines.append("\\begin{tabular}{l c c c c}")
    tex_lines.append("\\toprule")
    tex_lines.append("Prompt & $w=1$ & $w=3$ & $w=7.5$ & $w=15$ \\\\")
    tex_lines.append("\\midrule")
    for pi, label in enumerate(LABELS):
        ssas = []
        ses = []
        daggers = []
        for gs in GS_VALUES:
            cell = results[label].get(gs)
            if cell is None:
                ssas.append(None)
                ses.append(None)
                daggers.append(False)
                continue
            ssas.append(cell["ssa_fixed20"])
            rho_sq = np.array(cell["rho_sq_all"])
            boot = bootstrap_ssa(rho_sq, dt, B=10000, seed=42 + pi * 10 + GS_VALUES.index(gs))
            ses.append(np.std(boot))
            daggers.append(cell["rho_sq_T"] >= NOISE_FLOOR if cell["rho_sq_T"] is not None else False)

        peak_idx = np.argmax([s if s is not None else -np.inf for s in ssas])
        cells = []
        for i, (s, se, dag) in enumerate(zip(ssas, ses, daggers)):
            if s is None:
                cells.append("—")
                continue
            dagger = "$^\\dagger$" if dag else ""
            val = f"{s:.2f}{dagger} ({se:.2f})"
            if i == peak_idx:
                val = f"\\textbf{{{s:.2f}}}{dagger} ({se:.2f})"
            cells.append(val)

        tex_lines.append(f"  {PROMPTS[pi]} & {' & '.join(cells)} \\\\")

    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")

    tex_path = os.path.join(rd, "table_extended_sweep.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Saved: {tex_path}")

    # Table: peak probs
    tex_lines2 = []
    tex_lines2.append("\\begin{tabular}{l c c c}")
    tex_lines2.append("\\toprule")
    tex_lines2.append("Prompt & $\\Pr[\\hat S_{7.5} > \\hat S_1]$ & $\\Pr[\\hat S_{7.5} > \\hat S_3]$ & $\\Pr[\\hat S_{7.5} > \\hat S_{15}]$ \\\\")
    tex_lines2.append("\\midrule")
    for label in LABELS:
        probs = peak_probs.get(label, {})
        p1 = f"{probs.get(1.0, 0):.3f}" if probs.get(1.0) is not None else "—"
        p3 = f"{probs.get(3.0, 0):.3f}" if probs.get(3.0) is not None else "—"
        p15 = f"{probs.get(15.0, 0):.3f}" if probs.get(15.0) is not None else "—"
        tex_lines2.append(f"  {label} & {p1} & {p3} & {p15} \\\\")
    tex_lines2.append("\\bottomrule")
    tex_lines2.append("\\end{tabular}")

    tex_path2 = os.path.join(rd, "table_peak_probs.tex")
    with open(tex_path2, "w") as f:
        f.write("\n".join(tex_lines2) + "\n")
    print(f"Saved: {tex_path2}")

    # Table: convergence
    tex_lines3 = []
    tex_lines3.append("\\begin{tabular}{l c c c c}")
    tex_lines3.append("\\toprule")
    tex_lines3.append("Prompt & $w=1$ & $w=3$ & $w=7.5$ & $w=15$ \\\\")
    tex_lines3.append("\\midrule")
    for label in LABELS:
        cells = []
        for gs in GS_VALUES:
            v = convergence_table[label].get(gs)
            if v is None:
                cells.append("—")
            else:
                flag = "$^\\dagger$" if v >= NOISE_FLOOR else ""
                cells.append(f"{v:.4f}{flag}")
        tex_lines3.append(f"  {label} & {' & '.join(cells)} \\\\")
    tex_lines3.append("\\bottomrule")
    tex_lines3.append("\\end{tabular}")

    tex_path3 = os.path.join(rd, "table_convergence.tex")
    with open(tex_path3, "w") as f:
        f.write("\n".join(tex_lines3) + "\n")
    print(f"Saved: {tex_path3}")

    print("\nDone. All deliverables saved.")


if __name__ == "__main__":
    main()
