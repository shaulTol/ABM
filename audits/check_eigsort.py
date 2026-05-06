"""
Audit: abs-sort vs value-sort for SDXL DA eigenvalues.

Checks whether sorting eigenvalues of C_sym(tau*) by absolute value
(as the code does) vs by value (descending) ever changes which
eigenvectors are selected as DA1/DA2.
"""
import numpy as np
import os, sys, json

BASE = "/Users/stolk/github/ssa-paper"
DA_DIR = os.path.join(BASE, "experiments/sdxl/results_da_pcavda")

def check_eigsort(npz_path, label, out_lines):
    """Check one NPZ file for abs-sort vs value-sort discrepancy."""
    d = np.load(npz_path, allow_pickle=True)

    # Try to get full eigenvalues at tau*
    if "eigenvalues_full_at_tau_star" in d:
        eigvals = d["eigenvalues_full_at_tau_star"]
    else:
        # Find the highest lag with saved eigenvalues
        eig_keys = sorted([k for k in d.keys() if k.startswith("eigenvalues_lag")],
                          key=lambda k: int(k.split("lag")[1]))
        if not eig_keys:
            out_lines.append(f"\n{'='*60}")
            out_lines.append(f"  {label}: NO eigenvalue arrays found, skipping")
            return False
        eigvals = d[eig_keys[-1]]

    out_lines.append(f"\n{'='*60}")
    out_lines.append(f"  {label}")
    out_lines.append(f"  File: {os.path.basename(npz_path)}")
    out_lines.append(f"  Eigenvalue array shape: {eigvals.shape}")
    out_lines.append(f"{'='*60}")

    n = len(eigvals)
    show = min(10, n)

    # Sort by abs descending (what the code does)
    idx_abs = np.argsort(np.abs(eigvals))[::-1]
    eig_abs = eigvals[idx_abs]

    # Sort by value descending (standard)
    idx_val = np.argsort(eigvals)[::-1]
    eig_val = eigvals[idx_val]

    out_lines.append(f"\n  Top-{show} eigenvalues (abs-sort, as in code):")
    out_lines.append(f"  {'Rank':>4}  {'Index':>5}  {'Eigenvalue':>14}  {'|Eigenvalue|':>14}")
    for i in range(show):
        out_lines.append(f"  {i+1:>4}  {idx_abs[i]:>5}  {eig_abs[i]:>14.6e}  {abs(eig_abs[i]):>14.6e}")

    out_lines.append(f"\n  Top-{show} eigenvalues (value-sort, descending):")
    out_lines.append(f"  {'Rank':>4}  {'Index':>5}  {'Eigenvalue':>14}  {'|Eigenvalue|':>14}")
    for i in range(show):
        out_lines.append(f"  {i+1:>4}  {idx_val[i]:>5}  {eig_val[i]:>14.6e}  {abs(eig_val[i]):>14.6e}")

    # Check if orderings differ for top-m
    has_discrepancy = False
    for m in [1, 2, 3, 5, 10]:
        if m > n:
            break
        same = np.array_equal(idx_abs[:m], idx_val[:m])
        if not same:
            has_discrepancy = True
            out_lines.append(f"\n  *** DISCREPANCY at top-{m}: abs-sort gives indices {idx_abs[:m].tolist()}, "
                           f"value-sort gives {idx_val[:m].tolist()}")

    if not has_discrepancy:
        out_lines.append(f"\n  OK: abs-sort and value-sort give IDENTICAL top-{min(10,n)} ordering.")

    # Check for negative eigenvalues
    n_neg = np.sum(eigvals < 0)
    min_eig = eigvals.min()
    max_eig = eigvals.max()
    out_lines.append(f"\n  Eigenvalue stats: min={min_eig:.6e}, max={max_eig:.6e}, "
                   f"#negative={n_neg}/{n}")

    # The critical check: is any |negative eigenvalue| > smallest positive in top-10?
    neg_mask = eigvals < 0
    if neg_mask.any():
        max_neg_abs = np.abs(eigvals[neg_mask]).max()
        # Smallest positive eigenvalue in top-10 by value-sort
        pos_in_top10 = [eigvals[idx_val[i]] for i in range(show) if eigvals[idx_val[i]] > 0]
        if pos_in_top10:
            min_pos_top10 = min(pos_in_top10)
            if max_neg_abs > min_pos_top10:
                out_lines.append(f"  *** NEGATIVE eigenvalue (|{max_neg_abs:.6e}|) exceeds "
                               f"smallest positive in top-{show} ({min_pos_top10:.6e})")
                has_discrepancy = True
            else:
                out_lines.append(f"  Largest |negative|={max_neg_abs:.6e} < smallest positive in top-{show}={min_pos_top10:.6e}")

    # If we have C_tau_star, recompute eigdecomp both ways and compare DA1
    if "C_tau_star" in d and "da1_eigvec" in d:
        C = d["C_tau_star"]
        da1_saved = d["da1_eigvec"]

        eigvals_re, eigvecs_re = np.linalg.eigh(C)

        # abs-sort
        idx_a = np.argsort(np.abs(eigvals_re))[::-1]
        da1_abs = eigvecs_re[:, idx_a[0]]

        # value-sort
        idx_v = np.argsort(eigvals_re)[::-1]
        da1_val = eigvecs_re[:, idx_v[0]]

        cos_abs_val = abs(np.dot(da1_abs, da1_val))
        cos_abs_saved = abs(np.dot(da1_abs, da1_saved))

        out_lines.append(f"\n  Recomputed from C_tau_star:")
        out_lines.append(f"    |cos(DA1_abs, DA1_val)| = {cos_abs_val:.6f}")
        out_lines.append(f"    |cos(DA1_abs, DA1_saved)| = {cos_abs_saved:.6f}")
        out_lines.append(f"    Top-3 eigvals (abs-sort): {eigvals_re[idx_a[:3]].tolist()}")
        out_lines.append(f"    Top-3 eigvals (val-sort): {eigvals_re[idx_v[:3]].tolist()}")

        if cos_abs_val < 0.99:
            out_lines.append(f"    *** DA1 DIFFERS between abs-sort and value-sort!")
            has_discrepancy = True

    return has_discrepancy


def main():
    out_lines = []
    out_lines.append("SDXL DA Eigenvalue Sort Audit")
    out_lines.append("=" * 60)
    out_lines.append("Checking whether abs-sort vs value-sort changes DA directions.\n")

    any_discrepancy = False

    # 1. Canonical elderly (has C_tau_star)
    canonical = os.path.join(DA_DIR, "clip_da_elderly_gs7p5_fc_dt0p0025.npz")
    if os.path.exists(canonical):
        disc = check_eigsort(canonical, "CANONICAL: elderly (free-conv, dt=0.0025)", out_lines)
        any_discrepancy = any_discrepancy or disc

    # 2. All clip_da_*_n200.npz files
    for fname in sorted(os.listdir(DA_DIR)):
        if fname.startswith("clip_da_") and fname.endswith("_n200.npz"):
            path = os.path.join(DA_DIR, fname)
            disc = check_eigsort(path, fname, out_lines)
            any_discrepancy = any_discrepancy or disc

    # 3. Older clip_da files (non-n200)
    for fname in sorted(os.listdir(DA_DIR)):
        if fname.startswith("clip_da_") and fname.endswith(".npz") and "_n200" not in fname and "fc_dt" not in fname:
            path = os.path.join(DA_DIR, fname)
            disc = check_eigsort(path, fname, out_lines)
            any_discrepancy = any_discrepancy or disc

    # Summary
    out_lines.append(f"\n{'='*60}")
    out_lines.append("SUMMARY")
    out_lines.append(f"{'='*60}")
    if any_discrepancy:
        out_lines.append("*** DISCREPANCIES FOUND -- abs-sort and value-sort differ!")
    else:
        out_lines.append("NO discrepancies: abs-sort and value-sort give identical DA directions")
        out_lines.append("for all SDXL NPZ files checked.")

    output = "\n".join(out_lines)
    print(output)

    with open(os.path.join(BASE, "audits/eigsort_results.txt"), "w") as f:
        f.write(output + "\n")

if __name__ == "__main__":
    main()
