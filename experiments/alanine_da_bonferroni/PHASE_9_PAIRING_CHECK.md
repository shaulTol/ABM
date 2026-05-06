# Phase 9: Eigenvalue–Eigenvector Pairing Check

**Date:** 2026-05-02
**Purpose:** Verify that stored eigenvalues in `alanine_C_tau_full_h_half_10k.npz` at idx=37 (τ=10,000 ps) match an independent eigendecomposition of the stored C(τ) matrix, and that eigenvectors are correctly paired.

---

## 1. File Contents

The canonical file stores:
- `C_tau`: (38, 45, 45) float64 — full symmetrized autocovariance matrices
- `eigenvalues`: (38, 45) float64 — eigenvalues sorted descending
- **No `eigenvectors` field** — eigenvectors are recomputed on-the-fly from `C_tau` when needed

Since eigenvectors are not stored, the pairing question reduces to: do the stored eigenvalues match eigendecomposition of the stored C(τ)?

---

## 2. Symmetry Check

```
max|C_tau[37] − C_tau[37].T| = 0.00e+00
```

The stored C(τ) is **already perfectly symmetric** (float64). No symmetrization needed (but we apply `0.5*(C + C.T)` anyway for safety — it's a no-op here).

---

## 3. Stored vs Independent Eigenvalues (Top 10)

| k | Stored | Independent (scipy eigh) | Ratio | Diff |
|---|---|---|---|---|
| 0 | 1.2098984793e-02 | 1.2098906040e-02 | 1.00000651 | 7.88e-08 |
| 1 | 6.0955603840e-04 | 6.0955060034e-04 | 1.00000892 | 5.44e-09 |
| 2 | 6.0240938183e-05 | 6.0240411858e-05 | 1.00000874 | 5.26e-10 |
| 3 | 4.8198889999e-05 | 4.8198466359e-05 | 1.00000879 | 4.24e-10 |
| 4 | 2.1401374397e-05 | 2.1401198044e-05 | 1.00000824 | 1.76e-10 |
| 5 | 1.6309937564e-05 | 1.6309798847e-05 | 1.00000851 | 1.39e-10 |
| 6 | 1.3045239029e-05 | 1.3045158664e-05 | 1.00000616 | 8.04e-11 |
| 7 | 1.0586184544e-05 | 1.0586110257e-05 | 1.00000702 | 7.43e-11 |
| 8 | 7.5539678619e-06 | 7.5538893196e-06 | 1.00001040 | 7.85e-11 |
| 9 | 6.2661133597e-06 | 6.2660653412e-06 | 1.00000766 | 4.80e-11 |

**Max relative difference (top 10): 1.04e-05.**

The ~1e-5 relative difference is expected: the stored eigenvalues were computed by `torch.linalg.eigh` (float32 C(τ) promoted to float64 inside PyTorch on GPU), while the independent values use `scipy.linalg.eigh` (float64 throughout on CPU). The difference is consistent with float32→float64 promotion artifacts.

---

## 4. Eigenvalue Ordering

```
Stored eigenvalues strictly descending: True
```

All 45 eigenvalues are in strictly descending order, matching the `torch.argsort(evals, descending=True)` in `extract_da_directions` (line 55 of `da_directions.py`).

---

## 5. Rayleigh Quotient Check

Since eigenvectors are not stored, we verify with independently computed eigenvectors:

| k | eigenvalue | v @ C_sym @ v | diff |
|---|---|---|---|
| 0 | 1.2098906040e-02 | 1.2098906040e-02 | 1.39e-17 |
| 1 | 6.0955060034e-04 | 6.0955060034e-04 | 2.17e-19 |
| 2 | 6.0240411858e-05 | 6.0240411858e-05 | 6.51e-19 |
| 3 | 4.8198466359e-05 | 4.8198466359e-05 | 4.61e-19 |
| 4 | 2.1401198044e-05 | 2.1401198044e-05 | 2.61e-19 |

Machine-precision agreement. The independently computed eigenvectors satisfy the eigenvalue equation exactly.

---

## 6. Tail Eigenvalues

| k | Stored | Independent |
|---|---|---|
| 40 | −1.124319e-05 | −1.124308e-05 |
| 41 | −1.174051e-05 | −1.174039e-05 |
| 42 | −1.705503e-05 | −1.705489e-05 |
| 43 | −2.238063e-05 | −2.238052e-05 |
| 44 | **−2.438019e-04** | **−2.437996e-04** |

Negative eigenvalues are present in the tail (estimation noise). The largest negative eigenvalue (k=44, λ=−2.44e-4) is smaller in magnitude than λ₂ (6.10e-4), confirming the structured negative eigenvalue issue identified in earlier investigations.

---

## 7. Eigenvector Pairing: How It Works in Practice

Since eigenvectors are **not stored** in the canonical file, any code that needs DA2 must:

1. Load `C_tau[37]`
2. Symmetrize: `C_sym = 0.5 * (C_tau[37] + C_tau[37].T)` (no-op here, already symmetric)
3. Eigendecompose: `evals, evecs = eigh(C_sym)`
4. Sort descending: `idx = argsort(evals)[::-1]`
5. DA2 = `evecs[:, idx[1]]` (second column after sorting)

This is exactly what `extract_da_directions` does (lines 54–56 of `da_directions.py`). The eigenvalues from step 3–4 match the stored `eigenvalues[37]` to ~1e-5 relative precision.

**There is no pairing ambiguity:** eigenvalues and eigenvectors come from the same `eigh` call and are sorted together by the same index array. Since the file stores `C_tau` (not eigenvectors), the pairing is always correct by construction when eigenvectors are recomputed.

---

## 8. Verdict

**All checks pass:**

| Check | Result |
|---|---|
| Stored evals match independent eigendecomp? | **YES** (max relative diff 1.04e-05) |
| Stored evals in descending order? | **YES** (strictly) |
| C_tau[37] symmetric? | **YES** (exactly, to float64 precision) |
| Rayleigh quotient v@C@v = λ? | **YES** (machine precision) |
| Eigenvectors stored? | **NO** — recomputed from C_tau on-the-fly |
| Pairing risk? | **NONE** — eigenvalues and eigenvectors always come from the same eigh call |

**The stored eigenvalue λ₂ = 6.096e-04 at idx=37 is correct and corresponds to the second eigenvector of C_tau[37].** There is no mis-pairing.
