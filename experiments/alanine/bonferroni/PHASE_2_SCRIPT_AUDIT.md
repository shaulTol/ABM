# Phase 2: Script Audit — `run_alanine_null.py` Pipeline Checkpoints

**Date:** 2026-05-02
**File under audit:** `experiments/mc_dip_validation/run_alanine_null.py` (127 lines)
**Supporting files:** `src/simulation/generic_langevin.py`, `src/metrics/da_directions.py`

---

## Checkpoint 1: Null sample distribution

**What is x_null drawn from?**

**N(0, Σ̂_data)** — the full empirical covariance of the alanine data, NOT N(0, σ²_f · I).

Lines 56–68 of `run_alanine_null.py`:

```
56:    # --- Load Σ_hat from pre-saved C(0) ---
57:    # C(0) is the KDE-weighted data covariance (45×45), saved in the h/2 pipeline.
58:    # This avoids needing mdshare on the cluster.
59:    print(f"Loading C(0) from {args.ctau_path}...")
60:    ctau_data = np.load(args.ctau_path)
61:    cov_hat = np.array(ctau_data['C0'], dtype=np.float64)  # (45, 45)
62:    d = int(ctau_data['d'])
63:    print(f"  Σ_hat shape: {cov_hat.shape}, d={d}, cond: {np.linalg.cond(cov_hat):.0f}")
64:
65:    # Symmetrize and Cholesky for sampling
66:    cov_hat = 0.5 * (cov_hat + cov_hat.T)
67:    L = np.linalg.cholesky(cov_hat)  # (45, 45)
68:    mu_hat = np.zeros(d)  # Center at origin (C(0) is already centered)
```

Lines 79–83:

```
79:        # 1. Sample null data: x_null ~ N(mu_hat, Σ_hat)
80:        rng = np.random.RandomState(seed)
81:        z = rng.randn(args.N, d)  # (N, 45)
82:        x_null_np = mu_hat + z @ L.T  # (N, 45)
83:        x_null = torch.from_numpy(x_null_np).float()
```

**Covariance source:** `C0` from `alanine_C_tau_full_h_half_10k.npz` (the canonical observed-data file). This is the KDE-weighted lag-0 autocovariance of the alanine MD frames, i.e. the empirical covariance under the KDE density. It is the full 45×45 matrix, not σ²_f · I.

**Implication:** The null is N(0, Σ̂_data) — a Gaussian with the *same covariance structure* as the data. This is a stronger null than N(0, σ²_f · I). An isotropic null would produce eigenvalues distributed as Marchenko-Pastur; this structured null preserves the data's eigenspectrum shape, meaning the null's λ₂ can be large simply because the data's C(0) has a large second eigenvalue.

---

## Checkpoint 2: Score function on null

**Does it refit KDE on x_null, or reuse the KDE fit on alanine data?**

**Refits KDE on x_null.** Each null replicate builds a fresh KDE from the null sample.

Lines 85–86 of `run_alanine_null.py`:

```
85:        # 2. Build KDE score on null data
86:        kde = KDEScore(x_null, bandwidth=args.h, device=device, batch_size=500)
```

The `KDEScore` constructor receives `x_null` (the Gaussian sample), not the original alanine data. The score function ∇log f̂_null is the KDE score of the null sample.

---

## Checkpoint 3: Canonical diffusion drift

**Is the drift 0.5 · σ²_f · score(X_t)? Whose σ²_f — null's or data's?**

**Null's σ²_f**, computed empirically from x_null.

Lines 88–94 of `run_alanine_null.py`:

```
88:        # 3. Build simulator
89:        sim = GenericLangevinSimulator(
90:            score_fn=kde.score,
91:            samples=x_null,
92:            dt=args.dt,
93:            device=device,
94:        )
```

Neither `var_f` nor `cov_f` nor `mu_f` is passed explicitly. Therefore `GenericLangevinSimulator.__init__` falls through to the empirical estimation branch.

Lines 60–78 of `generic_langevin.py`:

```
60:        # Stationary moments — empirical if not provided
61:        if mu_f is not None:
62:            self.mu_f = mu_f.to(self.device)
63:        else:
64:            self.mu_f = self.samples.mean(dim=0)
65:
66:        if var_f is not None:
67:            ...
68:            ...
69:            ...
70:        elif cov_f is not None:
71:            ...
72:            ...
73:        else:
74:            centered = self.samples - self.mu_f.unsqueeze(0)
75:            self.cov_f = (centered.t() @ centered) / (self.samples.shape[0] - 1)
76:            self.var_f = torch.trace(self.cov_f).item() / self.d
77:
78:        self.sigma_f = np.sqrt(self.var_f)
```

So `var_f = tr(Cov_null) / d`, where `Cov_null` is the empirical covariance of the N=2000 null samples (not the population Σ̂_data, but the sample covariance of this particular draw).

The drift is then applied in `step()` at lines 92–106 of `generic_langevin.py`:

```
101:        grad = self.score_fn(x)  # (..., d)
102:        drift = 0.5 * self.var_f * self.dt * grad
```

**Summary:** drift = 0.5 · var_f_null · dt · ∇log f̂_null(X_t), noise = σ_f_null · √dt · Z.

---

## Checkpoint 4: Trajectory init

**What is X₀? Drawn from the same null x_null, or fresh draw?**

**Resampled (with replacement) from the same x_null.**

Line 91 of `run_alanine_null.py` passes `samples=x_null` to the simulator. Then `extract_da_directions` calls `sim.sample_initial(n_trajectories)` at line 118 of `da_directions.py`:

```
118:    x0 = sim.sample_initial(n_trajectories)  # (N, d)
```

And `sample_initial` at lines 84–90 of `generic_langevin.py`:

```
84:    def sample_initial(self, n: int) -> torch.Tensor:
85:        """
86:        Sample initial conditions by resampling (with replacement) from the
87:        stored samples. This is the nonparametric analogue of sampling from f.
88:        """
89:        idx = torch.randint(0, self.samples.shape[0], (n,), device=self.device)
90:        return self.samples[idx].clone()
```

Since `n_trajectories = args.N = 2000` (line 100 of `run_alanine_null.py`) and `self.samples` has shape (2000, 45), this is a bootstrap resample of the same 2000 null points. Every X₀ is one of the original x_null points (with replacement).

---

## Checkpoint 5: Ĉ(τ) computation — is μ_f subtracted? Whose μ?

**Yes, μ_f is subtracted. It is the null sample mean (mean of x_null).**

Lines 126–128 of `da_directions.py`:

```
126:    # Center
127:    x0_centered = x0 - mu_f.unsqueeze(0)
128:    x_tau_centered = x_tau - mu_f.unsqueeze(0)
```

The `mu_f` here comes from `_make_simulator` (line 111), which returns `simulator.mu_f` (line 34 of `da_directions.py`). As established in checkpoint 3, `simulator.mu_f = self.samples.mean(dim=0)` = mean of x_null (line 64 of `generic_langevin.py`).

Since x_null ~ N(0, Σ̂), the null mean is ≈ 0 (with N=2000 sampling noise). This is the null's own mean, not the alanine data's mean.

---

## Checkpoint 6: Eigendecomp — Ĉ or symmetrized 0.5·(Ĉ + Ĉᵀ)?

**Symmetrized: 0.5·(Ĉ(τ) + Ĉ(τ)ᵀ).**

Lines 50–56 of `da_directions.py`:

```
50: def _eigendecompose_C_tau(x_tau_centered, x0_centered, n):
51:     """Compute symmetrized C(τ), eigendecompose, return (evals_desc, evecs_desc, C_sym)."""
52:     C_tau = torch.matmul(x_tau_centered.t(), x0_centered) / n
53:     C_tau_sym = 0.5 * (C_tau + C_tau.t())
54:     evals, evecs = torch.linalg.eigh(C_tau_sym)  # ascending
55:     idx = torch.argsort(evals, descending=True)
56:     return evals[idx], evecs[:, idx], C_tau_sym
```

Called from `extract_da_directions` at lines 130–132:

```
130:    eigenvalues, eigenvectors, C_tau_sym = _eigendecompose_C_tau(
131:        x_tau_centered, x0_centered, n_trajectories
132:    )
```

`torch.linalg.eigh` is used (Hermitian eigendecomposition), which is appropriate for the symmetrized matrix.

---

## Summary

| Checkpoint | Answer | Key line(s) |
|---|---|---|
| 1. Null distribution | N(0, Σ̂_data) via Cholesky of C(0) from canonical file | `run_alanine_null.py:61,66-67,82` |
| 2. Score function | Refits KDE on x_null (fresh KDE per replicate) | `run_alanine_null.py:86` |
| 3. Diffusion coeff | Null's σ²_f = tr(Cov_null)/d (empirical from x_null) | `generic_langevin.py:74-76,102` |
| 4. Trajectory init | Bootstrap resample from x_null | `generic_langevin.py:89-90` |
| 5. μ_f subtraction | Yes, null sample mean (≈ 0) | `da_directions.py:127-128`, `generic_langevin.py:64` |
| 6. Eigendecomp | Symmetrized 0.5·(Ĉ + Ĉᵀ), then `eigh` | `da_directions.py:53-54` |

**Nothing is missing.** All six checkpoints have clear implementations in the code.

**Critical observation from Checkpoint 1:** The null uses N(0, Σ̂_data) — the full data covariance — not N(0, σ²_f · I). This means the null inherits the data's eigenspectrum shape. If the data's C(0) has a large second eigenvalue (which alanine does — it has anisotropic variance), then the null's C(τ) will also have a large λ₂ at short lags, even without any metastability. This may explain why DA2 fails: the null's λ₂ is inflated by the data's anisotropy, not by slow mixing.
