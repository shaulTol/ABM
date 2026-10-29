#!/usr/bin/env python3
"""
Step 1 of the pipeline (simulate the canonical diffusion, save C_sym(τ)).

It needs a score function, so it is called from Python rather than the shell:

    from src.pipeline import save_lags
    save_lags(score_fn, X0, sigma_f_sq, dt, T, lags, out_path, save_trajectories=True)

Then: python -m cli.compute_ssa <out_path>   and   python -m cli.compute_da <out_path>

Implementation: src/pipeline/save_lags.py
"""

if __name__ == '__main__':
    print(__doc__)
