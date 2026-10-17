#!/usr/bin/env python3
"""P3b follow-up: DA on a POLAR readout of the two-shell density.

The main P3b run computes DA only on the ambient Cartesian coordinates
x=(x1,x2), where the density is rotationally symmetric and DA1 is therefore an
arbitrary diameter. This script asks the complementary question the ambient run
cannot: given a readout that CAN express the radial coordinate, does DA find it?

Readout.  phi(x) = (r, cos(theta), sin(theta)) with r=||x||.  theta itself is
circular, so it is NOT usable as a linear coordinate in an autocovariance --
(cos, sin) is the standard well-defined embedding of the angle, and it spans the
same information. DA1 is then reported by how its weight splits between the
radial component and the 2-D angular block:

    radial_loading = w_r^2 / (w_r^2 + w_cos^2 + w_sin^2)

which is 1 when DA1 is exactly the radial coordinate and ~1/3 when DA1 is
indifferent between the three features.

Features are standardized before the autocovariance so the split reflects
dynamics, not the arbitrary relative scale of r versus a unit-modulus angle.
The ambient readout is recomputed here the same way as a control, so the two
numbers are produced by one code path.

Usage:  python3 p3b_polar_da.py [--smoke] [--ratio 5.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from p3b_continuous_rings import two_shell, make_simulator

OUT = Path(__file__).resolve().parent / "results/p3b_rings/p3b_polar_da.json"


def c_tau_eig(feat, lag):
    """Trajectory-averaged symmetrized C(tau) of a standardized feature block.

    feat: (N, T+1, k). Centering is per-starting-time (the process is not
    stationary at t=0), matching the trajectory-averaged convention used
    elsewhere in the project.
    """
    x = feat - feat.mean(axis=0, keepdims=True)      # per-starting-time centering
    sd = x.reshape(-1, x.shape[-1]).std(axis=0) + 1e-12
    x = x / sd                                        # standardize per feature
    a, b = x[:, :-lag, :], x[:, lag:, :]
    n = a.shape[0] * a.shape[1]
    C = np.einsum('nti,ntj->ij', a, b) / n
    C = 0.5 * (C + C.T)
    ev, evec = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1].copy()                   # negative stride breaks torch
    return ev[o], evec[:, o]


def polar_features(traj):
    x = traj[..., 0]
    y = traj[..., 1]
    r = np.sqrt(x ** 2 + y ** 2)
    th = np.arctan2(y, x)
    return np.stack([r, np.cos(th), np.sin(th)], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--ratio', type=float, default=5.0)
    ap.add_argument('--m1', type=float, default=2.25)
    ap.add_argument('--m2', type=float, default=9.0)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--N', type=int, default=None)
    ap.add_argument('--T', type=int, default=None, help='steps (physical time = T*dt)')
    ap.add_argument('--lags', type=str, default=None, help='comma-separated lag steps')
    a = ap.parse_args()

    cfg = dict(dt=0.002, N=a.N or (300 if a.smoke else 1000),
               T=a.T or (1500 if a.smoke else 3000), n_pool=200000)
    if a.lags:
        lags = [int(x) for x in a.lags.split(',')]
    else:
        lags = [50, 100, 250] if a.smoke else [50, 100, 250, 500, 1000]
    lags = [l for l in lags if l < cfg['T']]

    dens = two_shell(a.m1, a.m2, a.ratio)
    print(f"two_shell(m1={a.m1}, m2={a.m2}, ratio={a.ratio})  cfg={cfg}")

    rows = []
    for s in range(a.seeds):
        rng = np.random.default_rng(100000 + s)
        x0, _, _ = dens.sample(cfg['N'], rng)
        sim, mom = make_simulator(dens, cfg['dt'], cfg['n_pool'], s)
        torch.manual_seed(s)
        traj = sim.simulate(torch.tensor(x0, dtype=torch.float32), cfg['T'])
        traj = traj.cpu().numpy() if hasattr(traj, 'cpu') else np.asarray(traj)

        pol = polar_features(traj)
        for lag in lags:
            ev_p, vec_p = c_tau_eig(pol, lag)
            w = vec_p[:, 0] ** 2
            rad_load = float(w[0] / w.sum())
            ev_a, vec_a = c_tau_eig(traj, lag)
            rows.append(dict(seed=s, lag=lag, lag_phys=lag * cfg['dt'],
                             polar_eigs=ev_p.tolist(),
                             polar_da1=vec_p[:, 0].tolist(),
                             polar_radial_loading=rad_load,
                             polar_gap=float(ev_p[0] / max(ev_p[1], 1e-12)),
                             ambient_eigs=ev_a.tolist(),
                             ambient_da1=vec_a[:, 0].tolist(),
                             ambient_gap=float(ev_a[0] / max(ev_a[1], 1e-12))))
            print(f"  seed {s} lag {lag:>5} (t={lag*cfg['dt']:.2f}): "
                  f"polar DA1 radial loading = {rad_load:.4f}  "
                  f"gap={rows[-1]['polar_gap']:.2f} | "
                  f"ambient gap={rows[-1]['ambient_gap']:.3f}")

    print("\nsummary (median over seeds):")
    for lag in lags:
        v = [r['polar_radial_loading'] for r in rows if r['lag'] == lag]
        ga = [r['ambient_gap'] for r in rows if r['lag'] == lag]
        gp = [r['polar_gap'] for r in rows if r['lag'] == lag]
        print(f"  lag {lag:>5}: radial loading {np.median(v):.4f} "
              f"[{min(v):.4f}, {max(v):.4f}]   polar gap {np.median(gp):.2f}   "
              f"ambient gap {np.median(ga):.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        config=cfg, ratio=a.ratio, m1=a.m1, m2=a.m2, lags=lags,
        smoke=bool(a.smoke),
        readout="phi(x)=(r, cos th, sin th), standardized; per-starting-time centering",
        rows=rows), indent=1))
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
