#!/usr/bin/env python3
"""
P9 — Absolute SSA sensitivity to score error (pure analysis of the P1 results).

P1 established that the SSA->MI RANKING is stable under score perturbation. That is a
different and weaker claim than "the reported SSA VALUE is stable". This script
quantifies the second claim from the stored P1 artifacts. No simulation is rerun.

Inputs (all local, produced by P1):
  results/score_robustness_full_fc.json          Shat per (level, seed, gmm) + Spearman
  results/score_robustness_full_fc_shard_*.json  per-cell window-limited flags + tau*

Pairing: every perturbed cell is compared against the exact-score run for the SAME
GMM and the SAME seed (never a pooled mean). In P1 the exact-score baseline, the
additive alpha=0 cell and the smoothing frac=0 cell are the identical run
(repro_max_absdiff = 0), which this script re-verifies as a self-check.

Per cell we report signed change, relative change and ratio; across GMMs we report
median / IQR / p90 of ABSOLUTE RELATIVE ERROR, stratified by baseline SSA into
low/middle/high thirds (an aggregate hides the strongly separated cases).

The two perturbation families are kept separate throughout: a frozen additive score
error and a covariance-smoothing bias are different mechanisms and behave differently.

WINDOW-LIMITED CELLS: 202/1800 P1 cells never reached the 1/N floor within the
horizon, so their Shat is a finite-horizon LOWER BOUND. For those cells both the
numerator and the denominator of the relative error are lower bounds, so their
relative error is NOT a resolved infinite-horizon error. Headline statistics are
therefore computed on RESOLVED pairs (neither cell window-limited); window-limited
cells are reported separately and flagged, never silently pooled or dropped.

Usage:
  python analyze_absolute_sensitivity.py            # full analysis + figure
  python analyze_absolute_sensitivity.py --no_fig   # tables only
"""
import sys, os, json, glob, csv, argparse
import numpy as np

CODEDIR = os.path.dirname(os.path.abspath(__file__))
REPODIR = os.path.abspath(os.path.join(CODEDIR, '..', '..', '..'))
RESULTS = os.path.join(CODEDIR, 'results')
FIGDIR = os.path.join(REPODIR, 'experiments', 'robustness', 'figures')

# dataviz reference palette (validated all-pairs, light mode):
C_ADD = '#2a78d6'      # categorical slot 1 (blue)   — additive frozen field
C_SMO = '#eb6834'      # categorical slot 2 (orange) — covariance smoothing
C_GRID = '#d9d8d4'
C_TEXT = '#0b0b0b'
C_MUTED = '#52514e'


# ---------------------------------------------------------------- load

def load_p1():
    """Load merged Shat + per-cell window-limited flags from the shards."""
    full_p = os.path.join(RESULTS, 'score_robustness_full_fc.json')
    if not os.path.exists(full_p):
        raise SystemExit(f"MISSING {full_p} — P1 full results are required.")
    full = json.load(open(full_p))
    cfg = full['config']
    n_gmm = cfg['n_gmm']; seeds = cfg['seeds']; n_seeds = len(seeds)

    shards = sorted(glob.glob(os.path.join(RESULTS, 'score_robustness_full_fc_shard_*.json')))
    if not shards:
        raise SystemExit("MISSING shard files — per-cell window-limited flags "
                         "live only in the shards; cannot mark window-limited cells.")
    wl = {}     # (family, level_str, gid, si) -> bool ; family 'baseline' has level ''
    keep = {}   # same -> stopping lag
    covered = set()
    for p in shards:
        s = json.load(open(p))
        for gid_s, diag in s['diagnostics'].items():
            gid = int(gid_s); covered.add(gid)
            for si in range(n_seeds):
                wl[('baseline', '', gid, si)] = bool(diag['baseline_wl'][si])
                keep[('baseline', '', gid, si)] = int(diag['baseline_keep'][si])
                for lv, arr in diag['additive_wl'].items():
                    wl[('additive', lv, gid, si)] = bool(arr[si])
                for lv, arr in diag['additive_keep'].items():
                    keep[('additive', lv, gid, si)] = int(arr[si])
                for lv, arr in diag['smoothing_wl'].items():
                    wl[('smoothing', lv, gid, si)] = bool(arr[si])
                for lv, arr in diag['smoothing_keep'].items():
                    keep[('smoothing', lv, gid, si)] = int(arr[si])
    missing = sorted(set(range(n_gmm)) - covered)
    if missing:
        raise SystemExit(f"shards do not cover gids {missing}")
    return full, cfg, wl, keep


def build_cells(full, cfg, wl, keep):
    """Flatten to per-cell records with strict (gid, seed) pairing."""
    n_gmm = cfg['n_gmm']; n_seeds = len(cfg['seeds'])
    base = np.array(full['baseline']['shat'], dtype=float)          # (n_seeds, n_gmm)
    mi = np.array(full['mi'], dtype=float)
    metas = full['metas']
    cells = []
    for fam, key, levels in [('additive', 'additive', cfg['additive_alphas']),
                             ('smoothing', 'smoothing', cfg['smoothing_fracs'])]:
        for lv in levels:
            arr = np.array(full[key]['shat'][str(lv)], dtype=float)  # (n_seeds, n_gmm)
            for si in range(n_seeds):
                for gid in range(n_gmm):
                    s_ex = base[si, gid]; s_pe = arr[si, gid]
                    wl_ex = wl[('baseline', '', gid, si)]
                    wl_pe = wl[(fam, str(lv), gid, si)]
                    cells.append(dict(
                        family=fam, level=float(lv), gid=gid, seed=cfg['seeds'][si],
                        s_exact=s_ex, s_pert=s_pe,
                        signed=s_pe - s_ex,
                        rel=(s_pe - s_ex) / s_ex if s_ex else np.nan,
                        ratio=s_pe / s_ex if s_ex else np.nan,
                        wl_exact=wl_ex, wl_pert=wl_pe, wl_any=bool(wl_ex or wl_pe),
                        tau_star_exact=keep[('baseline', '', gid, si)],
                        tau_star_pert=keep[(fam, str(lv), gid, si)],
                        mi=mi[gid], separation=metas[gid]['separation']))
    return cells, base, mi


# ---------------------------------------------------------------- stats

def q_stats(vals):
    """median, IQR (q25,q75,width), p90 of |relative error|."""
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return dict(n=0, median=np.nan, q25=np.nan, q75=np.nan, iqr=np.nan, p90=np.nan)
    return dict(n=int(v.size), median=float(np.median(v)),
                q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
                iqr=float(np.percentile(v, 75) - np.percentile(v, 25)),
                p90=float(np.percentile(v, 90)))


def strata_from_baseline(base):
    """Assign each GMM to a low/mid/high tercile by its ACROSS-SEED MEAN baseline SSA.

    Stratifying per GMM (not per cell) keeps a GMM in one stratum for every level and
    seed, so strata are comparable across the perturbation grid.
    """
    per_gmm = base.mean(axis=0)                      # (n_gmm,)
    order = np.argsort(per_gmm)
    n = len(per_gmm); k = n // 3
    lab = {}
    for rank, gid in enumerate(order):
        lab[int(gid)] = 'low' if rank < k else ('mid' if rank < 2 * k else 'high')
    return lab, per_gmm


def summarize(cells, strata):
    """Aggregate |rel| by (family, level) overall and by baseline-SSA stratum."""
    out = {}
    fams = sorted({c['family'] for c in cells})
    for fam in fams:
        for lv in sorted({c['level'] for c in cells if c['family'] == fam}):
            sub = [c for c in cells if c['family'] == fam and c['level'] == lv]
            res = [c for c in sub if not c['wl_any']]
            wlc = [c for c in sub if c['wl_any']]
            rec = dict(
                family=fam, level=lv, n_cells=len(sub),
                n_window_limited=len(wlc),
                all_cells=q_stats([abs(c['rel']) for c in sub]),
                resolved=q_stats([abs(c['rel']) for c in res]),
                window_limited=q_stats([abs(c['rel']) for c in wlc]),
                signed_median_resolved=float(np.median([c['signed'] for c in res]))
                if res else np.nan,
                signed_rel_median_resolved=float(np.median([c['rel'] for c in res]))
                if res else np.nan,
                frac_inflated_resolved=float(np.mean([c['rel'] > 0 for c in res]))
                if res else np.nan,
                ratio_median_resolved=float(np.median([c['ratio'] for c in res]))
                if res else np.nan,
                frac_within_10pct_resolved=(
                    float(np.mean([abs(c['rel']) <= 0.10 for c in res])) if res else np.nan),
                strata={})
            for st in ['low', 'mid', 'high']:
                s_all = [c for c in sub if strata[c['gid']] == st]
                s_res = [c for c in s_all if not c['wl_any']]
                rec['strata'][st] = dict(
                    n_cells=len(s_all), n_window_limited=len(s_all) - len(s_res),
                    resolved=q_stats([abs(c['rel']) for c in s_res]),
                    all_cells=q_stats([abs(c['rel']) for c in s_all]),
                    signed_rel_median=float(np.median([c['rel'] for c in s_res]))
                    if s_res else np.nan,
                    frac_inflated=float(np.mean([c['rel'] > 0 for c in s_res]))
                    if s_res else np.nan)
            out[(fam, lv)] = rec
    return out


# ---------------------------------------------------------------- checks

def run_checks(full, cfg, cells, summary, strata):
    print("\n=== CHECKS ===")
    ok = True

    n_expected = cfg['n_gmm'] * len(cfg['seeds']) * (
        len(cfg['additive_alphas']) + len(cfg['smoothing_fracs']))
    print(f"[1] cell count: {len(cells)} (expected {n_expected})", end='  ')
    c1 = len(cells) == n_expected; ok &= c1; print("PASS" if c1 else "FAIL")

    # zero-perturbation must reproduce the exact baseline EXACTLY (pairing sanity)
    z = [c for c in cells if c['level'] == 0.0]
    zmax = max(abs(c['signed']) for c in z)
    print(f"[2] zero-perturbation |S_pert - S_exact| max = {zmax:.3e} over {len(z)} "
          f"cells", end='  ')
    c2 = zmax == 0.0; ok &= c2; print("PASS" if c2 else "FAIL")

    # window-limited totals must match the P1 merged bookkeeping
    wl_pert = sum(1 for c in cells if c['wl_pert'])
    # P1 counted every cell incl. the standalone baseline run (50*3 extra)
    n_wl_p1 = full['checks']['n_window_limited']
    wl_base = sum(1 for c in cells if c['wl_exact'] and c['family'] == 'additive'
                  and c['level'] == 0.0)
    total = wl_pert + wl_base
    print(f"[3] window-limited: perturbed-cell {wl_pert} + baseline {wl_base} = {total} "
          f"vs P1 n_window_limited {n_wl_p1}", end='  ')
    c3 = total == n_wl_p1; ok &= c3; print("PASS" if c3 else "FAIL")

    # pairing: each perturbed cell's s_exact must equal baseline[seed, gid]
    base = np.array(full['baseline']['shat'], dtype=float)
    seed_idx = {s: i for i, s in enumerate(cfg['seeds'])}
    bad = sum(1 for c in cells
              if base[seed_idx[c['seed']], c['gid']] != c['s_exact'])
    print(f"[4] (gid, seed) pairing violations: {bad}", end='  ')
    c4 = bad == 0; ok &= c4; print("PASS" if c4 else "FAIL")

    # strata partition the 50 GMMs
    counts = {st: sum(1 for g, s in strata.items() if s == st)
              for st in ['low', 'mid', 'high']}
    print(f"[5] strata sizes {counts} sum={sum(counts.values())} "
          f"(n_gmm={cfg['n_gmm']})", end='  ')
    c5 = sum(counts.values()) == cfg['n_gmm']; ok &= c5; print("PASS" if c5 else "FAIL")

    # resolved + window-limited must partition each level's cells
    part = all(summary[k]['resolved']['n'] + summary[k]['window_limited']['n']
               == summary[k]['n_cells'] for k in summary)
    print(f"[6] resolved + window-limited == n_cells for every level", end='  ')
    ok &= part; print("PASS" if part else "FAIL")

    print(f"=== {'ALL CHECKS PASSED' if ok else 'CHECK FAILURE'} ===\n")
    return ok


# ---------------------------------------------------------------- figure

def make_figure(cells, summary, cfg, strata, out_base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.edgecolor': C_MUTED, 'axes.linewidth': 0.7,
        'text.color': C_TEXT, 'axes.labelcolor': C_TEXT,
        'xtick.color': C_MUTED, 'ytick.color': C_MUTED,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    fam_color = {'additive': C_ADD, 'smoothing': C_SMO}
    fam_label = {'additive': 'Additive frozen score error',
                 'smoothing': 'Covariance smoothing bias'}

    # (a) paired plot vs identity, strongest common level (1.0) for both families
    ax = axes[0, 0]
    lim_lo, lim_hi = 0.6, 60
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=C_MUTED, lw=0.9, ls='--',
            zorder=1, label='identity')
    for fam in ['additive', 'smoothing']:
        sub = [c for c in cells if c['family'] == fam and c['level'] == 1.0]
        res = [c for c in sub if not c['wl_any']]
        wlc = [c for c in sub if c['wl_any']]
        ax.scatter([c['s_exact'] for c in res], [c['s_pert'] for c in res],
                   s=13, color=fam_color[fam], alpha=0.75, linewidths=0,
                   zorder=3, label=f"{fam_label[fam]} (resolved)")
        ax.scatter([c['s_exact'] for c in wlc], [c['s_pert'] for c in wlc],
                   s=17, facecolors='none', edgecolors=fam_color[fam], linewidths=0.9,
                   zorder=4, label=f"{fam_label[fam]} (window-limited)")
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(r'exact-score $\hat S$'); ax.set_ylabel(r'perturbed $\hat S$')
    ax.set_title('(a) Paired values at level 1.0', loc='left')
    ax.legend(frameon=False, loc='upper left', fontsize=6.2, handletextpad=0.4)
    ax.grid(True, color=C_GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    # (b) relative error vs exact SSA
    ax = axes[0, 1]
    ax.axhline(0, color=C_MUTED, lw=0.9, ls='--', zorder=1)
    for fam in ['additive', 'smoothing']:
        sub = [c for c in cells if c['family'] == fam and c['level'] == 1.0]
        res = [c for c in sub if not c['wl_any']]
        wlc = [c for c in sub if c['wl_any']]
        ax.scatter([c['s_exact'] for c in res], [100 * c['rel'] for c in res],
                   s=13, color=fam_color[fam], alpha=0.75, linewidths=0, zorder=3)
        ax.scatter([c['s_exact'] for c in wlc], [100 * c['rel'] for c in wlc],
                   s=17, facecolors='none', edgecolors=fam_color[fam],
                   linewidths=0.9, zorder=4)
    ax.set_xscale('log')
    ax.set_xlabel(r'exact-score $\hat S$')
    ax.set_ylabel('relative change (%)')
    ax.set_title('(b) Relative change vs baseline SSA (level 1.0)', loc='left')
    ax.grid(True, color=C_GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)

    # (c) median |relative error| vs perturbation level, IQR band — RESOLVED pairs
    ax = axes[1, 0]
    for fam in ['additive', 'smoothing']:
        lvs = sorted({c['level'] for c in cells if c['family'] == fam})
        med = [100 * summary[(fam, lv)]['resolved']['median'] for lv in lvs]
        q25 = [100 * summary[(fam, lv)]['resolved']['q25'] for lv in lvs]
        q75 = [100 * summary[(fam, lv)]['resolved']['q75'] for lv in lvs]
        ax.plot(lvs, med, '-o', color=fam_color[fam], lw=2, ms=4.5,
                label=fam_label[fam], zorder=3)
        ax.fill_between(lvs, q25, q75, color=fam_color[fam], alpha=0.16,
                        linewidth=0, zorder=2)
    ax.axhline(10, color=C_MUTED, lw=0.8, ls=':', zorder=1)
    ax.text(0.02, 10, ' 10%', color=C_MUTED, fontsize=6.5, va='bottom')
    ax.set_xlabel('perturbation level  (alpha x s0  /  frac x var)')
    ax.set_ylabel(r'median $|\Delta \hat S / \hat S|$ (%)')
    ax.set_title('(c) Absolute error vs level (resolved pairs, IQR band)', loc='left')
    ax.legend(frameon=False, loc='upper left', fontsize=6.8)
    ax.grid(True, color=C_GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)

    # (d) stratified by baseline SSA tercile at level 1.0
    ax = axes[1, 1]
    sts = ['low', 'mid', 'high']
    x = np.arange(len(sts)); w = 0.36
    for i, fam in enumerate(['additive', 'smoothing']):
        vals, hatch_flag = [], []
        for st in sts:
            s = summary[(fam, 1.0)]['strata'][st]
            v = s['resolved']['median']
            if not np.isfinite(v):        # every cell in this stratum is window-limited
                v = s['all_cells']['median']; hatch_flag.append(True)
            else:
                hatch_flag.append(False)
            vals.append(100 * v)
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=fam_color[fam],
                      label=fam_label[fam], zorder=3)
        for b, v, hf in zip(bars, vals, hatch_flag):
            if hf:
                b.set_hatch('///'); b.set_edgecolor('white'); b.set_linewidth(0)
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.0f}', ha='center',
                    va='bottom', fontsize=6.5, color=C_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}\nbaseline SSA' for s in sts])
    ax.set_ylabel(r'median $|\Delta \hat S / \hat S|$ (%)')
    ax.set_title('(d) Stratified by baseline SSA (level 1.0)', loc='left')
    ax.legend(frameon=False, loc='upper left', fontsize=6.8)
    ax.grid(True, axis='y', color=C_GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)

    fig.suptitle('P1/P9: absolute finite-horizon SSA under controlled score error '
                 f"(d={cfg['d']}, N={cfg['N']}, T={cfg['T_phys']:.0f}s, "
                 f"{cfg['n_gmm']} GMMs x {len(cfg['seeds'])} seeds; "
                 'hollow / hatched = window-limited lower bounds)',
                 fontsize=8.2, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(f'{out_base}.{ext}', dpi=200 if ext == 'png' else None,
                    bbox_inches='tight')
    plt.close(fig)
    return [f'{out_base}.png', f'{out_base}.pdf']


# ---------------------------------------------------------------- report

def fmt_pct(x):
    return 'n/a' if not np.isfinite(x) else f'{100 * x:.1f}%'


# ------------------------------------------------- additive-only presentation
# The covariance-smoothing family WAS measured and is retained in full in the
# machine-readable artifacts (JSON/CSV). It is deliberately excluded from the
# rebuttal table and figure below; this is a presentation restriction, not a
# deletion of data.

C_L1 = '#2a78d6'   # categorical slot 1 (blue)  — alpha = 1.0
C_L2 = '#1baf7a'   # categorical slot 3 (aqua)  — alpha = 2.0 / ranking series
# validated all-pairs light mode: CVD dE 23.1, normal-vision dE 24.0, all checks pass


def additive_headline(summary, full):
    """Derive the additive-only conservative sentence AND test whether the
    'absolute degrades before ranking' claim survives on additive evidence alone.

    Test: is there a level where the median value error exceeds 10% while the
    ranking still retains >=90% of its exact-score Spearman? Also compares the two
    degradations on a common scale (value error % vs ranking loss %).
    """
    lvls = [lv for lv in sorted({k[1] for k in summary if k[0] == 'additive'}) if lv > 0]
    sp_ex = full['baseline']['spearman']['mean']

    def med(lv):
        return summary[('additive', lv)]['resolved']['median']

    def p90(lv):
        return summary[('additive', lv)]['resolved']['p90']

    def sp(lv):
        return full['additive']['spearman'][str(lv)]['mean']

    def rank_loss(lv):
        return 1.0 - sp(lv) / sp_ex

    ok = [lv for lv in lvls if np.isfinite(med(lv)) and med(lv) <= 0.10]
    brk = next((lv for lv in lvls if np.isfinite(med(lv)) and med(lv) > 0.10), None)
    divergence = [lv for lv in lvls
                  if np.isfinite(med(lv)) and med(lv) > 0.10 and rank_loss(lv) <= 0.10]
    strongest = lvls[-1]
    st_vals = {st: summary[('additive', strongest)]['strata'][st]['resolved']['median']
               for st in ['low', 'mid', 'high']}
    st_vals = {k: v for k, v in st_vals.items() if np.isfinite(v)}
    worst_st = max(st_vals, key=st_vals.get) if st_vals else None

    sentence = (
        f"On resolved pairs (neither cell window-limited) the absolute finite-horizon "
        f"SSA stays within a median 10% of its exact-score value for additive score "
        f"error up to alpha={max(ok) if ok else 0} x s0 "
        f"(median {fmt_pct(med(max(ok))) if ok else 'n/a'}, "
        f"p90 {fmt_pct(p90(max(ok))) if ok else 'n/a'}), and degrades beyond that to a "
        f"median {fmt_pct(med(brk)) if brk else 'n/a'} (p90 "
        f"{fmt_pct(p90(brk)) if brk else 'n/a'}) at alpha={brk} x s0, with the "
        f"{worst_st} baseline-SSA third worst affected "
        f"({fmt_pct(st_vals[worst_st]) if worst_st else 'n/a'}); over this range the "
        f"value error and the ranking loss move together rather than the value "
        f"degrading first (alpha=1.0: {fmt_pct(med(1.0))} value vs "
        f"{fmt_pct(rank_loss(1.0))} ranking loss; alpha={strongest}: "
        f"{fmt_pct(med(strongest))} value vs {fmt_pct(rank_loss(strongest))} ranking "
        f"loss), so on additive evidence alone we claim only that both the reported "
        f"magnitude and the ordering remain usable up to roughly alpha=1 x s0 and "
        f"should be treated as approximate beyond it.")

    verdict = dict(
        divergence_levels=divergence,
        headline_survives=bool(divergence),
        sp_exact=sp_ex,
        per_level=[dict(level=lv, median_abs_rel=med(lv), p90_abs_rel=p90(lv),
                        spearman=sp(lv), rank_retention=sp(lv) / sp_ex,
                        rank_loss=rank_loss(lv)) for lv in lvls],
        statement=(
            "The 'absolute SSA degrades EARLIER than the ranking' headline does NOT "
            "survive on additive evidence alone. It rested on the covariance-smoothing "
            "family (at +0.25 x var: 21.7% median value error while Spearman was still "
            "+0.855 of an exact-score +0.878), which is excluded from this "
            "presentation. Under the additive frozen field the two degrade in step, and "
            "at the strongest level the ranking actually degrades slightly more than "
            f"the value ({fmt_pct(med(strongest))} median value error vs "
            f"{fmt_pct(rank_loss(strongest))} ranking loss at alpha={strongest}). What "
            "additive evidence does support is a tail statement: at alpha=2 x s0 the "
            f"upper decile of cells moves by {fmt_pct(p90(strongest))} even though the "
            f"ranking retains {100 * (1 - rank_loss(strongest)):.0f}% of its "
            "exact-score Spearman, so individual reported SSA magnitudes can shift "
            "materially in a minority of cases where the ordering still looks intact."))
    return sentence, verdict


def build_markdown_additive(cfg, summary, full, sentence, verdict):
    lvls = [lv for lv in sorted({k[1] for k in summary if k[0] == 'additive'}) if lv > 0]
    md = []
    md.append("**Absolute finite-horizon SSA under additive score error.** "
              f"{cfg['n_gmm']} random {cfg['d']}-D GMMs x {len(cfg['seeds'])} seeds, "
              f"N={cfg['N']}, dt={cfg['dt']}, T={cfg['T_phys']:.0f}s, 1/N floor. The "
              "perturbation is a frozen, deterministic, state-dependent score-error "
              "field (random Fourier features) calibrated so its per-coordinate RMS "
              "equals a fraction alpha of the baseline score RMS s0; the same field is "
              "reused at every strength with only its amplitude scaled. Each perturbed "
              "run is paired with the exact-score run for the SAME GMM and seed. "
              "Statistics are |relative change| over **resolved** pairs (neither cell "
              "window-limited); window-limited cells are counted separately because "
              "their SSA is a finite-horizon lower bound.")
    md.append("")
    md.append("| Additive level | median \\|rel err\\| | IQR | p90 | signed median "
              "| ratio (med) | within 10% | window-limited |")
    md.append("|---|---|---|---|---|---|---|---|")
    for lv in lvls:
        r = summary[('additive', lv)]; res = r['resolved']
        md.append(
            f"| alpha={lv} x s0 | {fmt_pct(res['median'])} "
            f"| {fmt_pct(res['q25'])}-{fmt_pct(res['q75'])} | {fmt_pct(res['p90'])} "
            f"| {100 * r['signed_rel_median_resolved']:+.1f}% "
            f"| {r['ratio_median_resolved']:.3f} "
            f"| {fmt_pct(r['frac_within_10pct_resolved'])} "
            f"| {r['n_window_limited']}/{r['n_cells']} |")
    md.append("")
    a1 = summary[('additive', 1.0)]; a2 = summary[('additive', max(lvls))]
    md.append("The error is **not one-signed**: at alpha=1.0 the signed median "
              f"relative change is {100 * a1['strata']['low']['signed_rel_median']:+.0f}% "
              f"(low third) / {100 * a1['strata']['mid']['signed_rel_median']:+.0f}% "
              f"(mid) / {100 * a1['strata']['high']['signed_rel_median']:+.0f}% (high), "
              "i.e. the high baseline-SSA third flips sign (only "
              f"{100 * a1['strata']['high']['frac_inflated']:.0f}% of its cells are "
              "inflated), so no single bias correction applies.")
    md.append("")
    md.append("**Stratified by baseline SSA (median |rel err|, resolved pairs).** "
              "Terciles of the 50 GMMs by across-seed mean baseline SSA.")
    md.append("")
    md.append("| Additive level | low third | middle third | high third |")
    md.append("|---|---|---|---|")
    for lv in lvls:
        r = summary[('additive', lv)]
        cs = []
        for st in ['low', 'mid', 'high']:
            s = r['strata'][st]; v = s['resolved']['median']
            cs.append(f"{fmt_pct(v)} (n={s['resolved']['n']})" if np.isfinite(v)
                      else f"n/a - all {s['n_cells']} window-limited")
        md.append(f"| alpha={lv} x s0 | " + " | ".join(cs) + " |")
    md.append("")
    md.append("**Value vs ranking, on one scale.** Ranking loss is "
              "1 - Spearman(alpha)/Spearman(exact), against an exact-score Spearman of "
              f"{verdict['sp_exact']:+.3f}.")
    md.append("")
    md.append("| Additive level | median value error | p90 value error "
              "| Spearman(SSA, MI) | ranking loss |")
    md.append("|---|---|---|---|---|")
    for r in verdict['per_level']:
        md.append(f"| alpha={r['level']} x s0 | {fmt_pct(r['median_abs_rel'])} "
                  f"| {fmt_pct(r['p90_abs_rel'])} | {r['spearman']:+.3f} "
                  f"| {fmt_pct(r['rank_loss'])} |")
    md.append("")
    md.append(f"**Does absolute SSA degrade before the ranking?** {verdict['statement']}")
    md.append("")
    md.append(f"**Conservative summary.** {sentence}")
    md.append("")
    md.append("_Scope note: a second perturbation family (exact score of the "
              "Gaussian-smoothed GMM via covariance inflation) was measured over the "
              "same 50 GMMs x 3 seeds and is retained in full in "
              "`p9_absolute_sensitivity_fc.json` / `_cells.csv` / `_summary.csv`. It is "
              "deliberately excluded from this table and figure._")
    return "\n".join(md)


def make_figure_additive(cells, summary, cfg, verdict, out_base):
    """Five coherent additive-only panels (gridspec: 3 on top, 2 wider below)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.edgecolor': C_MUTED, 'axes.linewidth': 0.7,
        'text.color': C_TEXT, 'axes.labelcolor': C_TEXT,
        'xtick.color': C_MUTED, 'ytick.color': C_MUTED,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })
    lvls = [lv for lv in sorted({c['level'] for c in cells if c['family'] == 'additive'})
            if lv > 0]
    strongest = lvls[-1]
    fig = plt.figure(figsize=(9.6, 5.9))
    gs = fig.add_gridspec(2, 6, hspace=0.46, wspace=0.62)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    def sub(lv, resolved=None):
        s = [c for c in cells if c['family'] == 'additive' and c['level'] == lv]
        if resolved is True:
            s = [c for c in s if not c['wl_any']]
        if resolved is False:
            s = [c for c in s if c['wl_any']]
        return s

    # (a) paired values vs identity, alpha = 1.0 and strongest
    lim_lo, lim_hi = 0.6, 60
    ax_a.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=C_MUTED, lw=0.9, ls='--',
              zorder=1, label='identity')
    for lv, col in [(1.0, C_L1), (strongest, C_L2)]:
        r = sub(lv, True); w = sub(lv, False)
        ax_a.scatter([c['s_exact'] for c in r], [c['s_pert'] for c in r], s=12,
                     color=col, alpha=0.75, linewidths=0, zorder=3,
                     label=f'alpha={lv} (resolved)')
        ax_a.scatter([c['s_exact'] for c in w], [c['s_pert'] for c in w], s=16,
                     facecolors='none', edgecolors=col, linewidths=0.9, zorder=4,
                     label=f'alpha={lv} (window-limited)')
    ax_a.set_xscale('log'); ax_a.set_yscale('log')
    ax_a.set_xlim(lim_lo, lim_hi); ax_a.set_ylim(lim_lo, lim_hi)
    ax_a.set_xlabel(r'exact-score $\hat S$'); ax_a.set_ylabel(r'perturbed $\hat S$')
    ax_a.set_title('(a) Paired values vs identity', loc='left')
    ax_a.legend(frameon=False, loc='upper left', fontsize=5.8, handletextpad=0.4)
    ax_a.grid(True, color=C_GRID, lw=0.5, zorder=0); ax_a.set_axisbelow(True)

    # (b) relative change vs exact SSA at the strongest level
    ax_b.axhline(0, color=C_MUTED, lw=0.9, ls='--', zorder=1)
    r = sub(strongest, True); w = sub(strongest, False)
    ax_b.scatter([c['s_exact'] for c in r], [100 * c['rel'] for c in r], s=12,
                 color=C_L2, alpha=0.75, linewidths=0, zorder=3, label='resolved')
    ax_b.scatter([c['s_exact'] for c in w], [100 * c['rel'] for c in w], s=16,
                 facecolors='none', edgecolors=C_L2, linewidths=0.9, zorder=4,
                 label='window-limited')
    ax_b.set_xscale('log')
    ax_b.set_xlabel(r'exact-score $\hat S$'); ax_b.set_ylabel('relative change (%)')
    ax_b.set_title(f'(b) Relative change (alpha={strongest})', loc='left')
    ax_b.legend(frameon=False, loc='lower left', fontsize=6.4)
    ax_b.grid(True, color=C_GRID, lw=0.5, zorder=0); ax_b.set_axisbelow(True)

    # (c) median |rel| vs alpha, IQR band
    allv = [0.0] + lvls
    med = [100 * summary[('additive', lv)]['resolved']['median'] for lv in allv]
    q25 = [100 * summary[('additive', lv)]['resolved']['q25'] for lv in allv]
    q75 = [100 * summary[('additive', lv)]['resolved']['q75'] for lv in allv]
    p90 = [100 * summary[('additive', lv)]['resolved']['p90'] for lv in allv]
    ax_c.fill_between(allv, q25, q75, color=C_L1, alpha=0.16, linewidth=0, zorder=2,
                      label='IQR')
    ax_c.plot(allv, med, '-o', color=C_L1, lw=2, ms=4.5, zorder=4, label='median')
    ax_c.plot(allv, p90, ':', color=C_L1, lw=1.4, zorder=3, label='p90')
    ax_c.axhline(10, color=C_MUTED, lw=0.8, ls=':', zorder=1)
    ax_c.text(0.02, 10, ' 10%', color=C_MUTED, fontsize=6.5, va='bottom')
    ax_c.set_xlabel(r'additive level  $\alpha \times s_0$')
    ax_c.set_ylabel(r'$|\Delta \hat S / \hat S|$ (%)')
    ax_c.set_title('(c) Error vs level (resolved)', loc='left')
    ax_c.legend(frameon=False, loc='upper left', fontsize=6.4)
    ax_c.grid(True, color=C_GRID, lw=0.5, zorder=0); ax_c.set_axisbelow(True)

    # (d) stratified by baseline SSA tercile, alpha = 1.0 and strongest
    sts = ['low', 'mid', 'high']
    x = np.arange(len(sts)); bw = 0.36
    for i, (lv, col) in enumerate([(1.0, C_L1), (strongest, C_L2)]):
        vals, hatched = [], []
        for st in sts:
            s = summary[('additive', lv)]['strata'][st]
            v = s['resolved']['median']
            hatched.append(not np.isfinite(v))
            vals.append(100 * (v if np.isfinite(v) else s['all_cells']['median']))
        bars = ax_d.bar(x + (i - 0.5) * bw, vals, bw, color=col, zorder=3,
                        label=f'alpha={lv}')
        for b, v, hf in zip(bars, vals, hatched):
            if hf:
                b.set_hatch('///'); b.set_edgecolor('white')
            ax_d.text(b.get_x() + b.get_width() / 2, v, f'{v:.1f}', ha='center',
                      va='bottom', fontsize=6.6, color=C_TEXT)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f'{s} baseline SSA' for s in sts])
    ax_d.set_ylabel(r'median $|\Delta \hat S / \hat S|$ (%)')
    ax_d.set_title('(d) Stratified by baseline SSA', loc='left')
    ax_d.legend(frameon=False, loc='upper left', fontsize=6.6)
    ax_d.grid(True, axis='y', color=C_GRID, lw=0.5, zorder=0); ax_d.set_axisbelow(True)

    # (e) value error vs ranking loss — both percentages, ONE axis
    pl = verdict['per_level']
    xs = [r['level'] for r in pl]
    ax_e.plot(xs, [100 * r['median_abs_rel'] for r in pl], '-o', color=C_L1, lw=2,
              ms=4.5, zorder=4, label='median value error')
    ax_e.plot(xs, [100 * r['p90_abs_rel'] for r in pl], ':', color=C_L1, lw=1.4,
              zorder=3, label='p90 value error')
    ax_e.plot(xs, [100 * r['rank_loss'] for r in pl], '-s', color=C_L2, lw=2, ms=4.5,
              zorder=4, label='ranking loss (1 - Spearman ratio)')
    # direct labels: value error always BELOW its point, ranking loss ABOVE its point,
    # so the two never collide even where the series cross or coincide.
    for r in pl:
        if r['level'] in (1.0, strongest):
            ax_e.annotate(f"{100 * r['median_abs_rel']:.0f}%",
                          (r['level'], 100 * r['median_abs_rel']),
                          textcoords='offset points', xytext=(-2, -12), ha='right',
                          fontsize=6.4, color=C_L1)
            ax_e.annotate(f"{100 * r['rank_loss']:.0f}%",
                          (r['level'], 100 * r['rank_loss']),
                          textcoords='offset points', xytext=(-2, 7), ha='right',
                          fontsize=6.4, color=C_L2)
    ax_e.set_xlabel(r'additive level  $\alpha \times s_0$')
    ax_e.set_ylabel('degradation (%)')
    ax_e.set_title('(e) Value error and ranking loss degrade together', loc='left')
    ax_e.legend(frameon=False, loc='upper left', fontsize=6.4)
    ax_e.grid(True, color=C_GRID, lw=0.5, zorder=0); ax_e.set_axisbelow(True)

    fig.suptitle('P1/P9: absolute finite-horizon SSA under additive score error '
                 f"(d={cfg['d']}, N={cfg['N']}, T={cfg['T_phys']:.0f}s, "
                 f"{cfg['n_gmm']} GMMs x {len(cfg['seeds'])} seeds; "
                 'hollow markers / hatched bars = window-limited lower bounds)',
                 fontsize=8.4, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(f'{out_base}.{ext}', dpi=200 if ext == 'png' else None,
                    bbox_inches='tight')
    plt.close(fig)
    return [f'{out_base}.png', f'{out_base}.pdf']


def build_markdown(cfg, summary, full, strata, headline):
    md = []
    md.append("**Absolute finite-horizon SSA under controlled score error.** "
              f"{cfg['n_gmm']} random {cfg['d']}-D GMMs x {len(cfg['seeds'])} seeds, "
              f"N={cfg['N']}, dt={cfg['dt']}, T={cfg['T_phys']:.0f}s, 1/N floor. Each "
              "perturbed run is paired with the exact-score run for the SAME GMM and "
              "seed. Statistics are |relative change| over **resolved** pairs "
              "(neither cell window-limited); window-limited cells are counted "
              "separately because their SSA is a finite-horizon lower bound.")
    md.append("")
    md.append("| Perturbation | Level | median \\|rel err\\| | IQR | p90 | signed median "
              "| ratio (med) | within 10% | window-limited |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    name = {'additive': 'Additive frozen field', 'smoothing': 'Covariance smoothing'}
    unit = {'additive': 'a=', 'smoothing': '+'}
    for fam in ['additive', 'smoothing']:
        for lv in sorted({k[1] for k in summary if k[0] == fam}):
            if lv == 0.0:
                continue
            r = summary[(fam, lv)]
            res = r['resolved']
            md.append(
                f"| {name[fam]} | {unit[fam]}{lv} | {fmt_pct(res['median'])} "
                f"| {fmt_pct(res['q25'])}-{fmt_pct(res['q75'])} | {fmt_pct(res['p90'])} "
                f"| {100 * r['signed_rel_median_resolved']:+.1f}% "
                f"| {r['ratio_median_resolved']:.3f} "
                f"| {fmt_pct(r['frac_within_10pct_resolved'])} "
                f"| {r['n_window_limited']}/{r['n_cells']} |")
    md.append("")
    sm1 = summary[('smoothing', 1.0)]; ad1 = summary[('additive', 1.0)]
    md.append(
        "The error is **not one-signed**, so no single bias correction applies. Among "
        "resolved pairs at the strongest levels the signed median relative change "
        "shrinks monotonically with baseline SSA - smoothing +1.0: "
        f"{100 * sm1['strata']['low']['signed_rel_median']:+.0f}% (low) / "
        f"{100 * sm1['strata']['mid']['signed_rel_median']:+.0f}% (mid) / "
        f"{100 * sm1['strata']['high']['signed_rel_median']:+.0f}% (high); "
        f"additive a=1.0: {100 * ad1['strata']['low']['signed_rel_median']:+.0f}% / "
        f"{100 * ad1['strata']['mid']['signed_rel_median']:+.0f}% / "
        f"{100 * ad1['strata']['high']['signed_rel_median']:+.0f}% - and the additive "
        "high third actually flips sign (only "
        f"{100 * ad1['strata']['high'].get('frac_inflated', float('nan')):.0f}% inflated). "
        "Panel (b) also shows strongly deflated cells (-50% to -75%) at large "
        "exact-score SSA, but those are predominantly **window-limited**, so they are "
        "lower bounds and are excluded from these resolved statistics rather than "
        "reported as resolved deflation.")
    md.append("")
    md.append("**Stratified by baseline SSA (median |rel err|, resolved pairs).** "
              "Terciles of the 50 GMMs by across-seed mean baseline SSA.")
    md.append("")
    md.append("| Perturbation | Level | low third | middle third | high third |")
    md.append("|---|---|---|---|---|")
    for fam in ['additive', 'smoothing']:
        for lv in sorted({k[1] for k in summary if k[0] == fam}):
            if lv == 0.0:
                continue
            r = summary[(fam, lv)]
            cells_txt = []
            for st in ['low', 'mid', 'high']:
                s = r['strata'][st]
                v = s['resolved']['median']
                if np.isfinite(v):
                    cells_txt.append(f"{fmt_pct(v)} (n={s['resolved']['n']})")
                else:
                    cells_txt.append(f"n/a - all {s['n_cells']} window-limited")
            md.append(f"| {name[fam]} | {unit[fam]}{lv} | " + " | ".join(cells_txt) + " |")
    md.append("")
    md.append("**Ranking vs value.** Spearman(SSA, MI) across the same 50 GMMs "
              "(from P1) is shown for contrast:")
    md.append("")
    md.append("| Perturbation | Level | Spearman(SSA, MI) | median \\|rel err\\| |")
    md.append("|---|---|---|---|")
    for fam, key in [('additive', 'additive'), ('smoothing', 'smoothing')]:
        for lv in sorted({k[1] for k in summary if k[0] == fam}):
            if lv == 0.0:
                continue
            sp = full[key]['spearman'][str(lv)]
            md.append(f"| {name[fam]} | {unit[fam]}{lv} | {sp['mean']:+.3f} "
                      f"+/- {sp['sd']:.3f} | "
                      f"{fmt_pct(summary[(fam, lv)]['resolved']['median'])} |")
    md.append("")
    md.append(f"**Conservative summary.** {headline}")
    md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no_fig', action='store_true')
    ap.add_argument('--present', choices=['additive', 'all'], default='additive',
                    help="which family the rebuttal TABLE and FIGURE present. "
                         "'additive' (default) restricts presentation to the frozen "
                         "additive score-error field; the covariance-smoothing family "
                         "is still measured and retained in the JSON/CSV outputs.")
    args = ap.parse_args()

    full, cfg, wl, keep = load_p1()
    print("=== P9: absolute SSA sensitivity to score error ===")
    print(f"source: results/score_robustness_full_fc.json + 50 shard files")
    print(f"d={cfg['d']} N={cfg['N']} dt={cfg['dt']} T_phys={cfg['T_phys']}s "
          f"floor=1/N={cfg['floor_1N']:g} n_gmm={cfg['n_gmm']} seeds={cfg['seeds']}")

    cells, base, mi = build_cells(full, cfg, wl, keep)
    strata, per_gmm_base = strata_from_baseline(base)
    summary = summarize(cells, strata)
    checks_ok = run_checks(full, cfg, cells, summary, strata)

    # ---- console: headline table ----
    print("--- |relative error| by family and level (RESOLVED pairs) ---")
    print(f"{'family':<11}{'level':>7}{'n_res':>7}{'median':>9}{'IQR':>17}"
          f"{'p90':>9}{'ratio':>8}{'WL':>9}")
    for fam in ['additive', 'smoothing']:
        for lv in sorted({k[1] for k in summary if k[0] == fam}):
            r = summary[(fam, lv)]; res = r['resolved']
            print(f"{fam:<11}{lv:>7}{res['n']:>7}{fmt_pct(res['median']):>9}"
                  f"{fmt_pct(res['q25']) + '-' + fmt_pct(res['q75']):>17}"
                  f"{fmt_pct(res['p90']):>9}{r['ratio_median_resolved']:>8.3f}"
                  f"{str(r['n_window_limited']) + '/' + str(r['n_cells']):>9}")

    print("\n--- stratified by baseline SSA tercile (median |rel err|, resolved) ---")
    print(f"{'family':<11}{'level':>7}{'low':>22}{'mid':>22}{'high':>22}")
    for fam in ['additive', 'smoothing']:
        for lv in sorted({k[1] for k in summary if k[0] == fam}):
            r = summary[(fam, lv)]
            row = []
            for st in ['low', 'mid', 'high']:
                s = r['strata'][st]; v = s['resolved']['median']
                row.append(f"{fmt_pct(v)} (n={s['resolved']['n']})" if np.isfinite(v)
                           else f"n/a all-WL (n={s['n_cells']})")
            print(f"{fam:<11}{lv:>7}" + "".join(f"{c:>22}" for c in row))

    # ---- conservative headline sentence (derived, not asserted) ----
    def med(fam, lv):
        return summary[(fam, lv)]['resolved']['median']
    add_lvls = [lv for lv in sorted({k[1] for k in summary if k[0] == 'additive'})
                if lv > 0]
    smo_lvls = [lv for lv in sorted({k[1] for k in summary if k[0] == 'smoothing'})
                if lv > 0]
    add_ok = [lv for lv in add_lvls if np.isfinite(med('additive', lv))
              and med('additive', lv) <= 0.10]
    smo_ok = [lv for lv in smo_lvls if np.isfinite(med('smoothing', lv))
              and med('smoothing', lv) <= 0.10]
    add_break = next((lv for lv in add_lvls if np.isfinite(med('additive', lv))
                      and med('additive', lv) > 0.10), None)
    smo_break = next((lv for lv in smo_lvls if np.isfinite(med('smoothing', lv))
                      and med('smoothing', lv) > 0.10), None)
    def p90(fam, lv):
        return summary[(fam, lv)]['resolved']['p90']

    def worst_stratum(fam, lv):
        """Which baseline-SSA third has the largest median |rel err| (computed)."""
        vals = {st: summary[(fam, lv)]['strata'][st]['resolved']['median']
                for st in ['low', 'mid', 'high']}
        vals = {k: v for k, v in vals.items() if np.isfinite(v)}
        if not vals:
            return None, np.nan
        st = max(vals, key=vals.get)
        return st, vals[st]

    # ranking contrast: where does Spearman still look intact while the value has moved?
    sp = {('additive', lv): full['additive']['spearman'][str(lv)]['mean']
          for lv in add_lvls}
    sp.update({('smoothing', lv): full['smoothing']['spearman'][str(lv)]['mean']
               for lv in smo_lvls})
    sp_base = full['baseline']['spearman']['mean']
    # first level where the median value error exceeds 10% while Spearman retains
    # >=90% of its exact-score value (i.e. the value moved but the ranking did not)
    divergence = None
    for fam, lvls in [('smoothing', smo_lvls), ('additive', add_lvls)]:
        for lv in lvls:
            if (np.isfinite(med(fam, lv)) and med(fam, lv) > 0.10
                    and sp[(fam, lv)] >= 0.90 * sp_base):
                divergence = (fam, lv, med(fam, lv), sp[(fam, lv)])
                break
        if divergence:
            break

    st_a, va = worst_stratum('additive', max(add_lvls))
    st_s, vs = worst_stratum('smoothing', max(smo_lvls))
    fam_word = {'additive': 'additive score error', 'smoothing': 'covariance smoothing'}
    headline = (
        f"On resolved pairs (neither cell window-limited) the absolute finite-horizon "
        f"SSA stays within a median 10% of its exact-score value for additive score "
        f"error up to alpha={max(add_ok) if add_ok else 0} x s0 "
        f"(median {fmt_pct(med('additive', max(add_ok))) if add_ok else 'n/a'}, "
        f"p90 {fmt_pct(p90('additive', max(add_ok))) if add_ok else 'n/a'}) but only up "
        f"to +{max(smo_ok) if smo_ok else 0} x var of covariance smoothing "
        f"(median {fmt_pct(med('smoothing', max(smo_ok))) if smo_ok else 'n/a'}, "
        f"p90 {fmt_pct(p90('smoothing', max(smo_ok))) if smo_ok else 'n/a'}), "
        f"degrading to {fmt_pct(med('additive', add_break)) if add_break else 'n/a'} at "
        f"alpha={add_break} and {fmt_pct(med('smoothing', smo_break)) if smo_break else 'n/a'} "
        f"at +{smo_break} x var; the two families also stratify oppositely at their "
        f"strongest level - additive error is largest for the {st_a} baseline-SSA third "
        f"({fmt_pct(va)}) whereas smoothing error is largest for the {st_s} third "
        f"({fmt_pct(vs)}) - and, critically, "
        + (f"absolute SSA degrades EARLIER than the ranking: at {fam_word[divergence[0]]} "
           f"{divergence[1]} the median value error is already {fmt_pct(divergence[2])} "
           f"while Spearman(SSA, MI) is {divergence[3]:+.3f} against an exact-score "
           f"{sp_base:+.3f}, so ranking stability must not be reported as evidence that "
           f"the reported SSA magnitude is stable."
           if divergence else
           "no level was found where the value degrades while the ranking holds.")
    )
    # ---- additive-only presentation (the rebuttal restriction) ----
    add_sentence, add_verdict = additive_headline(summary, full)

    if args.present == 'additive':
        headline_presented = add_sentence
        print("\n--- HEADLINE TEST ON ADDITIVE EVIDENCE ALONE ---")
        print(f"  divergence levels (value>10% while ranking retains >=90%): "
              f"{add_verdict['divergence_levels'] or 'NONE'}")
        print(f"  headline 'absolute degrades before ranking' survives: "
              f"{add_verdict['headline_survives']}")
        print(f"  {'level':>7} {'med|rel|':>10} {'p90':>8} {'Spearman':>10} "
              f"{'rank loss':>10}")
        for r in add_verdict['per_level']:
            print(f"  {r['level']:>7} {fmt_pct(r['median_abs_rel']):>10} "
                  f"{fmt_pct(r['p90_abs_rel']):>8} {r['spearman']:>+10.3f} "
                  f"{fmt_pct(r['rank_loss']):>10}")
        print("\n  " + add_verdict['statement'])
    else:
        headline_presented = headline

    print("\n--- CONSERVATIVE SUMMARY SENTENCE (presented: "
          f"{args.present}) ---")
    print(headline_presented)

    # ---- save artifacts ----
    os.makedirs(RESULTS, exist_ok=True)
    ser = {f"{fam}|{lv}": v for (fam, lv), v in summary.items()}
    out = dict(
        source=dict(full='score_robustness_full_fc.json',
                    shards=len(glob.glob(os.path.join(
                        RESULTS, 'score_robustness_full_fc_shard_*.json')))),
        config=cfg,
        stratification=dict(rule='terciles of across-seed mean baseline SSA per GMM',
                            per_gmm_baseline_mean=per_gmm_base.tolist(),
                            labels={str(k): v for k, v in strata.items()}),
        summary=ser,
        spearman_for_contrast=dict(
            additive=full['additive']['spearman'],
            smoothing=full['smoothing']['spearman'],
            baseline=full['baseline']['spearman']),
        window_limited_note=(
            'Window-limited cells never reached the 1/N floor within the horizon; '
            'their Shat is a finite-horizon lower bound, so their relative error is '
            'NOT a resolved infinite-horizon error. Headline statistics use resolved '
            'pairs only.'),
        headline=headline,
        headline_additive_only=add_sentence,
        additive_only_verdict=add_verdict,
        presentation_note=(
            'The rebuttal table and figure present the ADDITIVE family only. The '
            'covariance-smoothing family was measured over the same 50 GMMs x 3 seeds '
            'and is retained in full in this JSON and in the CSV outputs; its '
            'exclusion is a presentation restriction, not a deletion of data.'),
        presented_family=args.present,
        checks_passed=bool(checks_ok))
    p_json = os.path.join(RESULTS, 'p9_absolute_sensitivity_fc.json')
    with open(p_json, 'w') as fh:
        json.dump(out, fh, indent=2)

    p_cells = os.path.join(RESULTS, 'p9_absolute_sensitivity_fc_cells.csv')
    with open(p_cells, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0].keys()) + ['stratum'])
        w.writeheader()
        for c in cells:
            row = dict(c); row['stratum'] = strata[c['gid']]; w.writerow(row)

    p_sum = os.path.join(RESULTS, 'p9_absolute_sensitivity_fc_summary.csv')
    with open(p_sum, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['family', 'level', 'n_cells', 'n_window_limited', 'n_resolved',
                    'median_abs_rel', 'q25', 'q75', 'p90', 'ratio_median',
                    'frac_within_10pct',
                    'low_median', 'mid_median', 'high_median'])
        for (fam, lv), r in sorted(summary.items()):
            res = r['resolved']
            w.writerow([fam, lv, r['n_cells'], r['n_window_limited'], res['n'],
                        res['median'], res['q25'], res['q75'], res['p90'],
                        r['ratio_median_resolved'], r['frac_within_10pct_resolved'],
                        r['strata']['low']['resolved']['median'],
                        r['strata']['mid']['resolved']['median'],
                        r['strata']['high']['resolved']['median']])

    if args.present == 'additive':
        md_text = build_markdown_additive(cfg, summary, full, add_sentence, add_verdict)
        p_md = os.path.join(RESULTS, 'p9_absolute_sensitivity_fc_additive.md')
    else:
        md_text = build_markdown(cfg, summary, full, strata, headline)
        p_md = os.path.join(RESULTS, 'p9_absolute_sensitivity_fc.md')
    with open(p_md, 'w') as fh:
        fh.write(md_text + "\n")

    figs = []
    if not args.no_fig:
        if args.present == 'additive':
            figs = make_figure_additive(
                cells, summary, cfg, add_verdict,
                os.path.join(FIGDIR, 'p1_score_robustness_absolute_additive'))
        else:
            figs = make_figure(cells, summary, cfg, strata,
                               os.path.join(FIGDIR, 'p1_score_robustness_absolute'))

    print("\n--- artifacts ---")
    for p in [p_json, p_cells, p_sum, p_md] + figs:
        print(f"  {p}")
    print("\n" + md_text)


if __name__ == '__main__':
    main()
