"""Generate LaTeX table and summary for App. H weight comparability."""
import json, numpy as np, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, 'weight_table.json')) as f:
    data = json.load(f)

# Parse into prompt -> {gs -> metrics}
prompts_map = {}
for cell, r in data.items():
    if cell.startswith('traj_'):
        parts = cell.replace('traj_', '').replace('_dt0p005', '').rsplit('_gs', 1)
        prompt = parts[0]
    elif cell.startswith('ext_'):
        parts = cell.replace('_dt0p005', '').rsplit('_gs', 1)
        prompt = parts[0].replace('ext_p0_', '').replace('ext_p1_', '') \
                         .replace('ext_p2_', '').replace('ext_p3_', '') \
                         .replace('ext_p4_', '').replace('ext_p5_', '')
    else:
        continue
    gs = parts[1].replace('p', '.')
    if prompt not in prompts_map:
        prompts_map[prompt] = {}
    prompts_map[prompt][gs] = r

gs_order = ['1.0', '3.0', '7.5', '15.0']

# Display order: 8 original first, then 6 extended
orig = ['person', 'elderly', 'cat', 'food', 'mountains', 'car', 'flowers', 'street']
ext = ['bat', 'crane', 'mouse', 'seal', 'jaguar', 'palm']
all_prompts = orig + ext

# Compute per-prompt summary
ratios = {}
for prompt in all_prompts:
    vals = prompts_map[prompt]
    w2s = [vals[gs]['w_2'] for gs in gs_order if gs in vals]
    w23s = [vals[gs]['w_2_w_3'] for gs in gs_order if gs in vals]
    rhoNs = [vals[gs]['rho_T_N'] for gs in gs_order if gs in vals]
    if len(w2s) == 4:
        ratios[prompt] = {
            'w2_ratio': max(w2s) / min(w2s),
            'w23_ratio': max(w23s) / min(w23s),
            'min_rhoN': min(rhoNs),
        }

# --- LaTeX table ---
lines = []
lines.append(r"""\begin{table}[t]
\centering
\caption{Slow-mode weight comparability across guidance scales.
For each prompt$\times$scale cell we report $w_2 = \lambda_1(\hat C^{\mathrm{sym}}(T))/\operatorname{tr}\hat C(T)$,
the two-mode fraction $w_{2{+}3} = (\lambda_1{+}\lambda_2)/\operatorname{tr}\hat C(T)$,
and the convergence proxy $\hat\rho^2(T)\!\cdot\! N$ (values $\gg 1$ indicate the autocorrelation
remains well above the $1/N$ noise floor at $T{=}2000$).
Summary columns: $R_{w_2}$ and $R_{w_{2+3}}$ are the max/min ratio of each
weight across the four scales; $(\hat\rho^2 N)_{\min}$ is the worst-case
convergence proxy.}
\label{tab:weight_comparability}
\small
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l | ccc | ccc | ccc | ccc | cc c}
\toprule
& \multicolumn{3}{c|}{$w{=}1$}
& \multicolumn{3}{c|}{$w{=}3$}
& \multicolumn{3}{c|}{$w{=}7.5$}
& \multicolumn{3}{c|}{$w{=}15$}
& $R_{w_2}$ & $R_{w_{2+3}}$ & $(\hat\rho^2\!N)_{\min}$ \\
Prompt
& $w_2$ & $w_{2{+}3}$ & $\hat\rho^2\!N$
& $w_2$ & $w_{2{+}3}$ & $\hat\rho^2\!N$
& $w_2$ & $w_{2{+}3}$ & $\hat\rho^2\!N$
& $w_2$ & $w_{2{+}3}$ & $\hat\rho^2\!N$
& & & \\
\midrule""")

for i, prompt in enumerate(all_prompts):
    vals = prompts_map[prompt]
    r = ratios[prompt]
    label = prompt.replace('_', r'\_')

    cols = []
    for gs in gs_order:
        v = vals[gs]
        cols.append(f"{v['w_2']:.2f} & {v['w_2_w_3']:.2f} & {v['rho_T_N']:.2f}")

    line = (f"{label} & {cols[0]} & {cols[1]} & {cols[2]} & {cols[3]} "
            f"& {r['w2_ratio']:.1f} & {r['w23_ratio']:.1f} & {r['min_rhoN']:.2f} \\\\")
    lines.append(line)
    if i == len(orig) - 1:
        lines.append(r"\midrule")

# Summary rows
w2_all = [ratios[p]['w2_ratio'] for p in all_prompts]
w23_all = [ratios[p]['w23_ratio'] for p in all_prompts]

near_conv = [p for p in all_prompts if ratios[p]['min_rhoN'] >= 0.08]
w2_nc = [ratios[p]['w2_ratio'] for p in near_conv]
w23_nc = [ratios[p]['w23_ratio'] for p in near_conv]

lines.append(r"\midrule")
lines.append(r"\multicolumn{16}{l}{\textit{Summary statistics (median / max across prompts)}} \\")
lines.append(f"All 14 prompts & \\multicolumn{{12}}{{r|}}{{}} "
             f"& {np.median(w2_all):.1f}/{max(w2_all):.1f} "
             f"& {np.median(w23_all):.1f}/{max(w23_all):.1f} & \\\\")
lines.append(f"Truncation-relevant ({len(near_conv)}) & \\multicolumn{{12}}{{r|}}{{}} "
             f"& {np.median(w2_nc):.1f}/{max(w2_nc):.1f} "
             f"& {np.median(w23_nc):.1f}/{max(w23_nc):.1f} & \\\\")

lines.append(r"""\bottomrule
\end{tabular}}% end resizebox
\end{table}""")

tex = '\n'.join(lines)
tex_path = os.path.join(SCRIPT_DIR, 'weight_comparability_full_table.tex')
with open(tex_path, 'w') as f:
    f.write(tex)
print(f"Wrote {tex_path}")

# --- Summary paragraph ---
# Identify cleanest and noisiest in near-conv set
nc_sorted = sorted(near_conv, key=lambda p: ratios[p]['w2_ratio'])
cleanest = nc_sorted[0]
noisiest = nc_sorted[-1]

summary = f"""Weight-comparability summary (App.~H):

Across all 14 prompts, the max/min ratio of the leading slow-mode weight
$w_2 = \\lambda_1(\\hat C^{{\\mathrm{{sym}}}}(T))/\\operatorname{{tr}}\\hat C(T)$ across the four
guidance scales has median {np.median(w2_all):.1f}$\\times$ and maximum
{max(w2_all):.1f}$\\times$ (street); the two-mode fraction $w_2+w_3$ shows median
{np.median(w23_all):.1f}$\\times$ and maximum {max(w23_all):.1f}$\\times$.
Restricting to the {len(near_conv)} prompts whose worst-case convergence proxy
$\\hat\\rho^2(T)\\cdot N$ exceeds $0.08$ at every scale
({', '.join(nc_sorted)}),
the ratios tighten to median {np.median(w2_nc):.1f}$\\times$, max
{max(w2_nc):.1f}$\\times$ for $w_2$ and median {np.median(w23_nc):.1f}$\\times$,
max {max(w23_nc):.1f}$\\times$ for $w_2+w_3$.
The cleanest case is \\textit{{{cleanest}}} ($R_{{w_2}}={ratios[cleanest]['w2_ratio']:.2f}$),
whose polysemy produces a single dominant slow mode with stable weight across
all four scales; the noisiest near-converged prompt is
\\textit{{{noisiest}}} ($R_{{w_2}}={ratios[noisiest]['w2_ratio']:.2f}$),
where the leading eigenvalue at $w=1$ is inflated by residual noise
($\\hat\\rho^2 N={prompts_map[noisiest]['1.0']['rho_T_N']:.2f}$, closest to
the floor among its four cells).
"""

summary_path = os.path.join(SCRIPT_DIR, 'weight_comparability_summary.tex')
with open(summary_path, 'w') as f:
    f.write(summary)
print(f"Wrote {summary_path}")

# Also print for review
print("\n" + summary)
print(f"\nTruncation-relevant prompts (min ρ²N ≥ 0.08): {', '.join(nc_sorted)}")
print(f"Cleanest: {cleanest} (w2 ratio {ratios[cleanest]['w2_ratio']:.2f})")
print(f"Noisiest near-conv: {noisiest} (w2 ratio {ratios[noisiest]['w2_ratio']:.2f})")
