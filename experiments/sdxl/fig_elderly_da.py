#!/usr/bin/env python3
"""
Elderly DA figures: main grid (Fig 4), spectrum (Fig A7), histograms (Fig A8).

Modes:
  --mode grid      : DA1 vs PC1 image grid (Fig 4, default)
  --mode spectrum  : Eigenvalue spectrum + MC null + free-conv density (Fig A7)
  --mode histogram : DA1/DA2 projection histograms (Fig A8)
  --mode all       : All panels

Reads:
  results/clip_da_elderly_gs7p5_fc_dt0p0025.{json,npz}  (DA results)
  results/gen_mp_p5_elderly_gs7p5_quick_x0.npz            (x0 latents, grid mode)
  results/gen_mp_p5_elderly_gs7p5_quick_clip.npz           (CLIP embeddings, histogram mode)
  results/elderly_gs7p5_n200_C_tau_full.npz                (C(τ) data, spectrum mode)
  results/c0_eigvals_elderly_gs7p5_fc_dt0p0025.npz         (C(0) eigenvalues, spectrum mode)
"""

import os, sys, json, argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
VAE_SCALE = 0.13025


# ============================================================
# Fig 4: DA1 vs PC1 image grid
# ============================================================

def decode_latents(vae, latents_flat, latent_shape, device, batch_size=8):
    import torch
    from PIL import Image
    images = []
    N = latents_flat.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = torch.from_numpy(latents_flat[start:end]).float().to(device)
        z = batch.view(-1, *latent_shape) / VAE_SCALE
        with torch.no_grad():
            decoded = vae.decode(z, return_dict=False)[0]
        pixels = ((decoded + 1.0) / 2.0).clamp(0, 1)
        pixel_bytes = (pixels * 255).byte().cpu()
        for img_t in pixel_bytes:
            images.append(Image.fromarray(img_t.permute(1, 2, 0).numpy()))
    return images


def make_grid_figure(args):
    """DA1 vs PC1 grid (Fig 4). Requires GPU for VAE decode + CLIP."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_shape = (4, 128, 128)

    print("Loading SDXL VAE...")
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    vae = pipe.vae.to(device).float().eval()
    del pipe.unet, pipe
    torch.cuda.empty_cache()

    js_path = os.path.join(RESULTS_DIR, f"clip_da_{args.name}.json")
    npz_path = os.path.join(RESULTS_DIR, f"clip_da_{args.name}.npz")
    js = json.load(open(js_path))
    da_npz = np.load(npz_path)

    if args.use_taustar:
        V = da_npz["eigvecs_full_topk_at_tau_star"]
        da1 = V[:, 0].astype(np.float64)
        da2 = V[:, 1].astype(np.float64)
        tau_star = int(da_npz["tau_star_freeconv"])
        print(f"Using DA1/DA2 from C(tau* freeconv = {tau_star})")
    else:
        da1 = da_npz["da1_eigvec"]
        da2 = da_npz["da2_eigvec"]
        print("Using DA1/DA2 from peak-gap lag")

    x0_path = os.path.join(RESULTS_DIR, "gen_mp_p5_elderly_gs7p5_quick_x0.npz")
    x0_flat = np.load(x0_path)["x0"]
    print(f"Decoding {x0_flat.shape[0]} images...")
    images = decode_latents(vae, x0_flat, latent_shape, device)

    from sdxl_components import RealCLIPEmbedder
    clip_embedder = RealCLIPEmbedder(vae, device=device, clip_batch_size=16)
    z0_clip = clip_embedder.embed(x0_flat, latent_shape=latent_shape)

    z0c = z0_clip - z0_clip.mean(axis=0)
    cov = (z0c.T @ z0c) / z0_clip.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1].copy()
    pc1 = eigvecs[:, idx[0]]
    cos_da1_pc1 = abs(float(np.dot(da1, pc1)))

    da1_scores = z0c @ da1
    pc1_scores = z0c @ pc1

    k = args.k
    da1_order = np.argsort(da1_scores)
    pc1_order = np.argsort(pc1_scores)

    da1_low = da1_order[:k]
    da1_high = da1_order[-k:][::-1]
    pc1_low = pc1_order[:k]
    pc1_high = pc1_order[-k:][::-1]

    print(f"|cos(DA1, PC1)| = {cos_da1_pc1:.4f}")

    # Probe correlations
    from scipy.stats import spearmanr
    from transformers import CLIPModel, CLIPTokenizer
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model.eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    probe_pairs = {
        "gender": ("a photograph of a man", "a photograph of a woman"),
        "age": ("a photograph of a young person", "a photograph of an old person"),
        "realism": ("a realistic photograph", "an artistic painting"),
    }
    for name, (pos, neg) in probe_pairs.items():
        with torch.no_grad():
            tok_p = tokenizer(pos, return_tensors="pt", padding=True).to(device)
            tok_n = tokenizer(neg, return_tensors="pt", padding=True).to(device)
            emb_p = clip_model.get_text_features(**tok_p)
            emb_n = clip_model.get_text_features(**tok_n)
            emb_p = emb_p / emb_p.norm(dim=-1, keepdim=True)
            emb_n = emb_n / emb_n.norm(dim=-1, keepdim=True)
            axis = (emb_p - emb_n).squeeze().cpu().numpy()
            axis = axis / np.linalg.norm(axis)
        proj = z0c @ axis
        rho_da1, _ = spearmanr(da1_scores, proj)
        rho_pc1, _ = spearmanr(pc1_scores, proj)
        print(f"  {name}: DA1 ρ={rho_da1:.3f}, PC1 ρ={rho_pc1:.3f}")

    # --- Figure ---
    thumb = 180
    fig_w = k * thumb / 100 + 1.5
    fig_h = 5.5
    fig = plt.figure(figsize=(fig_w + 1, fig_h))
    gs = GridSpec(4, k + 1, figure=fig, hspace=0.15, wspace=0.05,
                  left=0.01, right=0.99, top=0.98, bottom=0.02,
                  width_ratios=[0.6] + [1] * k)

    def place_row(row, indices, label_text):
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.text(0.9, 0.5, label_text, ha="right", va="center",
                      fontsize=9, fontweight="bold", transform=ax_label.transAxes)
        ax_label.axis("off")
        for col, img_idx in enumerate(indices):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(images[img_idx])
            ax.axis("off")

    place_row(0, da1_high, "DA1\nhigh")
    place_row(1, da1_low, "DA1\nlow")
    place_row(2, pc1_high, "PC1\nhigh")
    place_row(3, pc1_low, "PC1\nlow")

    out_path = os.path.join(RESULTS_DIR, "fig_elderly_da_vs_pca_freeconv.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    hires = out_path.replace(".png", "_hires.png")
    fig.savefig(hires, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close(fig)

    # Save indices
    out_npz = out_path.replace(".png", "_indices.npz")
    np.savez_compressed(out_npz,
        da1_high=da1_high, da1_low=da1_low,
        pc1_high=pc1_high, pc1_low=pc1_low,
        da1_scores=da1_scores, pc1_scores=pc1_scores,
        cos_da1_pc1=np.array(cos_da1_pc1),
    )
    print(f"Saved: {out_npz}")


# ============================================================
# Fig A7: Eigenvalue spectrum + MC null + free-conv density
# ============================================================

def make_spectrum_figure(args):
    """Eigenvalue spectrum (Fig A7). CPU-only (uses pre-computed data)."""
    from src.noise_floor.free_conv_utils import edges as fc_edges, density as fc_density, edge_tauinf
    from src.plotting.spike_fig_style import (
        HIST_EMPIRICAL, HIST_MC_NULL, FREECONV_CURVE, FLOOR_TAU, FLOOR_INF, MC_NULL_PCTL,
        HIST_ALPHA_EMP, HIST_ALPHA_NULL, CURVE_LW, VLINE_LW, VLINE_STYLE,
    )
    from src.plotting.style_config import setup_style
    setup_style()

    # Load empirical eigenvalues at τ*
    da_npz_path = os.path.join(RESULTS_DIR, f"clip_da_{args.name}.npz")
    da_npz = np.load(da_npz_path, allow_pickle=True)
    eigs_raw = da_npz['eigenvalues_full_at_tau_star']
    eigs = np.sort(eigs_raw)[::-1].copy()

    tau_star = 5.0  # dimensionless OU time

    # Load C(τ) data for d, N, gamma, sigma
    c_tau_path = os.path.join(RESULTS_DIR, "elderly_gs7p5_n200_C_tau_full.npz")
    c_data = np.load(c_tau_path)
    sigma_f_sq = float(c_data['sigma_f_sq'])
    d = int(c_data['d'])
    N = int(c_data['N'])
    gamma = d / N
    sigma = np.sqrt(sigma_f_sq)

    C0 = c_data['C0']
    print(f'd={d}, N={N}, gamma={gamma:.2f}, sigma_f^2={sigma_f_sq:.6e}')
    print(f'tau*={tau_star}, DA1={eigs[0]:.4e}')

    # Free-conv edges
    lm_tau, lp_tau = fc_edges(tau_star, sigma, gamma=gamma)
    _, lp_inf = edge_tauinf(sigma, gamma)
    print(f'lambda_+(tau*)={lp_tau:.4e}, lambda_+(inf)={lp_inf:.4e}')

    # MC null
    N_MC_REPS = 200
    print(f'Computing MC null ({N_MC_REPS} reps)...')
    rng = np.random.RandomState(42)

    eigvals_C0, eigvecs_C0 = np.linalg.eigh(C0)
    eigvals_C0 = np.maximum(eigvals_C0, 0)
    sqrt_C0 = eigvecs_C0 * np.sqrt(eigvals_C0)[np.newaxis, :]

    rho = np.exp(-tau_star / 2.0)
    noise_scale = np.sqrt(max(1.0 - np.exp(-tau_star), 1e-20))

    mc_all_eigs = np.empty((N_MC_REPS, d))
    mc_max_eigs = np.empty(N_MC_REPS)

    for i in range(N_MC_REPS):
        z0 = rng.randn(N, d) @ sqrt_C0.T
        z_noise = rng.randn(N, d) @ sqrt_C0.T
        x_tau = rho * z0 + noise_scale * z_noise
        mu = z0.mean(axis=0)
        z0_c = z0 - mu
        x_tau_c = x_tau - mu
        C_raw = (x_tau_c.T @ z0_c) / N
        C_sym = 0.5 * (C_raw + C_raw.T)
        e = np.linalg.eigvalsh(C_sym)
        mc_all_eigs[i] = e
        mc_max_eigs[i] = e[-1]
        if (i + 1) % 50 == 0:
            print(f'  MC rep {i+1}/{N_MC_REPS}')

    mc_95 = float(np.percentile(mc_max_eigs, 95))

    # Free-conv density curve
    pad = 0.3 * (lp_tau - lm_tau)
    lam_grid = np.linspace(lm_tau - pad, lp_tau + pad, 800)
    rho_grid = fc_density(lam_grid, tau_star, sigma, gamma=gamma)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    linthresh = 5e-4
    def symlog_fn(x):
        return np.sign(x) * np.log10(1 + np.abs(x) / linthresh)
    def inv_symlog_fn(y):
        return np.sign(y) * linthresh * (10**np.abs(y) - 1)

    eig_min = min(eigs.min(), mc_all_eigs.min())
    eig_max = max(eigs.max(), mc_all_eigs.max())
    sl_lo = symlog_fn(eig_min - 0.02 * (eig_max - eig_min))
    sl_hi = symlog_fn(eig_max + 0.02 * (eig_max - eig_min))
    bins_sl = inv_symlog_fn(np.linspace(sl_lo, sl_hi, 80))

    MC_NULL_DARK = '#B94055'
    mc_pooled = mc_all_eigs.ravel()
    ax.hist(mc_pooled, bins=bins_sl, density=True, color=MC_NULL_DARK,
            alpha=HIST_ALPHA_NULL, edgecolor='none', zorder=1, label='MC null')
    ax.hist(eigs, bins=bins_sl, density=True, color=HIST_EMPIRICAL,
            alpha=HIST_ALPHA_EMP, edgecolor='white', linewidth=0.3, zorder=2,
            label='Empirical')
    ax.plot(lam_grid, rho_grid, color=FREECONV_CURVE, lw=CURVE_LW, zorder=4)

    ax.axvline(lp_tau, color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW, zorder=5)
    ax.axvline(lp_inf, color=FLOOR_INF, ls=VLINE_STYLE, lw=VLINE_LW, zorder=5)
    ax.axvline(mc_95, color=MC_NULL_PCTL, ls=VLINE_STYLE, lw=VLINE_LW, zorder=5)

    # DA1 marker
    lam1 = eigs[0]
    ax.plot(lam1, ax.get_ylim()[1] * 0.05, marker='v', color=FREECONV_CURVE,
            markersize=5, zorder=6, clip_on=False)
    ax.annotate('DA1', xy=(lam1, 0), fontsize=7, color=FREECONV_CURVE,
                ha='center', va='bottom', fontweight='bold',
                xytext=(lam1, ax.get_ylim()[1] * 0.08))

    ax.set_xscale('symlog', linthresh=linthresh)
    ax.set_yscale('symlog', linthresh=100)
    ax.set_ylim(bottom=0)

    ATOM_COLOR = '#7B2D8E'
    ZERO_TOL = 1e-8
    atom_mass_pred = max(0.0, 1.0 - 2.0 / gamma)
    atom_mc = np.mean(np.abs(mc_all_eigs) < ZERO_TOL)
    atom_empirical = np.sum(np.abs(eigs) < ZERO_TOL) / len(eigs)
    box_text = (f'Atom at $\\lambda=0$\n'
                f'Predicted  $w_0 = {atom_mass_pred:.2f}$\n'
                f'MC null    $w_0 = {atom_mc:.2f}$\n'
                f'Empirical  $w_0 = {atom_empirical:.2f}$')
    ax.annotate(box_text,
                xy=(0, 0), xycoords='data',
                xytext=(0.72, 0.92), textcoords='axes fraction',
                fontsize=6.5, color=ATOM_COLOR, fontweight='bold',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#F3E5F5',
                          edgecolor=ATOM_COLOR, lw=1.2, alpha=0.95),
                arrowprops=dict(arrowstyle='->', color=ATOM_COLOR, lw=1.5,
                                connectionstyle='arc3,rad=0.3'),
                zorder=8)

    ax.set_xlabel(r'$\lambda$', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.tick_params(labelsize=7)

    legend_handles = [
        Patch(facecolor=HIST_EMPIRICAL, alpha=HIST_ALPHA_EMP, edgecolor='white',
              label='Empirical eigenvalues'),
        Patch(facecolor=MC_NULL_DARK, alpha=HIST_ALPHA_NULL, edgecolor='none',
              label='MC null eigenvalues'),
        Line2D([], [], color=FREECONV_CURVE, lw=CURVE_LW,
               label='Free-conv. density'),
        Line2D([], [], color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW,
               label=fr'$\lambda_+(\tau^*) = {lp_tau:.2e}$'),
        Line2D([], [], color=FLOOR_INF, ls=VLINE_STYLE, lw=VLINE_LW,
               label=fr'$\lambda_+(\infty) = {lp_inf:.2e}$'),
        Line2D([], [], color=MC_NULL_PCTL, ls=VLINE_STYLE, lw=VLINE_LW,
               label=f'MC null 95th pctl = {mc_95:.2e}'),
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc='upper left',
              frameon=True, fancybox=True, edgecolor='#cccccc')

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'fig_elderly_spectrum.png')
    fig.savefig(out_path, dpi=300)
    print(f'Saved: {out_path}')
    plt.close(fig)

    print(f'\n  DA1 = {lam1:.4e}, lambda_+(tau*) = {lp_tau:.4e}')
    print(f'  DA1 / MC_95 = {lam1 / mc_95:.1f}x')


# ============================================================
# Fig A8: DA1/DA2 projection histograms
# ============================================================

def make_histogram_figure(args):
    """DA1/DA2 projection histograms (Fig A8). CPU-only."""
    from src.plotting.style_config import setup_style
    setup_style()

    da_npz_path = os.path.join(RESULTS_DIR, f"clip_da_{args.name}.npz")
    da_npz = np.load(da_npz_path)

    if args.use_taustar:
        V = da_npz["eigvecs_full_topk_at_tau_star"]
        da1 = V[:, 0].astype(np.float64)
        da2 = V[:, 1].astype(np.float64)
    else:
        da1 = da_npz["da1_eigvec"]
        da2 = da_npz["da2_eigvec"]

    # Load CLIP embeddings
    clip_path = os.path.join(RESULTS_DIR, "gen_mp_p5_elderly_gs7p5_quick_clip.npz")
    clip_data = np.load(clip_path)
    z0_clip = clip_data["z0_clip"] if "z0_clip" in clip_data else clip_data["projections"]
    N = z0_clip.shape[0]

    mu = z0_clip.mean(axis=0)
    z0c = z0_clip - mu

    da1_scores = z0c @ da1
    da2_scores = z0c @ da2

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    axes[0].hist(da1_scores, bins=30, color='steelblue', edgecolor='white',
                 linewidth=0.5, alpha=0.85)
    axes[0].set_xlabel('DA1 score', fontsize=10)
    axes[0].set_ylabel('Count', fontsize=10)
    axes[0].set_title('DA1 projection (bimodal)', fontsize=10)

    axes[1].hist(da2_scores, bins=30, color='coral', edgecolor='white',
                 linewidth=0.5, alpha=0.85)
    axes[1].set_xlabel('DA2 score', fontsize=10)
    axes[1].set_ylabel('Count', fontsize=10)
    axes[1].set_title('DA2 projection (unimodal)', fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'fig_elderly_da_histograms.png')
    fig.savefig(out_path, dpi=300)
    print(f'Saved: {out_path}')
    plt.close(fig)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Elderly DA figures: grid (Fig 4), spectrum (Fig A7), histograms (Fig A8)")
    parser.add_argument("--mode", choices=["grid", "spectrum", "histogram", "all"],
                        default="grid")
    parser.add_argument("--name", default="elderly_gs7p5_fc_dt0p0025",
                        help="basename of clip_da_<name>.{json,npz}")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--k", type=int, default=5, help="Number of extreme images per row")
    parser.add_argument("--use_taustar", action="store_true",
                        help="Use eigvecs from C(tau*_freeconv) instead of peak-gap-lag")
    args = parser.parse_args()

    if args.mode in ("grid", "all"):
        make_grid_figure(args)
    if args.mode in ("spectrum", "all"):
        make_spectrum_figure(args)
    if args.mode in ("histogram", "all"):
        make_histogram_figure(args)


if __name__ == "__main__":
    main()
