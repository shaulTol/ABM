"""
sdxl_components.py — Reusable components for SDXL SSA/DA computation.

Provides:
  - LatentCLIPEmbedder: (N, 16384) → (N, 512)  via Latent-CLIP (no pixel decode)
  - RealCLIPEmbedder:   (N, 16384) → (N, 768)  via VAE decode + CLIP ViT-L/14
  - compute_ssa_da:     Space-agnostic SSA + DA from pre-computed embeddings
  - TextProbeAnalyzer:  CLIP text probes for interpreting directions
  - Visualization utilities: decode, grid, plots
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.metrics.autocorrelation import aggregate_ssa


def _load_free_conv_edges():
    """Lazy-load the free-conv upper-edge function."""
    import importlib.util as _ilu
    repo_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    fc_path = os.path.join(repo_root, 'experiments', 'noise_floor',
                           'free_conv_utils.py')
    spec = _ilu.spec_from_file_location('free_conv_utils', fc_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.edges


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VAE_SCALE = 0.13025  # SDXL-specific latent scaling factor

DEFAULT_PROBE_PAIRS = [
    ("gender", "a photograph of a woman", "a photograph of a man"),
    ("age", "a photograph of a young person", "a photograph of an old person"),
    ("realism", "a realistic photograph", "a painting"),
    ("brightness", "a bright image", "a dark image"),
    ("framing", "a close-up portrait", "a wide-angle photograph"),
    ("complexity", "a simple image", "a complex detailed image"),
    ("expression", "a smiling person", "a serious person"),
    ("glasses", "an animal wearing sunglasses", "an animal without sunglasses"),
]


# ---------------------------------------------------------------------------
# LatentCLIPEmbedder (wraps existing pattern from compute_sdxl_ssa_v2.py)
# ---------------------------------------------------------------------------

class LatentCLIPEmbedder:
    """Maps VAE latents (4x64x64) to Latent-CLIP space (512D) without pixel decoding."""

    def __init__(self, device, checkpoint_path=None):
        import latent_clip
        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                "wendlerc/latent-clip-b-8-512-34b-80k", "checkpoints/epoch_34.pt"
            )
        self.model, _, _ = latent_clip.create_model_and_transforms(
            "Latent-ViT-B-8-512", pretrained=checkpoint_path, device=device,
        )
        self.model.eval()
        self.device = device
        self._embed_dim = 512
        print(f"  LatentCLIPEmbedder loaded: embed_dim={self._embed_dim}")

    @property
    def embed_dim(self):
        return self._embed_dim

    @torch.no_grad()
    def embed(self, latents_flat, latent_shape=(4, 64, 64), batch_size=512):
        """(N, 16384) → (N, 512) L2-normalized embeddings. Returns numpy."""
        N = latents_flat.shape[0]
        if isinstance(latents_flat, np.ndarray):
            latents_flat = torch.from_numpy(latents_flat).float()
        embeddings = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = latents_flat[start:end].to(self.device)
            z = batch.view(-1, *latent_shape) / VAE_SCALE
            emb = self.model.encode_image(z)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())
        return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# RealCLIPEmbedder (new: VAE decode + CLIP ViT-L/14)
# ---------------------------------------------------------------------------

class RealCLIPEmbedder:
    """VAE decode + CLIP image encode. (N, 16384) → (N, 768) L2-normalized."""

    def __init__(self, vae, clip_model_name="openai/clip-vit-large-patch14",
                 device="cuda", vae_batch_size=8, clip_batch_size=16):
        from transformers import CLIPModel, CLIPProcessor
        self.vae = vae
        self.device = device
        self.vae_batch_size = vae_batch_size
        self.clip_batch_size = clip_batch_size

        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self._embed_dim = self.clip_model.config.projection_dim  # 768 for ViT-L/14
        print(f"  RealCLIPEmbedder loaded: {clip_model_name}, embed_dim={self._embed_dim}")

    @property
    def embed_dim(self):
        return self._embed_dim

    @torch.no_grad()
    def embed(self, latents_flat, latent_shape=(4, 64, 64), batch_size=None):
        """(N, D) → (N, 768) L2-normalized CLIP embeddings. Returns numpy."""
        if isinstance(latents_flat, np.ndarray):
            latents_flat = torch.from_numpy(latents_flat).float()
        N = latents_flat.shape[0]
        D = latents_flat.shape[1]
        expected_D = int(np.prod(latent_shape))
        if D != expected_D:
            raise ValueError(
                f"RealCLIPEmbedder: latent dim {D} != prod(latent_shape) {expected_D}. "
                f"latent_shape={latent_shape}. For 1024x1024 images use (4,128,128)."
            )
        all_embs = []

        for start in range(0, N, self.vae_batch_size):
            end = min(start + self.vae_batch_size, N)
            batch = latents_flat[start:end].to(self.device).float()
            z = batch.view(-1, *latent_shape) / VAE_SCALE

            # VAE decode → pixels in [0, 1]
            decoded = self.vae.decode(z, return_dict=False)[0]
            pixels = ((decoded + 1.0) / 2.0).clamp(0, 1)  # (B, 3, H, W)

            # Convert to PIL for CLIPProcessor
            pil_images = []
            pixel_bytes = (pixels * 255).byte().cpu()
            for img_t in pixel_bytes:
                pil_images.append(Image.fromarray(img_t.permute(1, 2, 0).numpy()))

            # CLIP encode in sub-batches
            for clip_start in range(0, len(pil_images), self.clip_batch_size):
                clip_end = min(clip_start + self.clip_batch_size, len(pil_images))
                inputs = self.clip_processor(
                    images=pil_images[clip_start:clip_end],
                    return_tensors="pt",
                ).to(self.device)
                emb = self.clip_model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                all_embs.append(emb.cpu().numpy())

        return np.concatenate(all_embs, axis=0)


# ---------------------------------------------------------------------------
# compute_ssa_da — Space-agnostic SSA + DA computation
# ---------------------------------------------------------------------------

def compute_ssa_da(z0, z_snapshots, lags, n_eigvals=10, n_eigvecs=3,
                   ssa_min_lag=0, ssa_only=False, dt=None,
                   freeconv_stop_M=None):
    """Compute SSA and DA from pre-computed embeddings (any dimension).

    Args:
        z0: (N, D) numpy array — embeddings of initial samples
        z_snapshots: dict {lag: (N, D) numpy array} — embeddings at each lag
        lags: sorted list of lag values
        n_eigvals: number of top eigenvalues to keep per lag
        n_eigvecs: number of top eigenvectors to keep per lag
        ssa_min_lag: only include lags >= this value when computing SSA mean
        ssa_only: if True, skip DA eigendecomposition (only compute SSA)
        freeconv_stop_M: if int, apply free-conv stopping criterion: at each
            lag compute the upper edge lambda_plus of the bulk (using
            sigma_f^2 = mean per-dim variance of z0 and gamma=D/N) and stop
            when lambda_2(C(tau)) < lambda_plus(tau) for M consecutive lags.
            tau_star = lag just before the first crossing. Saves the full
            symmetric autocovariance matrix C(tau_star) and full eigenspectrum
            in the result for later eigenvalue-distribution analysis.

    Returns:
        dict with: mu, C0_trace, rho_sq (array), ssa (scalar),
        eigenvalues/eigenvectors per lag, gap profiles, DA1/DA2 optimal info
        (DA fields are None when ssa_only=True). When freeconv_stop_M is set
        and ssa_only=False, also includes:
          tau_star_freeconv, C_tau_star (D,D), eigenvalues_full (D,),
          eigvecs_full_topk (D, k_full=20), freeconv_lambda_plus_profile
    """
    N, D = z0.shape
    mu = z0.mean(axis=0)
    z0_c = z0 - mu

    # tr(C(0))
    tr_c0 = (z0_c ** 2).sum(axis=-1).mean()
    print(f"  tr(C(0)) = {tr_c0:.6f}, D = {D}")
    if tr_c0 < 1e-12:
        print("  WARNING: tr(C(0)) near zero")
        return {"ssa": 0.0, "rho_sq": np.zeros(len(lags))}

    rho_sq_list = []
    eigenvalues_list = []
    gap12_list = []
    gap23_list = []
    top_eigvecs_list = []

    used_lags = [l for l in lags if l in z_snapshots]

    # Free-conv stopping prep
    fc_active = (freeconv_stop_M is not None) and (not ssa_only)
    if fc_active:
        # CLIP-space sigma_f: mean per-dim variance of z0 (centered).
        sigma_f_sq = float((z0_c ** 2).sum() / (N * D))
        sigma_f = float(np.sqrt(sigma_f_sq))
        gamma = D / N
        free_conv_edges = _load_free_conv_edges()
        print(f"  Free-conv stopping: M={freeconv_stop_M}, sigma_f^2={sigma_f_sq:.3e}, gamma={gamma:.3f}")
        fc_lambda_plus_list = []
        fc_lambda2_list = []
        fc_consec = 0
        fc_first_cross_idx = None  # index into used_lags where first crossing occurs
        fc_C_tau_star = None
        fc_eigvals_full = None
        fc_eigvecs_topk_full = None
        fc_topk_full = 20

    for i, lag in enumerate(used_lags):
        z_tau = z_snapshots[lag]
        z_tau_c = z_tau - mu

        # tr(C(tau)) for SSA
        tr_c_tau = (z0_c * z_tau_c).sum(axis=-1).mean()
        rho = tr_c_tau / tr_c0
        rho_sq_list.append(rho ** 2)

        if not ssa_only:
            # Full C(tau) for DA eigdecomp
            z0_t = torch.from_numpy(z0_c).float()
            z_tau_t = torch.from_numpy(z_tau_c).float()
            C_tau = (z_tau_t.T @ z0_t) / N
            C_tau = 0.5 * (C_tau + C_tau.T)

            eigvals, eigvecs = torch.linalg.eigh(C_tau)
            idx = eigvals.argsort(descending=True)
            eigvals_full_arr = eigvals[idx].numpy()
            eigvecs_full_arr = eigvecs[:, idx].numpy()

            eigenvalues_list.append(eigvals_full_arr[:n_eigvals])
            top_eigvecs_list.append(eigvecs_full_arr[:, :n_eigvecs])

            abs_eig = np.abs(eigvals_full_arr)
            g12 = abs_eig[0] / max(abs_eig[1], 1e-12)
            g23 = abs_eig[1] / max(abs_eig[2], 1e-12)
            gap12_list.append(g12)
            gap23_list.append(g23)

            if fc_active:
                tau_phys = lag * dt if dt is not None else lag
                _, lam_plus = free_conv_edges(tau_phys, sigma_f, gamma=gamma)
                lam2 = float(abs_eig[1])
                fc_lambda_plus_list.append(float(lam_plus))
                fc_lambda2_list.append(lam2)
                if lam2 < lam_plus:
                    fc_consec += 1
                    if fc_consec >= freeconv_stop_M and fc_first_cross_idx is None:
                        fc_first_cross_idx = i
                        # tau_star = lag just before first crossing (i.e. previous index)
                        prev_i = max(i - freeconv_stop_M, 0)
                        # Snapshot full C(tau*), full eigvals, top-k eigvecs from prev iteration:
                        # we already overwrote those tensors. Recompute at prev lag.
                        prev_lag = used_lags[prev_i]
                        z_tau_prev = z_snapshots[prev_lag] - mu
                        z_prev_t = torch.from_numpy(z_tau_prev).float()
                        C_prev = (z_prev_t.T @ z0_t) / N
                        C_prev = 0.5 * (C_prev + C_prev.T)
                        ev_prev, evec_prev = torch.linalg.eigh(C_prev)
                        idx_prev = ev_prev.argsort(descending=True)
                        fc_C_tau_star = C_prev.numpy().astype(np.float32)
                        fc_eigvals_full = ev_prev[idx_prev].numpy().astype(np.float32)
                        fc_eigvecs_topk_full = evec_prev[:, idx_prev[:fc_topk_full]].numpy().astype(np.float32)
                        print(f"  Free-conv stop hit: lag={lag} (lambda2={lam2:.4e} < lambda+={lam_plus:.4e}), "
                              f"tau_star_lag={prev_lag}")
                else:
                    fc_consec = 0

    rho_sq = np.array(rho_sq_list)

    # SSA: Ŝ = ∫ ρ²(τ) dτ (Riemann sum in physical time, truncated at noise floor)
    used_lags_arr = np.array(used_lags, dtype=float)
    mask = used_lags_arr >= ssa_min_lag

    # Convert step-lags to physical time for correct integration
    phys_lags = used_lags_arr * dt if dt is not None else used_lags_arr

    if mask.any():
        ssa = aggregate_ssa(rho_sq[mask], lags=phys_lags[mask])
    else:
        ssa = aggregate_ssa(rho_sq, lags=phys_lags)
    ssa_all = aggregate_ssa(rho_sq, lags=phys_lags)  # unfiltered for reference

    result = {
        "ssa": ssa,
        "ssa_all": ssa_all,
        "ssa_min_lag": ssa_min_lag,
        "rho_sq": rho_sq,
        "snapshot_lags": used_lags,
        "mu": mu,
        "C0_trace": float(tr_c0),
        "embed_dim": D,
    }

    print(f"  SSA = {ssa:.6f}" + (f" (lags >= {ssa_min_lag})" if ssa_min_lag > 0 else ""))
    if ssa_min_lag > 0:
        print(f"  SSA_all = {ssa_all:.6f} (all lags)")

    if ssa_only:
        # Fill DA fields with None
        result.update({
            "eigenvalues": None, "gap12": None, "gap23": None,
            "top_eigvecs": None,
            "da1_optimal_lag": None, "da1_gap": None, "da1_eigvec": None,
            "da2_optimal_lag": None, "da2_gap": None, "da2_eigvec": None,
        })
    else:
        gap12 = np.array(gap12_list)
        gap23 = np.array(gap23_list)
        opt1_idx = int(np.argmax(gap12))
        opt2_idx = int(np.argmax(gap23))

        result.update({
            "eigenvalues": eigenvalues_list,
            "gap12": gap12,
            "gap23": gap23,
            "top_eigvecs": top_eigvecs_list,
            "da1_optimal_lag": used_lags[opt1_idx],
            "da1_gap": float(gap12[opt1_idx]),
            "da1_eigvec": top_eigvecs_list[opt1_idx][:, 0],
            "da2_optimal_lag": used_lags[opt2_idx],
            "da2_gap": float(gap23[opt2_idx]),
            "da2_eigvec": top_eigvecs_list[opt2_idx][:, 1],
        })
        print(f"  DA1: gap={gap12[opt1_idx]:.2f}x at lag={used_lags[opt1_idx]}")
        print(f"  DA2: gap={gap23[opt2_idx]:.2f}x at lag={used_lags[opt2_idx]}")

        if fc_active:
            # If the criterion never triggered, fall back to tau* = last lag
            # and compute its full spectrum + C(tau*) for completeness.
            if fc_first_cross_idx is None:
                last_i = len(used_lags) - 1
                last_lag = used_lags[last_i]
                z_last = z_snapshots[last_lag] - mu
                z_last_t = torch.from_numpy(z_last).float()
                z0_t_local = torch.from_numpy(z0_c).float()
                C_last = (z_last_t.T @ z0_t_local) / N
                C_last = 0.5 * (C_last + C_last.T)
                ev_l, evec_l = torch.linalg.eigh(C_last)
                idx_l = ev_l.argsort(descending=True)
                fc_C_tau_star = C_last.numpy().astype(np.float32)
                fc_eigvals_full = ev_l[idx_l].numpy().astype(np.float32)
                fc_eigvecs_topk_full = evec_l[:, idx_l[:fc_topk_full]].numpy().astype(np.float32)
                tau_star_lag = last_lag
                print(f"  Free-conv stop NEVER hit; fallback tau_star=T={last_lag}")
            else:
                tau_star_lag = used_lags[max(fc_first_cross_idx - freeconv_stop_M, 0)]

            result.update({
                "tau_star_freeconv": int(tau_star_lag),
                "freeconv_M": int(freeconv_stop_M),
                "freeconv_lambda_plus_profile": np.array(fc_lambda_plus_list),
                "freeconv_lambda2_profile": np.array(fc_lambda2_list),
                "freeconv_sigma_f_sq": float(sigma_f_sq),
                "freeconv_gamma": float(gamma),
                "C_tau_star": fc_C_tau_star,
                "eigenvalues_full_at_tau_star": fc_eigvals_full,
                "eigvecs_full_topk_at_tau_star": fc_eigvecs_topk_full,
                "freeconv_first_cross_idx": (-1 if fc_first_cross_idx is None
                                             else int(fc_first_cross_idx)),
            })
            print(f"  Free-conv: tau_star_lag={tau_star_lag}, "
                  f"sigma_f^2={sigma_f_sq:.3e}, saved C(tau*) ({D}x{D}) "
                  f"+ full {D} eigenvalues")

    return result


# ---------------------------------------------------------------------------
# TextProbeAnalyzer — CLIP text-based semantic probes
# ---------------------------------------------------------------------------

class TextProbeAnalyzer:
    """Compute CLIP text-direction probes for interpreting embedding directions."""

    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", device="cuda"):
        from transformers import CLIPModel, CLIPTokenizer
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_model.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.device = device
        self.probe_axes = {}  # name → (D,) numpy
        self.probe_names = []

    @torch.no_grad()
    def _encode_text(self, text):
        """Encode a single text string → (D,) L2-normalized numpy."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True).to(self.device)
        emb = self.clip_model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().squeeze(0)

    def define_probes(self, probe_triplets=None):
        """Define probe axes from (name, text_pos, text_neg) triplets.

        Each axis = normalize(encode(text_pos) - encode(text_neg)).
        """
        if probe_triplets is None:
            probe_triplets = DEFAULT_PROBE_PAIRS
        self.probe_axes = {}
        self.probe_names = []
        for name, text_pos, text_neg in probe_triplets:
            emb_pos = self._encode_text(text_pos)
            emb_neg = self._encode_text(text_neg)
            axis = emb_pos - emb_neg
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            self.probe_axes[name] = axis
            self.probe_names.append(name)
        print(f"  Defined {len(self.probe_axes)} text probes: {self.probe_names}")

    def score_samples(self, clip_embeddings):
        """Project (N, D) embeddings onto each probe axis → {name: (N,) scores}."""
        results = {}
        for name, axis in self.probe_axes.items():
            results[name] = clip_embeddings @ axis
        return results

    def score_direction(self, direction):
        """Cosine similarity of a D-dimensional direction with each probe axis.

        Args:
            direction: (D,) numpy array (need not be normalized)

        Returns:
            dict {probe_name: cosine_similarity}
        """
        d_norm = direction / (np.linalg.norm(direction) + 1e-12)
        results = {}
        for name, axis in self.probe_axes.items():
            results[name] = float(np.dot(d_norm, axis))
        return results

    def correlate_with_ranking(self, clip_embeddings, ranking):
        """Spearman correlation of each probe score with an external ranking.

        Args:
            clip_embeddings: (N, D) numpy array
            ranking: (N,) numpy array of scores to correlate with

        Returns:
            dict {probe_name: {"rho": float, "pvalue": float}}
        """
        from scipy.stats import spearmanr
        scores = self.score_samples(clip_embeddings)
        results = {}
        for name, probe_scores in scores.items():
            rho, pval = spearmanr(probe_scores, ranking)
            results[name] = {"rho": float(rho), "pvalue": float(pval)}
        return results


# ---------------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------------

def decode_latents_to_pil(vae, latents_flat, latent_shape=(4, 64, 64),
                          batch_size=8, device="cuda"):
    """Decode flat latent vectors to PIL images via SDXL VAE.

    Args:
        vae: SDXL VAE decoder
        latents_flat: (N, d) numpy or tensor, in pipeline scale
        latent_shape: (4, H//8, W//8)
        batch_size: decode in batches
        device: torch device

    Returns:
        list of PIL.Image
    """
    if isinstance(latents_flat, np.ndarray):
        latents_flat = torch.from_numpy(latents_flat).float()

    images = []
    for start in range(0, len(latents_flat), batch_size):
        end = min(start + batch_size, len(latents_flat))
        batch = latents_flat[start:end].to(device).float()
        z = batch.view(-1, *latent_shape) / VAE_SCALE

        with torch.no_grad():
            decoded = vae.decode(z, return_dict=False)[0]

        decoded = ((decoded + 1.0) / 2.0).clamp(0, 1)
        decoded = (decoded * 255).byte().cpu()

        for img_tensor in decoded:
            img = img_tensor.permute(1, 2, 0).numpy()
            images.append(Image.fromarray(img))

    return images


def make_grid_png(images, n_cols, path):
    """Save list of PIL images as a grid PNG."""
    if not images:
        return
    w, h = images[0].size
    n_rows = (len(images) + n_cols - 1) // n_cols
    grid = Image.new("RGB", (n_cols * w, n_rows * h), color=(255, 255, 255))
    for i, img in enumerate(images):
        row, col = divmod(i, n_cols)
        grid.paste(img, (col * w, row * h))
    grid.save(path)
    print(f"  Saved grid: {path}")


def make_topbottom_grid(vae, latents_flat, scores, n=8, path="grid.png",
                        latent_shape=(4, 64, 64), device="cuda"):
    """Decode top-n and bottom-n by score, save as 2-row grid with separator.

    Args:
        vae: SDXL VAE
        latents_flat: (N, d) numpy or tensor
        scores: (N,) numpy array
        n: number of top/bottom images
        path: output path
    """
    if isinstance(latents_flat, np.ndarray):
        latents_flat = torch.from_numpy(latents_flat).float()

    sorted_idx = np.argsort(scores)
    top_idx = sorted_idx[-n:][::-1].copy()
    bot_idx = sorted_idx[:n].copy()

    top_images = decode_latents_to_pil(vae, latents_flat[top_idx],
                                       latent_shape=latent_shape, device=device)
    bot_images = decode_latents_to_pil(vae, latents_flat[bot_idx],
                                       latent_shape=latent_shape, device=device)

    w, h = top_images[0].size
    grid = Image.new("RGB", (n * w, 3 * h), color=(255, 255, 255))

    for i, img in enumerate(top_images):
        grid.paste(img, (i * w, 0))
    for i, img in enumerate(bot_images):
        grid.paste(img, (i * w, 2 * h))

    grid.save(path)
    print(f"  Saved top/bottom grid: {path}")


def plot_rho_decay(rho_sq, lags, label="", ax=None):
    """Plot rho(tau)^2 decay curve."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(lags, rho_sq, "o-", label=label or "rho^2(tau)")
    ax.set_xlabel("Lag (Langevin steps)")
    ax.set_ylabel("rho(tau)^2")
    ax.set_title("Autocorrelation Decay")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_gap_profile(gap12, gap23, lags, ax=None):
    """Plot eigenvalue gap profiles (DA1/DA2 selection criterion)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(lags, gap12, "o-", label="lambda1/lambda2 (DA1)")
    ax.plot(lags, gap23, "s-", label="lambda2/lambda3 (DA2)")
    ax.set_xlabel("Lag (Langevin steps)")
    ax.set_ylabel("Eigenvalue ratio")
    ax.set_title("DA Gap Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
