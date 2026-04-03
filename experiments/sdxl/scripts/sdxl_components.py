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
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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
        """(N, 16384) → (N, 768) L2-normalized CLIP embeddings. Returns numpy."""
        if isinstance(latents_flat, np.ndarray):
            latents_flat = torch.from_numpy(latents_flat).float()
        N = latents_flat.shape[0]
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

def compute_ssa_da(z0, z_snapshots, lags, n_eigvals=10, n_eigvecs=3):
    """Compute SSA and DA from pre-computed embeddings (any dimension).

    Args:
        z0: (N, D) numpy array — embeddings of initial samples
        z_snapshots: dict {lag: (N, D) numpy array} — embeddings at each lag
        lags: sorted list of lag values
        n_eigvals: number of top eigenvalues to keep per lag
        n_eigvecs: number of top eigenvectors to keep per lag

    Returns:
        dict with: mu, C0_trace, rho_sq (array), ssa (scalar),
        eigenvalues/eigenvectors per lag, gap profiles, DA1/DA2 optimal info
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

    for lag in lags:
        if lag not in z_snapshots:
            continue
        z_tau = z_snapshots[lag]
        z_tau_c = z_tau - mu

        # tr(C(tau)) for SSA
        tr_c_tau = (z0_c * z_tau_c).sum(axis=-1).mean()
        rho = tr_c_tau / tr_c0
        rho_sq_list.append(rho ** 2)

        # Full C(tau) for DA eigdecomp
        z0_t = torch.from_numpy(z0_c).float()
        z_tau_t = torch.from_numpy(z_tau_c).float()
        C_tau = (z_tau_t.T @ z0_t) / N
        C_tau = 0.5 * (C_tau + C_tau.T)

        eigvals, eigvecs = torch.linalg.eigh(C_tau)
        idx = eigvals.abs().argsort(descending=True)
        eigvals = eigvals[idx].numpy()
        eigvecs = eigvecs[:, idx].numpy()

        eigenvalues_list.append(eigvals[:n_eigvals])
        top_eigvecs_list.append(eigvecs[:, :n_eigvecs])

        abs_eig = np.abs(eigvals)
        g12 = abs_eig[0] / max(abs_eig[1], 1e-12)
        g23 = abs_eig[1] / max(abs_eig[2], 1e-12)
        gap12_list.append(g12)
        gap23_list.append(g23)

    rho_sq = np.array(rho_sq_list)
    ssa = float(np.mean(rho_sq))
    gap12 = np.array(gap12_list)
    gap23 = np.array(gap23_list)

    used_lags = [l for l in lags if l in z_snapshots]

    opt1_idx = int(np.argmax(gap12))
    opt2_idx = int(np.argmax(gap23))

    result = {
        "ssa": ssa,
        "rho_sq": rho_sq,
        "snapshot_lags": used_lags,
        "eigenvalues": eigenvalues_list,
        "gap12": gap12,
        "gap23": gap23,
        "top_eigvecs": top_eigvecs_list,
        "mu": mu,
        "C0_trace": float(tr_c0),
        "embed_dim": D,
        # DA1
        "da1_optimal_lag": used_lags[opt1_idx],
        "da1_gap": float(gap12[opt1_idx]),
        "da1_eigvec": top_eigvecs_list[opt1_idx][:, 0],
        # DA2
        "da2_optimal_lag": used_lags[opt2_idx],
        "da2_gap": float(gap23[opt2_idx]),
        "da2_eigvec": top_eigvecs_list[opt2_idx][:, 1],
    }

    print(f"  SSA = {ssa:.6f}")
    print(f"  DA1: gap={gap12[opt1_idx]:.2f}x at lag={used_lags[opt1_idx]}")
    print(f"  DA2: gap={gap23[opt2_idx]:.2f}x at lag={used_lags[opt2_idx]}")

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
    top_idx = sorted_idx[-n:][::-1]
    bot_idx = sorted_idx[:n]

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
