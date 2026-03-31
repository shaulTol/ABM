"""
Train class-conditional DDPM on MNIST 28x28.

The UNet receives digit labels as class_labels via num_class_embeds=10.
This enables per-digit score extraction for SSA and DA computation.

Usage:
    accelerate launch scripts/train_mnist_ddpm.py --config configs/mnist.yaml
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image

import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.training_utils import EMAModel

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train class-conditional DDPM on MNIST")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, default=None, help="Override MNIST data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint or 'latest'")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def build_model(config):
    """Build UNet2DModel with class conditioning from config."""
    model_cfg = config["model"]
    base_ch = model_cfg["base_channels"]
    mults = model_cfg["channel_mult"]
    block_out_channels = tuple(base_ch * m for m in mults)
    attn_res = model_cfg.get("attention_resolution", 7)
    image_size = config["data"]["resolution"]  # 28

    n_levels = len(mults)
    down_block_types = []
    up_block_types = []
    for i in range(n_levels):
        res_at_level = image_size // (2 ** (i + 1))
        if res_at_level == attn_res:
            down_block_types.append("AttnDownBlock2D")
            up_block_types.append("AttnUpBlock2D")
        else:
            down_block_types.append("DownBlock2D")
            up_block_types.append("UpBlock2D")
    up_block_types = list(reversed(up_block_types))

    model = UNet2DModel(
        sample_size=image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=model_cfg.get("layers_per_block", 2),
        block_out_channels=block_out_channels,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        dropout=model_cfg.get("dropout", 0.0),
        norm_num_groups=model_cfg.get("norm_num_groups", 32),
        num_class_embeds=model_cfg.get("num_class_embeds", 10),
    )
    return model


def build_dataset(config, data_dir_override=None):
    """Build MNIST dataset."""
    data_dir = data_dir_override or "data"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
    ])
    dataset = MNIST(root=data_dir, train=True, download=True, transform=transform)
    return dataset


@torch.no_grad()
def generate_class_conditional_samples(model, scheduler, device, n_per_class=2, seed=42):
    """Manual DDPM reverse process with class labels.

    DDPMPipeline doesn't support class_labels in diffusers 0.30.0,
    so we implement the reverse process manually.

    Returns:
        images: (10 * n_per_class, 1, 28, 28) tensor in [-1, 1]
        labels: (10 * n_per_class,) tensor of digit labels
    """
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)

    all_labels = torch.arange(10, device=device).repeat_interleave(n_per_class)
    n_total = len(all_labels)

    # Start from pure noise
    x = torch.randn(n_total, 1, 28, 28, device=device, generator=generator)

    scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    for t in scheduler.timesteps:
        t_batch = t.expand(n_total).to(device)
        eps_pred = model(x, t_batch, class_labels=all_labels, return_dict=False)[0]
        x = scheduler.step(eps_pred, t, x, generator=generator).prev_sample

    return x, all_labels


def main():
    args = parse_args()
    config = load_config(args.config)

    output_dir = args.output_dir or config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=config["training"].get("mixed_precision", "fp16"),
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        log_with="wandb" if config["training"].get("use_wandb", True) else None,
        project_dir=output_dir,
    )

    model = build_model(config)
    n_params = count_parameters(model)
    if accelerator.is_main_process:
        logger.info(f"Model: {config['model']['name']}")
        logger.info(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
        logger.info(f"Block channels: {tuple(config['model']['base_channels'] * m for m in config['model']['channel_mult'])}")
        logger.info(f"Class conditioning: {config['model'].get('num_class_embeds', 'None')}")

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["diffusion"]["num_timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        beta_schedule=config["diffusion"]["beta_schedule"],
        prediction_type="epsilon",
    )

    train_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(train_cfg.get("adam_beta1", 0.9), train_cfg.get("adam_beta2", 0.999)),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        eps=train_cfg.get("adam_epsilon", 1e-8),
    )

    dataset = build_dataset(config, args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset size: {len(dataset)}")

    ema_model = None
    if train_cfg.get("use_ema", True):
        ema_model = EMAModel(
            model.parameters(),
            decay=train_cfg.get("ema_decay", 0.9999),
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if ema_model is not None:
        ema_model.to(accelerator.device)

    if accelerator.is_main_process and config["training"].get("use_wandb", True):
        accelerator.init_trackers(
            project_name=config["training"].get("wandb_project", "ddpm-mnist-ssa"),
            config={
                "model_name": config["model"]["name"],
                "n_params": n_params,
                "base_channels": config["model"]["base_channels"],
                "channel_mult": config["model"]["channel_mult"],
                **train_cfg,
            },
        )

    total_steps = train_cfg["total_steps"]
    checkpoint_every = train_cfg.get("checkpoint_every", 25000)
    log_every = train_cfg.get("log_every", 100)
    sample_every = train_cfg.get("sample_every", 5000)

    global_step = 0
    start_epoch = 0

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            dirs = sorted(
                [d for d in Path(output_dir).iterdir()
                 if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda d: int(d.name.split("-")[1]),
            )
            if dirs:
                args.resume_from_checkpoint = str(dirs[-1])
            else:
                args.resume_from_checkpoint = None

        if args.resume_from_checkpoint:
            accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(Path(args.resume_from_checkpoint).name.split("-")[1])
            start_epoch = global_step // len(dataloader)
            if accelerator.is_main_process:
                logger.info(f"Resumed from checkpoint at step {global_step}")

    if accelerator.is_main_process:
        logger.info(f"Starting training for {total_steps} steps")

    model.train()
    epoch = start_epoch
    while global_step < total_steps:
        for batch in dataloader:
            if global_step >= total_steps:
                break

            clean_images, labels = batch

            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,),
                device=clean_images.device,
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(
                    noisy_images, timesteps,
                    class_labels=labels,
                    return_dict=False,
                )[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if ema_model is not None:
                ema_model.step(model.parameters())

            global_step += 1

            if global_step % log_every == 0 and accelerator.is_main_process:
                logs = {"loss": loss.detach().item(), "step": global_step, "epoch": epoch}
                accelerator.log(logs, step=global_step)
                logger.info(f"Step {global_step}/{total_steps} | Loss: {loss.item():.4f}")

            if global_step % sample_every == 0 and accelerator.is_main_process:
                save_samples(model, ema_model, noise_scheduler, output_dir,
                             global_step, accelerator)

            if global_step % checkpoint_every == 0:
                if accelerator.is_main_process:
                    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(ckpt_dir)
                    logger.info(f"Saved checkpoint at step {global_step}")

        epoch += 1

    # Final save
    if accelerator.is_main_process:
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(ckpt_dir)

        # Save EMA weights as standalone model
        if ema_model is not None:
            ema_model.copy_to(model.parameters())
        unwrapped = accelerator.unwrap_model(model)
        save_dir = os.path.join(output_dir, "model")
        os.makedirs(save_dir, exist_ok=True)
        unwrapped.save_pretrained(save_dir)
        noise_scheduler.save_pretrained(save_dir)
        logger.info(f"Training complete. Model saved to {save_dir}")

    accelerator.end_training()


@torch.no_grad()
def save_samples(model, ema_model, noise_scheduler, output_dir, step, accelerator):
    """Generate and save a grid of class-conditional samples."""
    unwrapped_model = accelerator.unwrap_model(model)

    if ema_model is not None:
        ema_model.store(unwrapped_model.parameters())
        ema_model.copy_to(unwrapped_model.parameters())

    images, labels = generate_class_conditional_samples(
        unwrapped_model, noise_scheduler, accelerator.device, n_per_class=2, seed=step,
    )

    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Rescale [-1,1] -> [0,1] for saving
    grid = make_grid((images + 1) / 2, nrow=10, padding=2)
    save_image(grid, os.path.join(samples_dir, f"samples_{step:07d}.png"))

    if ema_model is not None:
        ema_model.restore(unwrapped_model.parameters())


if __name__ == "__main__":
    main()
