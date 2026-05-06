#!/bin/bash
#SBATCH --job-name=T5k_resume
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-21
#SBATCH --output=/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/results_T5000_uncomplete/T5k_resume_%a_%j.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/results_T5000_uncomplete/T5k_resume_%a_%j.err

# Resume from saved fp16 Langevin checkpoints in results_T5000/.
# Skip generation + Langevin (already done). Run CLIP encode + DA + entropy
# + JSON output. JSONs go to a NEW dir results_T5000_uncomplete/ to avoid
# any collision with the existing 8-prompt (T=2000) analysis JSONs in
# results_guidance_sweep/ and results_multiprompt_sweep/.
#
# Coverage: 22 of 24 cells. Missing: elderly × {gs=7.5, 15.0} × dt=0.0025
# (those 2 stragglers got scancelled — slow node, would have hit SLURM wall).
#
# Written 2026-04-28.

export PYTHONUNBUFFERED=1
export PATH=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env
NV_BASE=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NV_BASE/cublas/lib:$NV_BASE/cuda_cupti/lib:$NV_BASE/cuda_nvrtc/lib:$NV_BASE/cuda_runtime/lib:$NV_BASE/cudnn/lib:$NV_BASE/cufft/lib:$NV_BASE/cusolver/lib:$NV_BASE/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nccl/lib:/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export HF_TOKEN_PATH=/dev/null
cd /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl

mkdir -p results_T5000_uncomplete

CKPT_DIR=results_T5000

# Index → (name, prompt). Order matches saved checkpoint files.
NAMES=(
    sweep_gs1p0_dt0p005_T5000
    sweep_gs3p0_dt0p005_T5000
    sweep_gs7p5_dt0p005_T5000
    sweep_gs15p0_dt0p005_T5000
    sweep_gs1p0_dt0p0025_T5000
    sweep_gs3p0_dt0p0025_T5000
    sweep_gs7p5_dt0p0025_T5000
    sweep_gs15p0_dt0p0025_T5000
    mp_p0_cat_gs1p0_dt0p005_T5000
    mp_p0_cat_gs3p0_dt0p005_T5000
    mp_p0_cat_gs7p5_dt0p005_T5000
    mp_p0_cat_gs15p0_dt0p005_T5000
    mp_p0_cat_gs1p0_dt0p0025_T5000
    mp_p0_cat_gs3p0_dt0p0025_T5000
    mp_p0_cat_gs7p5_dt0p0025_T5000
    mp_p0_cat_gs15p0_dt0p0025_T5000
    mp_p5_elderly_gs1p0_dt0p005_T5000
    mp_p5_elderly_gs3p0_dt0p005_T5000
    mp_p5_elderly_gs7p5_dt0p005_T5000
    mp_p5_elderly_gs15p0_dt0p005_T5000
    mp_p5_elderly_gs1p0_dt0p0025_T5000
    mp_p5_elderly_gs3p0_dt0p0025_T5000
)

PROMPTS=(
    "a superrealistic professional photograph of a person"
    "a superrealistic professional photograph of a person"
    "a superrealistic professional photograph of a person"
    "a superrealistic professional photograph of a person"
    "a superrealistic professional photograph of a person"
    "a superrealistic professional photograph of a person"
    "a superrealistic professional photograph of a person"
    "a superrealistic professional photograph of a person"
    "a photograph of a cat sitting on a windowsill"
    "a photograph of a cat sitting on a windowsill"
    "a photograph of a cat sitting on a windowsill"
    "a photograph of a cat sitting on a windowsill"
    "a photograph of a cat sitting on a windowsill"
    "a photograph of a cat sitting on a windowsill"
    "a photograph of a cat sitting on a windowsill"
    "a photograph of a cat sitting on a windowsill"
    "a portrait of an elderly person"
    "a portrait of an elderly person"
    "a portrait of an elderly person"
    "a portrait of an elderly person"
    "a portrait of an elderly person"
    "a portrait of an elderly person"
)

GS_VALUES=(1.0 3.0 7.5 15.0 1.0 3.0 7.5 15.0 1.0 3.0 7.5 15.0 1.0 3.0 7.5 15.0 1.0 3.0 7.5 15.0 1.0 3.0)
DT_VALUES=(0.005 0.005 0.005 0.005 0.0025 0.0025 0.0025 0.0025 0.005 0.005 0.005 0.005 0.0025 0.0025 0.0025 0.0025 0.005 0.005 0.005 0.005 0.0025 0.0025)

NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
PROMPT=${PROMPTS[$SLURM_ARRAY_TASK_ID]}
GS=${GS_VALUES[$SLURM_ARRAY_TASK_ID]}
DT=${DT_VALUES[$SLURM_ARRAY_TASK_ID]}
CKPT=${CKPT_DIR}/langevin_ckpt_${NAME}.npz

echo "=== Resume task $SLURM_ARRAY_TASK_ID: name=$NAME, dt=$DT, gs=$GS ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Ckpt: $CKPT"
echo "Start: $(date)"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi

python scripts/run_ssa_da_clip.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --name "$NAME" \
    --prompt "$PROMPT" \
    --guidance_scale "$GS" --gen_steps 50 \
    --height 1024 --width 1024 \
    --resume_checkpoint "$CKPT" \
    --n_samples 200 --n_traj 200 \
    --T 5000 --dt "$DT" --t_score 1 \
    --snapshot_schedule log_and_fixed \
    --compute_entropy \
    --score_batch_size 16 \
    --clip_batch_size 4 \
    --output_dir results_T5000_uncomplete

echo "Done task $SLURM_ARRAY_TASK_ID: $(date)"
