#!/bin/bash
#SBATCH --job-name=reemb
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7
#SBATCH --output=results_da_pcavda/reembed_n200_%a_%j.out
#SBATCH --error=results_da_pcavda/reembed_n200_%a_%j.err

# Re-embed existing Langevin checkpoints with fixed latent_shape.
# Skips generation + Langevin entirely — only VAE decode + CLIP + DA.
# ~1-2h per prompt instead of ~15h.

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

PROMPTS=(
    "a photograph of a cat sitting on a windowsill"
    "a landscape photograph of mountains at sunset"
    "a still life painting of flowers in a vase"
    "a street photograph of a busy city intersection"
    "a photograph of a plate of food at a restaurant"
    "a portrait of an elderly person"
    "a photograph of a car parked on a street"
    "a superrealistic professional photograph of a person"
)
LABELS=(cat mountains flowers street food elderly car person)

IDX=$SLURM_ARRAY_TASK_ID
PROMPT="${PROMPTS[$IDX]}"
LABEL="${LABELS[$IDX]}"
NAME="${LABEL}_gs7p5_n200"
CKPT="results_da_pcavda/langevin_ckpt_${NAME}.npz"

echo "=== Re-embed N=200 task $IDX: $LABEL ==="
echo "Checkpoint: $CKPT"
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

python scripts/run_ssa_da_clip.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --name "$NAME" \
    --prompt "$PROMPT" \
    --guidance_scale 7.5 --gen_steps 50 \
    --height 1024 --width 1024 \
    --resume_checkpoint "$CKPT" \
    --n_samples 200 --n_traj 200 \
    --T 2000 --dt 0.0005 --t_score 1 \
    --snapshot_schedule log_and_fixed \
    --clip_outlier_sigma 2.0 \
    --clip_batch_size 4 \
    --score_batch_size 16 \
    --output_dir results_da_pcavda

echo "Done task $IDX ($LABEL): $(date)"
