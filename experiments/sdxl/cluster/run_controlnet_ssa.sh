#!/bin/bash
#SBATCH --job-name=sdxl_cn_ssa
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=sdxl_cn_%A.out
#SBATCH --error=sdxl_cn_%A.err

# ControlNet SSA experiment: measure how structural conditioning affects multimodality
# Expects: control image already preprocessed (scribble PNG) at CONTROL_IMG path below

export PYTHONUNBUFFERED=1

export PATH=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env

# NVIDIA library paths (required for torch to find CUDA libs)
NV_BASE=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NV_BASE/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_cupti/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cufft/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cusolver/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# HF cache
export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export HF_TOKEN_PATH=/dev/null

# Wandb: use local /tmp to avoid filling NFS quota
export WANDB_DIR=/tmp/wandb_${SLURM_JOB_ID}
mkdir -p $WANDB_DIR

EXPDIR=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa-paper/experiments/sdxl
cd $EXPDIR

# --- Configuration ---
CONTROL_IMG="data/scribble_face.png"
CN_MODEL="xinsir/controlnet-scribble-sdxl-1.0"
CN_SCALE=0.8
PROMPT="A photo of a human face"
W=7.5

echo "=== SDXL SSA v2: ControlNet ==="
echo "ControlNet: $CN_MODEL"
echo "Control image: $CONTROL_IMG"
echo "ControlNet scale: $CN_SCALE"
echo "Guidance scale: w=$W"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

# Run with ControlNet
python scripts/compute_sdxl_ssa_v2.py \
    --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
    --name "sdxl_controlnet_scribble_s${CN_SCALE}" \
    --prompt "$PROMPT" \
    --gen_steps 50 \
    --guidance_scale $W \
    --height 512 --width 512 \
    --n_samples 2000 \
    --n_traj 2000 \
    --T 2500 --dt 0.02 --t_score 1 \
    --score_batch_size 96 \
    --gen_batch_size 4 \
    --compute_entropy --entropy_k 10 \
    --controlnet_model_id "$CN_MODEL" \
    --control_image_path "$CONTROL_IMG" \
    --controlnet_scale $CN_SCALE \
    --output_dir results_controlnet \
    --seed 42

echo "=== Done: ControlNet SSA ==="
echo "Date: $(date)"
