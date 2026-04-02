#!/bin/bash
#SBATCH --job-name=cn_ssa_cmp
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=cn_ssa_cmp_%A_%a.out
#SBATCH --error=cn_ssa_cmp_%A_%a.err

# ControlNet SSA comparison: circle scribble vs DPS face scribble
# Both use SDXL Turbo + ControlNet-Scribble, CFG=0.0, 4 gen steps

export PYTHONUNBUFFERED=1

export PATH=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env

# NVIDIA library paths
NV_BASE=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NV_BASE/cublas/lib:$NV_BASE/cuda_cupti/lib:$NV_BASE/cuda_nvrtc/lib:$NV_BASE/cuda_runtime/lib:$NV_BASE/cudnn/lib:$NV_BASE/cufft/lib:$NV_BASE/cusolver/lib:$NV_BASE/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nccl/lib:/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# HF cache
export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export HF_TOKEN_PATH=/dev/null

# Wandb: use local /tmp to avoid filling NFS quota
export WANDB_DIR=/tmp/wandb_${SLURM_JOB_ID}
mkdir -p $WANDB_DIR

EXPDIR=/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl
cd $EXPDIR

# --- Array configuration ---
SCRIBBLES=("data/circle_scribble.png" "data/dps_face_scribble.png")
NAMES=("cn_circle" "cn_dps_face")

CONTROL_IMG=${SCRIBBLES[$SLURM_ARRAY_TASK_ID]}
RUN_NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

# Shared params
CN_MODEL="xinsir/controlnet-scribble-sdxl-1.0"
CN_SCALE=0.8
PROMPT="a superrealistic professional photograph of"

echo "=== ControlNet SSA Compare: $RUN_NAME ==="
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Control image: $CONTROL_IMG"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

python scripts/compute_sdxl_ssa_v2.py \
    --model_id "stabilityai/sdxl-turbo" \
    --name "sdxl_ssa_${RUN_NAME}" \
    --prompt "$PROMPT" \
    --gen_steps 4 \
    --guidance_scale 0.0 \
    --height 512 --width 512 \
    --n_samples 200 \
    --n_traj 200 \
    --T 500 --dt 0.02 --t_score 1 \
    --score_batch_size 128 \
    --gen_batch_size 4 \
    --compute_entropy --entropy_k 10 \
    --controlnet_model_id "$CN_MODEL" \
    --control_image_path "$CONTROL_IMG" \
    --controlnet_scale $CN_SCALE \
    --output_dir results_controlnet \
    --seed 42

echo "=== Done: $RUN_NAME ==="
echo "Date: $(date)"
