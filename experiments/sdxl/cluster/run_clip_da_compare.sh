#!/bin/bash
#SBATCH --job-name=clip_da_cmp_%a
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=clip_da_cmp_%A_%a.out
#SBATCH --error=clip_da_cmp_%A_%a.err
#SBATCH --array=0-1

# Full CLIP-mode SSA+DA for circle (0) and DPS face (1).
# Matches existing LC-mode params: SDXL Turbo, gen_steps=2, cfg=0.0, cs=0.5,
# N=200, T=500, dt=0.0001. Also computes LC SSA for direct comparison.

export PYTHONUNBUFFERED=1

export PATH=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env

# NVIDIA library paths
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

export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export HF_TOKEN_PATH=/dev/null
export WANDB_DIR=/tmp/wandb_${SLURM_JOB_ID}
mkdir -p $WANDB_DIR

cd /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl

# --- Array config ---
NAMES=("clip_cn_circle" "clip_cn_dps_face")
CONTROL_IMAGES=("data/circle_scribble.png" "data/dps_face_scribble.png")

NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
CTRL=${CONTROL_IMAGES[$SLURM_ARRAY_TASK_ID]}

echo "=== CLIP-mode SSA+DA: ${NAME} ==="
echo "Control image: ${CTRL}"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

# Step 1: Post-hoc interpretation of EXISTING LC results
echo ""
echo "=== Step 1: Post-hoc interpretation of existing LC-DA ==="
LC_NAMES=("sdxl_ssa_cn_circle" "sdxl_ssa_cn_dps_face")
LC_NAME=${LC_NAMES[$SLURM_ARRAY_TASK_ID]}

python scripts/interpret_da_clip.py \
    --results_dir results_controlnet \
    --name "${LC_NAME}" \
    --model_id "stabilityai/sdxl-turbo" \
    --clip_model openai/clip-vit-large-patch14 \
    --n_extremes 50 \
    --n_grid 8 \
    --output_dir results_clip_da

echo ""
echo "=== Step 2: Full CLIP-mode SSA+DA ==="

# Step 2: Full CLIP-mode pipeline (new Langevin + CLIP embedding)
# Matches original params: SDXL Turbo, gen_steps=2, cfg=0.0, cs=0.5
python scripts/run_ssa_da_clip.py \
    --model_id "stabilityai/sdxl-turbo" \
    --name "${NAME}" \
    --prompt "a superrealistic professional photograph of" \
    --guidance_scale 0.0 \
    --gen_steps 2 \
    --height 512 --width 512 \
    --n_samples 200 \
    --n_traj 200 \
    --T 500 --dt 0.0001 --t_score 1 \
    --score_batch_size 128 \
    --gen_batch_size 4 \
    --controlnet_model_id "xinsir/controlnet-scribble-sdxl-1.0" \
    --control_image_path "${CTRL}" \
    --controlnet_scale 0.5 \
    --clip_model openai/clip-vit-large-patch14 \
    --compute_entropy --entropy_k 10 \
    --also_compute_lc \
    --output_dir results_clip_da \
    --seed 42

echo "=== Done: ${NAME} ==="
echo "Date: $(date)"
