#!/bin/bash
#SBATCH --job-name=mp_ssa
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-27
#SBATCH --output=results/logs/ssa_%a_%j.out
#SBATCH --error=results/logs/ssa_%a_%j.err

export PYTHONUNBUFFERED=1
export PATH=<CONDA_BIN>:$PATH
source <CONDA_SH>
conda activate <CONDA_ENV>
NV_BASE=<NV_SITE_PACKAGES>
export LD_LIBRARY_PATH=$NV_BASE/cublas/lib:$NV_BASE/cuda_cupti/lib:$NV_BASE/cuda_nvrtc/lib:$NV_BASE/cuda_runtime/lib:$NV_BASE/cudnn/lib:$NV_BASE/cufft/lib:$NV_BASE/cusolver/lib:$NV_BASE/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=<USER_SITE_PACKAGES>/nccl/lib:<USER_SITE_PACKAGES>/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=<TORCH_LIB>:$LD_LIBRARY_PATH
export HF_HOME=<HF_CACHE>
export HF_TOKEN_PATH=/dev/null
cd <PROJECT_DIR>/experiments/sdxl

mkdir -p results/logs

PROMPTS=(
    "a photograph of a cat sitting on a windowsill"
    "a landscape photograph of mountains at sunset"
    "a still life painting of flowers in a vase"
    "a street photograph of a busy city intersection"
    "a photograph of a plate of food at a restaurant"
    "a portrait of an elderly person"
    "a photograph of a car parked on a street"
)
PROMPT_LABELS=(cat mountains flowers street food elderly car)

GS_VALUES=(1.0 3.0 7.5 15.0)
GS_LABELS=(gs1p0 gs3p0 gs7p5 gs15p0)

PROMPT_IDX=$((SLURM_ARRAY_TASK_ID / 4))
GS_IDX=$((SLURM_ARRAY_TASK_ID % 4))

PROMPT="${PROMPTS[$PROMPT_IDX]}"
PROMPT_LABEL="${PROMPT_LABELS[$PROMPT_IDX]}"
GS="${GS_VALUES[$GS_IDX]}"
GS_LABEL="${GS_LABELS[$GS_IDX]}"

NAME="mp_p${PROMPT_IDX}_${PROMPT_LABEL}_${GS_LABEL}"

echo "=== SSA task $SLURM_ARRAY_TASK_ID: prompt='$PROMPT', gs=$GS ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

python run_ssa_da_clip.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --name "$NAME" \
    --prompt "$PROMPT" \
    --guidance_scale "$GS" --gen_steps 50 \
    --height 1024 --width 1024 \
    --resume_x0 "results/gen_${NAME}_quick_x0.npz" \
    --n_samples 200 --n_traj 200 \
    --T 2000 --dt 0.0005 --t_score 1 \
    --snapshot_schedule log_and_fixed \
    --ssa_only \
    --compute_entropy \
    --score_batch_size 16 \
    --clip_batch_size 4 \
    --output_dir results

echo "Done task $SLURM_ARRAY_TASK_ID: $(date)"
