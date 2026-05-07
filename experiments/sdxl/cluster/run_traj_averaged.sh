#!/bin/bash
#SBATCH --job-name=traj_embed_ssa
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-23
#SBATCH --output=results/logs/phase_d_%a_%j.out
#SBATCH --error=results/logs/phase_d_%a_%j.err

# Phase D: CLIP embedding + trajectory-averaged SSA for 24 extended prompt cells.
# These already have dt=0.005 Langevin checkpoints — only need embed + recompute.

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

# 24 cells: 6 extended prompts × 4 guidance scales
# Checkpoints exist at: results/langevin_ckpt_ext_p*_dt0p005.npz
PROMPT_LABELS=(bat crane mouse seal jaguar palm)
GS_LABELS=(gs1p0 gs3p0 gs7p5 gs15p0)

# Compute prompt and gs indices from array task ID
PROMPT_IDX=$((SLURM_ARRAY_TASK_ID / 4))
GS_IDX=$((SLURM_ARRAY_TASK_ID % 4))

PLABEL="${PROMPT_LABELS[$PROMPT_IDX]}"
GSLABEL="${GS_LABELS[$GS_IDX]}"
NAME="ext_p${PROMPT_IDX}_${PLABEL}_${GSLABEL}_dt0p005"

# Find checkpoint
CKPT_DIR="results"
CKPT="${CKPT_DIR}/langevin_ckpt_ext_p${PROMPT_IDX}_${PLABEL}_${GSLABEL}_dt0p005.npz"

if [ ! -f "$CKPT" ]; then
    # Try alternate naming
    CKPT="${CKPT_DIR}/langevin_ckpt_${NAME}.npz"
fi
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found for ${NAME}" >&2
    echo "Tried: ${CKPT_DIR}/langevin_ckpt_ext_p${PROMPT_IDX}_${PLABEL}_${GSLABEL}_dt0p005.npz"
    ls ${CKPT_DIR}/langevin_ckpt_ext_p${PROMPT_IDX}* 2>/dev/null
    exit 1
fi

echo "=== Phase D: Embed + Traj-Avg SSA ==="
echo "Cell: $NAME"
echo "Checkpoint: $CKPT"

python traj_averaged_pipeline.py \
    --checkpoint "$CKPT" \
    --dt 0.005 --T 2000 \
    --name "$NAME" \
    --output_dir results \
    --save_clip_traj

echo "Done: $NAME"
