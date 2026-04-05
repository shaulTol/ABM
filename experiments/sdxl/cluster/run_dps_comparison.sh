#!/bin/bash
#SBATCH --job-name=dps_compare
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5
#SBATCH --output=dps_compare_%A_%a.out
#SBATCH --error=dps_compare_%A_%a.err

# Compare 3 DPS scribbles × 2 dt values:
#   0: dps_face   dt=0.001     3: dps_face   dt=0.005
#   1: dps_first  dt=0.001     4: dps_first  dt=0.005
#   2: dps_second dt=0.001     5: dps_second dt=0.005

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

# --- Task config ---
SCRIBBLES=(
    "data/dps_face_scribble.png"
    "data/dps_first_scribble.jpeg"
    "data/dps_second_scribble.jpeg"
)
NAMES_DT001=("dps_face_dt001" "dps_first_dt001" "dps_second_dt001")
NAMES_DT005=("dps_face_dt005" "dps_first_dt005" "dps_second_dt005")

TASK=${SLURM_ARRAY_TASK_ID}
SCR_IDX=$((TASK % 3))
DT_IDX=$((TASK / 3))

SCRIBBLE=${SCRIBBLES[$SCR_IDX]}
if [ $DT_IDX -eq 0 ]; then
    DT=0.001
    NAME=${NAMES_DT001[$SCR_IDX]}
else
    DT=0.005
    NAME=${NAMES_DT005[$SCR_IDX]}
fi

PROMPT="a superrealistic professional photograph of"
OUTDIR="results_clip_da"

echo "=== DPS Comparison: Task $TASK ==="
echo "  Name: $NAME"
echo "  Scribble: $SCRIBBLE"
echo "  dt: $DT"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# --- Stage 1: SSA + DA (with gender-sliced SSA) ---
echo ""
echo "=== Stage 1: run_ssa_da_clip.py ==="
python scripts/run_ssa_da_clip.py \
    --model_id "stabilityai/sdxl-turbo" \
    --name "$NAME" \
    --prompt "$PROMPT" \
    --guidance_scale 0.0 \
    --gen_steps 2 \
    --height 512 --width 512 \
    --n_samples 200 \
    --n_traj 200 \
    --T 500 --dt $DT --t_score 1 \
    --score_batch_size 128 \
    --gen_batch_size 4 \
    --clip_model openai/clip-vit-large-patch14 \
    --controlnet_model_id xinsir/controlnet-scribble-sdxl-1.0 \
    --control_image_path "$SCRIBBLE" \
    --controlnet_scale 0.5 \
    --compute_entropy --entropy_k 10 \
    --output_dir "$OUTDIR" \
    --seed 42

# --- Stage 2: Correlation table ---
echo ""
echo "=== Stage 2: full_correlation_table.py ==="
python scripts/full_correlation_table.py \
    --latents_npz "${OUTDIR}/clip_da_latents_${NAME}.npz" \
    --clip_da_npz "${OUTDIR}/clip_da_${NAME}.npz" \
    --model_id "stabilityai/sdxl-turbo" \
    --label "$NAME" \
    --output_dir "$OUTDIR"

# --- Stage 3: Entropy comparison ---
echo ""
echo "=== Stage 3: compute_entropy_comparison.py ==="
python scripts/compute_entropy_comparison.py \
    --latents_npz "${OUTDIR}/clip_da_latents_${NAME}.npz" \
    --model_id "stabilityai/sdxl-turbo" \
    --output_path "${OUTDIR}/entropy_${NAME}.json"

echo ""
echo "=== Done: $NAME ==="
echo "Date: $(date)"
