#!/bin/bash
#SBATCH --job-name=auto_da
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --output=auto_da_%A_%a.out
#SBATCH --error=auto_da_%A_%a.err

# Run SSA+DA in CLIP space for the 3 autoresearch experiments:
#   0 = Egg (face_blurred + "a portrait photograph")
#   1 = Square (face_boxed + "a street portrait")
#   2 = Street (face_blurred + "a street portrait")
# Each: ~35 min Langevin + ~5 min CLIP embedding + ~3 min correlation table

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

# --- Step 0: Generate scribble images ---
python scripts/make_autoresearch_scribbles.py

# --- Config per array task ---
NAMES=("egg" "square" "street")
PROMPTS=("a portrait photograph" "a street portrait" "a street portrait")
SCRIBBLES=("data/face_blurred_scribble.png" "data/face_boxed_scribble.png" "data/face_blurred_scribble.png")

NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
PROMPT="${PROMPTS[$SLURM_ARRAY_TASK_ID]}"
SCRIBBLE="${SCRIBBLES[$SLURM_ARRAY_TASK_ID]}"

echo "=== Task $SLURM_ARRAY_TASK_ID: $NAME ==="
echo "  Prompt: $PROMPT"
echo "  Scribble: $SCRIBBLE"

# --- Step 1: SSA + DA in CLIP space ---
python scripts/run_ssa_da_clip.py \
    --model_id stabilityai/sdxl-turbo \
    --name "autoresearch_${NAME}" \
    --prompt "$PROMPT" \
    --guidance_scale 0.0 \
    --gen_steps 4 \
    --controlnet_model_id xinsir/controlnet-scribble-sdxl-1.0 \
    --control_image_path "$SCRIBBLE" \
    --controlnet_scale 0.8 \
    --n_samples 200 --n_traj 200 \
    --T 500 --dt 0.001 \
    --t_score 1 \
    --also_compute_lc \
    --output_dir results_autoresearch

# --- Step 2: Correlation table ---
echo ""
echo "=== Correlation table for $NAME ==="
python scripts/full_correlation_table.py \
    --latents_npz results_autoresearch/clip_da_latents_autoresearch_${NAME}.npz \
    --clip_da_npz results_autoresearch/clip_da_autoresearch_${NAME}.npz \
    --model_id stabilityai/sdxl-turbo \
    --label "$NAME" \
    --output_dir results_autoresearch

echo ""
echo "=== Done: $NAME ==="
