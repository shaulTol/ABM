#!/bin/bash
#SBATCH --job-name=ext_ssa
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-23
#SBATCH --output=/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl_extended_prompts/ssa_%a_%j.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl_extended_prompts/ssa_%a_%j.err

# Phase 2: Langevin + SSA. Resumes from Phase 1a x0 latents.
# dt=0.005, T=2000, T_phys=10s. Matching Table 4 exactly.
# 6 prompts × 4 guidance scales = 24 tasks.
# Depends on run_ext_quick.sh completing first.

export PYTHONUNBUFFERED=1
export PATH=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env

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

cd /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl

OUTDIR=/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl_extended_prompts
mkdir -p $OUTDIR

PROMPTS=(
    "a photo of a bat"
    "a photo of a crane"
    "a photo of a mouse"
    "a photo of a seal"
    "a photo of a jaguar"
    "a photo of a palm"
)
PROMPT_LABELS=(bat crane mouse seal jaguar palm)

GS_VALUES=(1.0 3.0 7.5 15.0)
GS_LABELS=(gs1p0 gs3p0 gs7p5 gs15p0)

PROMPT_IDX=$((SLURM_ARRAY_TASK_ID / 4))
GS_IDX=$((SLURM_ARRAY_TASK_ID % 4))

PROMPT="${PROMPTS[$PROMPT_IDX]}"
PROMPT_LABEL="${PROMPT_LABELS[$PROMPT_IDX]}"
GS="${GS_VALUES[$GS_IDX]}"
GS_LABEL="${GS_LABELS[$GS_IDX]}"

NAME="ext_p${PROMPT_IDX}_${PROMPT_LABEL}_${GS_LABEL}_dt0p005"
X0_PATH="${OUTDIR}/gen_ext_p${PROMPT_IDX}_${PROMPT_LABEL}_${GS_LABEL}_quick_x0.npz"

echo "=== Phase 2 (Langevin + SSA) task $SLURM_ARRAY_TASK_ID: prompt='$PROMPT', gs=$GS ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "x0: $X0_PATH"
echo "Start: $(date)"

if [ ! -f "$X0_PATH" ]; then
    echo "ERROR: x0 latents not found at $X0_PATH — Phase 1a must complete first!"
    exit 1
fi

python scripts/run_ssa_da_clip.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --name "$NAME" \
    --prompt "$PROMPT" \
    --guidance_scale "$GS" --gen_steps 50 \
    --height 1024 --width 1024 \
    --resume_x0 "$X0_PATH" \
    --n_samples 200 --n_traj 200 \
    --T 2000 --dt 0.005 --t_score 1 \
    --snapshot_schedule log_and_fixed \
    --ssa_only \
    --compute_entropy \
    --score_batch_size 16 \
    --clip_batch_size 4 \
    --output_dir "$OUTDIR"

echo "Done task $SLURM_ARRAY_TASK_ID: $(date)"
