#!/bin/bash
#SBATCH --job-name=mp_quick
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-27
#SBATCH --output=results_multiprompt_sweep/quick_%a_%j.out
#SBATCH --error=results_multiprompt_sweep/quick_%a_%j.err

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

mkdir -p results_multiprompt_sweep

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

NAME="mp_p${PROMPT_IDX}_${PROMPT_LABEL}_${GS_LABEL}_quick"

echo "=== Quick gen task $SLURM_ARRAY_TASK_ID: prompt='$PROMPT' (p$PROMPT_IDX/$PROMPT_LABEL), gs=$GS ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

python generate_and_entropy.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --prompt "$PROMPT" \
    --guidance_scale "$GS" --gen_steps 50 \
    --height 1024 --width 1024 \
    --n_samples 200 \
    --gen_batch_size 1 \
    --clip_batch_size 4 \
    --seed 42 \
    --name "$NAME" \
    --output_dir results_multiprompt_sweep

echo "Done task $SLURM_ARRAY_TASK_ID: $(date)"
