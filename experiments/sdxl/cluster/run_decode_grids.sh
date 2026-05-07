#!/bin/bash
#SBATCH --job-name=mp_decode
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-27
#SBATCH --output=results/decode_%a_%j.out
#SBATCH --error=results/decode_%a_%j.err

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

PROMPT_LABELS=(cat mountains flowers street food elderly car)
GS_LABELS=(gs1p0 gs3p0 gs7p5 gs15p0)

PROMPT_IDX=$((SLURM_ARRAY_TASK_ID / 4))
GS_IDX=$((SLURM_ARRAY_TASK_ID % 4))

PROMPT_LABEL="${PROMPT_LABELS[$PROMPT_IDX]}"
GS_LABEL="${GS_LABELS[$GS_IDX]}"

NAME="mp_p${PROMPT_IDX}_${PROMPT_LABEL}_${GS_LABEL}"
OUTDIR="results/grids/p${PROMPT_IDX}_${PROMPT_LABEL}"
mkdir -p "$OUTDIR"

echo "=== Decode task $SLURM_ARRAY_TASK_ID: $NAME ==="
echo "Host: $(hostname)"
echo "Start: $(date)"

python -u decode_all_grids.py \
    --latents_npz results/gen_${NAME}_quick_x0.npz \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --latent_h 128 --latent_w 128 \
    --batch_size 1 \
    --n_cols 20 \
    --thumb_size 256 \
    --output_path "${OUTDIR}/grid_${PROMPT_LABEL}_${GS_LABEL}.png"

echo "Done: $(date)"
