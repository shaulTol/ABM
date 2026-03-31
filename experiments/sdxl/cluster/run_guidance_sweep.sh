#!/bin/bash
#SBATCH --job-name=sdxl_v2_%a
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=sdxl_v2_%A_%a.out
#SBATCH --error=sdxl_v2_%A_%a.err
#SBATCH --array=0-6

# v2: Guidance scale sweep with fp16 UNet + wandb monitoring
# Array 0-6 maps to guidance_scale in {1, 2, 3, 5, 7.5, 10, 15}

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

EXPDIR=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/modality_score/autocorrelation-mixture-analysis/experiments/sdxl_ssa
cd $EXPDIR

# Guidance scale sweep
GUIDANCE_VALUES=(1.0 2.0 3.0 5.0 7.5 10.0 15.0)
W=${GUIDANCE_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "=== SDXL SSA v2: Guidance Sweep ==="
echo "Guidance scale: w=$W"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

python scripts/compute_sdxl_ssa_v2.py \
    --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
    --name "sdxl_base_w${W}" \
    --prompt "A photo of a human face" \
    --gen_steps 50 \
    --guidance_scale $W \
    --height 512 --width 512 \
    --n_samples 2000 \
    --n_traj 2000 \
    --T 2500 --dt 0.02 --t_score 1 \
    --score_batch_size 128 \
    --gen_batch_size 4 \
    --compute_entropy --entropy_k 10 \
    --output_dir results_guidance_sweep \
    --seed 42

echo "=== Done: w=$W ==="
echo "Date: $(date)"
