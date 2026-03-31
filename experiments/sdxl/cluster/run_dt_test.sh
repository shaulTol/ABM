#!/bin/bash
#SBATCH --job-name=sdxl_dt
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=sdxl_dt_%j.out
#SBATCH --error=sdxl_dt_%j.err

# Diagnostic: dt=0.0001 vs dt=0.02, same T=2500 steps, w=1.0 (no CFG)
# Physical time = 0.25 (vs 50 in main runs)
# Tests whether large dt causes artificial fast mixing

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

export WANDB_DIR=/tmp/wandb_${SLURM_JOB_ID}
mkdir -p $WANDB_DIR

EXPDIR=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/modality_score/autocorrelation-mixture-analysis/experiments/sdxl_ssa
cd $EXPDIR

echo "=== SDXL SSA: dt diagnostic (dt=0.0001) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

python scripts/compute_sdxl_ssa_v2.py \
    --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
    --name "sdxl_base_w1.0_dt0.0001" \
    --prompt "A photo of a human face" \
    --gen_steps 50 \
    --guidance_scale 1.0 \
    --height 512 --width 512 \
    --n_samples 2000 \
    --n_traj 2000 \
    --T 2500 --dt 0.0001 --t_score 1 \
    --score_batch_size 128 \
    --gen_batch_size 4 \
    --compute_entropy --entropy_k 10 \
    --output_dir results_dt_test \
    --seed 42

echo "=== Done ==="
echo "Date: $(date)"
