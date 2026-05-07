#!/bin/bash
#SBATCH --job-name=ell_null
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/ell_null_%j.out
#SBATCH --error=results/ell_null_%j.err

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
cd <PROJECT_DIR>/experiments/synthetic

mkdir -p results

echo "=== Parallel Ellipses MC Null + Dip Test ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# B=1000 null replicates, N=10000 trajectories, lag=800
# Estimated: ~10 min total on GPU
python run_ellipses_null.py \
    --B 1000 \
    --lag 800 \
    --n_traj 10000 \
    --dt 0.01 \
    --seed 42 \
    --output_dir results

echo "Done: $(date)"
