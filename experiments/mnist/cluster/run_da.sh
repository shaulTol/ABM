#!/bin/bash
#SBATCH --job-name=mnist_da_%a
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=mnist_da_%A_%a.out
#SBATCH --error=mnist_da_%A_%a.err
#SBATCH --array=0,1

# Compute DA directions for each MNIST digit via DDPM Tweedie score
# Array task ID = digit (0-9)
# Expected: ~15-20 min per digit

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

EXPDIR=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/modality_score/autocorrelation-mixture-analysis/experiments/ddpm_mnist
cd $EXPDIR

DIGIT=$SLURM_ARRAY_TASK_ID

echo "=== MNIST DA: digit $DIGIT ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

python scripts/compute_mnist_da.py \
    --model_dir outputs/mnist/model \
    --digit $DIGIT \
    --noise_level 1 \
    --n_samples 5000 \
    --n_traj 5000 \
    --n_checkpoints 25 \
    --max_lag 30000 \
    --dt 0.01 \
    --output_dir results_da \
    --seed 42

echo "=== Done: digit $DIGIT ==="
echo "Date: $(date)"
