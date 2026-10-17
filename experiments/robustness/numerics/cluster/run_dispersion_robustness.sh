#!/bin/bash
#SBATCH --job-name=dispersion_robustness
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=dispersion_robustness_%j.out
#SBATCH --error=dispersion_robustness_%j.err

# Gaussian vs parallel-ellipses robustness (appendix "Numerical parameters", last
# paragraph): five seeds over step sizes, trajectory counts, lag windows and both
# stopping rules (first crossing = the paper's; largest lag = sensitivity).
# Writes results/p2_dispersion_robustness.{json,_table.md} (+ _rho_sq.npz).
# --allow_baseline_mismatch: the script first checks it reproduces the figure's
# values; the remaining ~2-4% gap is quadrature plus finite-sample draw.

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PATH=<CONDA_BIN>:$PATH
source <CONDA_SH>
conda activate <CONDA_ENV>

NV_BASE=<NV_SITE_PACKAGES>
export LD_LIBRARY_PATH=$NV_BASE/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_cupti/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cufft/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cusolver/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=<TORCH_LIB>:$LD_LIBRARY_PATH

cd <PROJECT_DIR>/experiments/robustness/numerics

python -u compute_dispersion_robustness.py --device cuda --n_seeds 5 --allow_baseline_mismatch
