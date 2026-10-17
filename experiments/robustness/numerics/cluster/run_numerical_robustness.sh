#!/bin/bash
#SBATCH --job-name=numerical_robustness
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=numerical_robustness_%j.out
#SBATCH --error=numerical_robustness_%j.err

# Numerical parameters (appendix "Numerical parameters"): seven-rung 10D bimodal
# ladder, dt in {0.005, 0.01, 0.02} at N=200 and N in {50, 100, 200, 400} at
# dt=0.01, five seeds, 1/N first-crossing stopping rule.
# Writes results/p2_numerical_robustness_fc.{json,_table.md} (+ _rho_sq.npz).
# RNG stays on the CPU generator, so --device cuda matches a CPU run.

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

python -u compute_numerical_robustness.py --device cuda --stopping first --out_tag _fc
