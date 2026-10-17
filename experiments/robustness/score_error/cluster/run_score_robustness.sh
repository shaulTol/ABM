#!/bin/bash
#SBATCH --job-name=score_robustness
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-49%16
#SBATCH --output=score_robustness_%A_%a.out
#SBATCH --error=score_robustness_%A_%a.err

# Sensitivity to score-estimation error (appendix "Sensitivity to score-estimation
# error"): SSA sweep, one of the 50 random 10D GMMs per array task, 3 seeds,
# T_phys=100, max_lag=T, 1/N first-crossing stopping rule (the paper's).
# Writes results/score_robustness_full_fc_shard_<i>_<i+1>.json (+ _rho_sq.npz).
# Afterwards run cluster/run_score_robustness_merge.sh.

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

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

cd <PROJECT_DIR>/experiments/robustness/score_error

python -u compute_score_robustness.py \
    --gmm_start ${SLURM_ARRAY_TASK_ID} --gmm_end $(( SLURM_ARRAY_TASK_ID + 1 )) \
    --device cuda --stopping first --run_tag full_fc
