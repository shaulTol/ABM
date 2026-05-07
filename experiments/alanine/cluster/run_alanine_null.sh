#!/bin/bash
#SBATCH --job-name=ala_null2k
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --output=results/ala_null2k_%A_%a.out
#SBATCH --error=results/ala_null2k_%A_%a.err

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
cd <PROJECT_DIR>/experiments/alanine

mkdir -p results

echo "=== Alanine MC Null lag=2000: task ${SLURM_ARRAY_TASK_ID} ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# 10 tasks × 10 reps/task = 100 null replicates
# lag=2000 (~50x shorter than 100k) → ~45s/rep, ~8 min/task
python run_alanine_null.py \
    --task_id ${SLURM_ARRAY_TASK_ID} \
    --reps_per_task 10 \
    --N 2000 \
    --lag 2000 \
    --dt 0.1 \
    --h 0.0091 \
    --n_eigvals 5 \
    --ctau_path results/alanine_C_tau_full_h_half_10k.npz \
    --output_dir results \
    --output_prefix alanine_null_lag2k

echo "Done: $(date)"
