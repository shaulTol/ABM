#!/bin/bash
#SBATCH --job-name=ala_niso100k
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-33
#SBATCH --output=/sci/labs/orzuk/shaulytolk/ABM/experiments/mc_dip_validation/results/ala_niso100k_%A_%a.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/ABM/experiments/mc_dip_validation/results/ala_niso100k_%A_%a.err

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
cd /sci/labs/orzuk/shaulytolk/ABM/experiments/mc_dip_validation

mkdir -p results

echo "=== Alanine MC Null ISOTROPIC lag100k: task ${SLURM_ARRAY_TASK_ID} ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# 34 tasks × 3 reps/task = 102 reps (~37 min/rep, ~111 min/task, fits in 4h)
python run_alanine_null_iso.py \
    --task_id ${SLURM_ARRAY_TASK_ID} \
    --reps_per_task 3 \
    --N 2000 \
    --lag 100000 \
    --dt 0.1 \
    --h 0.0091 \
    --n_eigvals 10 \
    --ctau_path ../save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz \
    --output_dir results \
    --output_prefix alanine_null_iso_lag100k

echo "Done: $(date)"
