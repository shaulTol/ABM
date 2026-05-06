#!/bin/bash
#SBATCH --job-name=ala_hs36
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-19
#SBATCH --output=/sci/labs/orzuk/shaulytolk/ABM/experiments/mc_dip_validation/results/ala_hs36_%A_%a.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/ABM/experiments/mc_dip_validation/results/ala_hs36_%A_%a.err

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

echo "=== Alanine MC Null h_scott lag=3626 (362.6ps): task ${SLURM_ARRAY_TASK_ID} ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

python run_alanine_null.py \
    --task_id ${SLURM_ARRAY_TASK_ID} \
    --reps_per_task 10 \
    --N 2000 \
    --lag 3626 \
    --dt 0.1 \
    --h 0.018282 \
    --n_eigvals 5 \
    --ctau_path ../save_autocovariance/results/alanine_C_tau_full.npz \
    --output_dir results \
    --output_prefix alanine_null_hscott_lag3626

echo "Done: $(date)"
