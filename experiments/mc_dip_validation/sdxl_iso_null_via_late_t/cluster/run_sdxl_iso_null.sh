#!/bin/bash
#SBATCH --job-name=sdxl_iso_null
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15
#SBATCH --output=/sci/labs/orzuk/shaulytolk/ABM/experiments/mc_dip_validation/sdxl_iso_null_via_late_t/results/array_%A_%a.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/ABM/experiments/mc_dip_validation/sdxl_iso_null_via_late_t/results/array_%A_%a.err

# SDXL elderly iso-Gaussian null array job.
# 16 tasks × N_per_task=50 = N_big=800 trajectories.
# Estimated per-task runtime: ~5.8h on L40S (extrapolating from 19h@N=200 to
# N=50 with score_batch_size=16: 4 batches/step instead of 13). 8h time limit
# gives ~2h margin.
#
# DO NOT submit this until the sanity check (run_sanity_check.sh) has been
# verified — see README.md for pass criteria.

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

cd /sci/labs/orzuk/shaulytolk/ABM/experiments/mc_dip_validation/sdxl_iso_null_via_late_t
mkdir -p results

echo "=== SDXL iso-null array task ${SLURM_ARRAY_TASK_ID} ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

python run_sdxl_iso_null.py \
    --task_id ${SLURM_ARRAY_TASK_ID} \
    --t_null 999 \
    --N_per_task 50 \
    --T 2000 \
    --dt 0.0025 \
    --guidance_scale 7.5 \
    --base_seed 9000 \
    --output_dir results \
    --score_batch_size 16 \
    --clip_batch_size 4 \
    --var_f_mode sigma_t_null_sq

echo "Done: $(date)"
