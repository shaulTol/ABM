#!/bin/bash
#SBATCH --job-name=sdxl_null
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --output=results/sdxl_null_%A_%a.out
#SBATCH --error=results/sdxl_null_%A_%a.err

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
cd <PROJECT_DIR>/experiments/sdxl

mkdir -p results

echo "=== SDXL Elderly MC Null: task ${SLURM_ARRAY_TASK_ID} ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# RUNTIME WARNING: Each replicate takes ~19 hours of Langevin on L40S
# (200 traj × 2000 steps × SDXLScore UNet forward passes).
# 3 tasks × 1 rep/task = 3 total null replicates.
# With 48h limit, 1 replicate per job fits comfortably.
python run_sdxl_mc_null.py \
    --task_id ${SLURM_ARRAY_TASK_ID} \
    --reps_per_task 1 \
    --N 200 \
    --T 2000 \
    --dt 0.0025 \
    --guidance_scale 7.5 \
    --n_eigvals 10 \
    --latents_path results/clip_da_latents_elderly_gs7p5.npz \
    --output_dir results \
    --score_batch_size 16 \
    --clip_batch_size 4

echo "Done: $(date)"
