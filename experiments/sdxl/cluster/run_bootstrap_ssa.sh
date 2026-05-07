#!/bin/bash
#SBATCH --job-name=traj_bootstrap
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --array=0-58
#SBATCH --output=results/logs/bootstrap_%a_%j.out
#SBATCH --error=results/logs/bootstrap_%a_%j.err

export PYTHONUNBUFFERED=1
export PATH=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env

NV_BASE=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NV_BASE/cublas/lib:$NV_BASE/cuda_cupti/lib:$NV_BASE/cuda_nvrtc/lib:$NV_BASE/cuda_runtime/lib:$NV_BASE/cudnn/lib:$NV_BASE/cufft/lib:$NV_BASE/cusolver/lib:$NV_BASE/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nccl/lib:/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

cd /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl

mkdir -p results/logs

# Get the cell name from sorted NPZ list
CELL=$(ls results/z_clip_traj_*.npz | sort | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" | sed 's|results/z_clip_traj_||;s/.npz//')

echo "=== Bootstrap: cell=$CELL, array_id=$SLURM_ARRAY_TASK_ID ==="

python3 -u traj_averaged_pipeline.py --B 2000 --cell "$CELL" --output "results/bootstrap_${CELL}.json"

echo "Done: $CELL"
