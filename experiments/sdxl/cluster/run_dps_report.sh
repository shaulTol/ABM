#!/bin/bash
#SBATCH --job-name=dps_report
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=dps_report_%j.out
#SBATCH --error=dps_report_%j.err

# PDF report for all DPS conditions (9 experiments: 3 scribbles x 3 dt).
# Submit with: sbatch --dependency=afterok:<JOBID> cluster/run_dps_report.sh
# where <JOBID> is the array job from run_dps_converged.sh.

export PYTHONUNBUFFERED=1

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

export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export HF_TOKEN_PATH=/dev/null

cd /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl

echo "=== DPS Full Report ==="
echo "Date: $(date)"

python scripts/plot_dps_scatter.py \
    --results_dir results_clip_da \
    --model_id stabilityai/sdxl-turbo \
    --output_path results_clip_da/dps_scatter_all.pdf

echo "=== Done ==="
echo "Date: $(date)"
