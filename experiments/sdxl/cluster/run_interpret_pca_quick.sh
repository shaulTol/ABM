#!/bin/bash
#SBATCH --job-name=interp_pca
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=interp_pca_%j.out
#SBATCH --error=interp_pca_%j.err

# Quick PCA + DA interpretation on saved circle + dps_face latents (~3 min each)

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

cd /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl

echo "=== Circle ==="
python scripts/interpret_da_clip.py \
    --results_dir results_controlnet \
    --name sdxl_ssa_cn_circle \
    --model_id stabilityai/sdxl-turbo \
    --n_extremes 50 --n_grid 8 \
    --output_dir results_controlnet

echo ""
echo "=== DPS Face ==="
python scripts/interpret_da_clip.py \
    --results_dir results_controlnet \
    --name sdxl_ssa_cn_dps_face \
    --model_id stabilityai/sdxl-turbo \
    --n_extremes 50 --n_grid 8 \
    --output_dir results_controlnet

echo "=== Done ==="
