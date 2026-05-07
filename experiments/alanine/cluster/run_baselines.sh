#!/bin/bash
#SBATCH --job-name=ala_base
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=results/ala_base_%j.out
#SBATCH --error=results/ala_base_%j.err

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
export LD_LIBRARY_PATH=<USER_SITE_PACKAGES>/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=<USER_SITE_PACKAGES>/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=<TORCH_LIB>:$LD_LIBRARY_PATH

EXPDIR=<PROJECT_DIR>/experiments/alanine
cd $EXPDIR

mkdir -p results

# Run each baseline sequentially (CPU-only, ~10 min each)
echo "=== Alanine Baselines ==="
echo "Start: $(date)"

echo "--- Sparse K-means ---"
python -u run_sparse_kmeans.py

echo "--- Spectral Map ---"
python -u run_spectral_map.py

echo "--- Diffusion Maps ---"
python -u run_diffusion_maps.py

echo "Done: $(date)"
