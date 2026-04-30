#!/bin/bash
#SBATCH --job-name=spk_var
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/sci/labs/orzuk/shaulytolk/ABM/experiments/synthetic_spikes/spk_var_%j.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/ABM/experiments/synthetic_spikes/spk_var_%j.err

export PYTHONUNBUFFERED=1
export PATH=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env
NV_BASE=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NV_BASE/cublas/lib:$NV_BASE/cuda_cupti/lib:$NV_BASE/cuda_nvrtc/lib:$NV_BASE/cuda_runtime/lib:$NV_BASE/cudnn/lib:$NV_BASE/cufft/lib:$NV_BASE/cusolver/lib:$NV_BASE/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nccl/lib:/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

cd /sci/labs/orzuk/shaulytolk/ABM/experiments/synthetic_spikes

echo "=== Long run (tau_max=2000, var sweep): $(hostname) ==="
echo "Start: $(date)"

python -u simulate_planted_spikes_variants.py \
    --tau_max 2000.0 \
    --dt 0.01 \
    --var_factors "1.5,2,3,5,10" \
    --out_suffix _long

echo "Done: $(date)"
