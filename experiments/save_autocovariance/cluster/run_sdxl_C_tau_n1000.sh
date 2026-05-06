#!/bin/bash
#SBATCH --job-name=ctau1k
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=/sci/labs/orzuk/shaulytolk/ABM/experiments/save_autocovariance/results/sdxl_ctau_n1000_%j.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/ABM/experiments/save_autocovariance/results/sdxl_ctau_n1000_%j.err

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
cd /sci/labs/orzuk/shaulytolk/ABM/experiments/save_autocovariance

echo "=== SDXL C_tau N=1000: $(hostname), GPU=$CUDA_VISIBLE_DEVICES ==="
echo "Start: $(date)"

RESUME_X0="../sdxl/results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_x0.npz"

python save_sdxl_C_tau.py \
    --n_samples 1000 \
    --resume_x0 "$RESUME_X0" \
    --clip_batch_size 4 \
    --vae_batch_size 4 \
    --score_batch_size 16 \
    --langevin_ckpt_interval 500

echo "Done: $(date)"
