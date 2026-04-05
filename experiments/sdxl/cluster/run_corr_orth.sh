#!/bin/bash
#SBATCH --job-name=corr_orth
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=corr_orth_%j.out
#SBATCH --error=corr_orth_%j.err

# Rerun correlation tables with orthogonalized probes for all experiments.

export PYTHONUNBUFFERED=1

export PATH=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env

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

# Circle/Face experiments (dt=0.001 — the converged ones)
for COND in circle dps_face; do
    echo "=== ${COND} dt=0.001 ==="
    python scripts/full_correlation_table.py \
        --latents_npz results_clip_da/clip_da_latents_clip_cn_${COND}_dt0.001.npz \
        --clip_da_npz results_clip_da/clip_da_clip_cn_${COND}_dt0.001.npz \
        --label "${COND} dt=0.001" \
        --output_dir results_clip_da
    echo ""
done

# Autoresearch experiments
for EXP in egg square street; do
    echo "=== ${EXP} ==="
    python scripts/full_correlation_table.py \
        --latents_npz results_autoresearch/clip_da_latents_autoresearch_${EXP}.npz \
        --clip_da_npz results_autoresearch/clip_da_autoresearch_${EXP}.npz \
        --label "${EXP}" \
        --output_dir results_autoresearch
    echo ""
done

echo "=== All done ==="
