#!/bin/bash
#SBATCH --job-name=corr_tbl
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=corr_tbl_%j.out
#SBATCH --error=corr_tbl_%j.err

# Full correlation tables for all 6 runs (2 conditions x 3 dt values).
# ~3 min per run on catfish (VAE decode + CLIP encode only, no Langevin).

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

# dt=0.0001: circle and face
for COND in circle dps_face; do
    echo "=== ${COND} dt=0.0001 ==="
    python scripts/full_correlation_table.py \
        --latents_npz results_clip_da/clip_da_latents_clip_cn_${COND}.npz \
        --clip_da_npz results_clip_da/clip_da_clip_cn_${COND}.npz \
        --lc_da_npz results_controlnet/sdxl_da_sdxl_ssa_cn_${COND}.npz \
        --model_id stabilityai/sdxl-turbo \
        --label "${COND} dt=0.0001" \
        --output_dir results_clip_da
    echo ""
done

# dt=0.001 and dt=0.005
for DT in 0.001 0.005; do
    for COND in circle dps_face; do
        echo "=== ${COND} dt=${DT} ==="
        python scripts/full_correlation_table.py \
            --latents_npz results_clip_da/clip_da_latents_clip_cn_${COND}_dt${DT}.npz \
            --clip_da_npz results_clip_da/clip_da_clip_cn_${COND}_dt${DT}.npz \
            --model_id stabilityai/sdxl-turbo \
            --label "${COND} dt=${DT}" \
            --output_dir results_clip_da
        echo ""
    done
done

echo "=== All done ==="
