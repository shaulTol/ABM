#!/bin/bash
#SBATCH --job-name=ssa_v27
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=results_clip_da/ssa_v27_%a_%j.out
#SBATCH --error=results_clip_da/ssa_v27_%a_%j.err

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

mkdir -p results_clip_da

PROMPT="Anthropomorphic animal in human clothes, profile view, city street"

# Array 0: v27 Turbo (512x512, gs=0.0, 4 steps)
# Array 1: v27c Base (1024x1024, gs=5.0, 50 steps)
# Both use --ssa_only + --compute_entropy. SSA compared in CLIP 768D space.

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "=== v27 Turbo (512x512, gs=0.0) ==="
    python scripts/run_ssa_da_clip.py \
        --model_id stabilityai/sdxl-turbo \
        --name "mammal_v27_turbo" \
        --prompt "$PROMPT" \
        --guidance_scale 0.0 --gen_steps 4 \
        --height 512 --width 512 \
        --n_samples 200 --n_traj 200 \
        --T 1000 --dt 0.0005 --t_score 1 \
        --ssa_min_lag 100 \
        --ssa_only \
        --compute_entropy \
        --score_batch_size 64 \
        --clip_batch_size 4 \
        --output_dir results_clip_da
else
    echo "=== v27c Base (1024x1024, gs=5.0) ==="
    python scripts/run_ssa_da_clip.py \
        --model_id stabilityai/stable-diffusion-xl-base-1.0 \
        --name "mammal_v27c_base" \
        --prompt "$PROMPT" \
        --guidance_scale 5.0 --gen_steps 50 \
        --height 1024 --width 1024 \
        --n_samples 200 --n_traj 200 \
        --T 1000 --dt 0.0005 --t_score 1 \
        --ssa_min_lag 100 \
        --ssa_only \
        --compute_entropy \
        --score_batch_size 16 \
        --clip_batch_size 1 \
        --output_dir results_clip_da

fi

echo "Done task $SLURM_ARRAY_TASK_ID: $(date)"
