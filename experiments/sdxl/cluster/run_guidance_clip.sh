#!/bin/bash
#SBATCH --job-name=gs_clip
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=1
#SBATCH --output=results_clip_da/guidance_gs_%a_%j.out
#SBATCH --error=results_clip_da/guidance_gs_%a_%j.err

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

# Array mapping: 0 → gs=1.0, 1 → gs=7.5
GS_VALUES=(1.0 7.5)
GS_NAMES=(guidance_gs1p0 guidance_gs7p5)

GS=${GS_VALUES[$SLURM_ARRAY_TASK_ID]}
NAME=${GS_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "=== Task $SLURM_ARRAY_TASK_ID: guidance_scale=$GS, name=$NAME ==="

# Stage 1: SSA + DA + entropy suite + gender analysis
python scripts/run_ssa_da_clip.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --name "$NAME" \
    --prompt "a superrealistic professional photograph of a person" \
    --guidance_scale "$GS" --gen_steps 50 \
    --height 1024 --width 1024 \
    --n_samples 200 --n_traj 200 \
    --T 500 --dt 0.001 --t_score 1 \
    --score_batch_size 16 \
    --clip_batch_size 1 \
    --output_dir results_clip_da

# Stage 2: Full correlation table
python scripts/full_correlation_table.py \
    --latents_npz "results_clip_da/clip_da_latents_${NAME}.npz" \
    --clip_da_npz "results_clip_da/clip_da_${NAME}.npz" \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --label "Guidance $GS" \
    --output_dir results_clip_da

echo "Done task $SLURM_ARRAY_TASK_ID: $(date)"
