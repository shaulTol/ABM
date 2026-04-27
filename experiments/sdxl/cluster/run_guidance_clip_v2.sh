#!/bin/bash
#SBATCH --job-name=gs_clip_v2
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=results_clip_da/guidance_gs_v2_%a_%j.out
#SBATCH --error=results_clip_da/guidance_gs_v2_%a_%j.err

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
# Reuses x0 from previous guidance runs — finer dt, longer T, later lags
GS_VALUES=(1.0 7.5)
GS_LABELS=(1p0 7p5)

GS=${GS_VALUES[$SLURM_ARRAY_TASK_ID]}
GS_LABEL=${GS_LABELS[$SLURM_ARRAY_TASK_ID]}

echo "=== Task $SLURM_ARRAY_TASK_ID: gs=$GS, label=$GS_LABEL ==="

python scripts/run_ssa_da_clip.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --name "guidance_gs${GS_LABEL}_v2" \
    --prompt "a superrealistic professional photograph of a person" \
    --guidance_scale "$GS" --gen_steps 50 \
    --height 1024 --width 1024 \
    --resume_x0 "results_clip_da/clip_da_latents_guidance_gs${GS_LABEL}.npz" \
    --n_samples 200 --n_traj 200 \
    --T 1000 --dt 0.0005 --t_score 1 \
    --ssa_min_lag 100 \
    --ssa_only \
    --compute_entropy \
    --score_batch_size 16 \
    --clip_batch_size 1 \
    --output_dir results_clip_da

echo "Done task $SLURM_ARRAY_TASK_ID: $(date)"
