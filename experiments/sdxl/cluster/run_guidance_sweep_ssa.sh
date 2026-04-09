#!/bin/bash
#SBATCH --job-name=gs_sweep_ssa
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=results_guidance_sweep/ssa_%a_%j.out
#SBATCH --error=results_guidance_sweep/ssa_%a_%j.err

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

mkdir -p results_guidance_sweep

# Array mapping: 0-3 → guidance scales (must match Phase 1)
GS_VALUES=(1.0 3.0 7.5 15.0)
GS_LABELS=(gs1p0 gs3p0 gs7p5 gs15p0)

GS=${GS_VALUES[$SLURM_ARRAY_TASK_ID]}
GS_LABEL=${GS_LABELS[$SLURM_ARRAY_TASK_ID]}

echo "=== Task $SLURM_ARRAY_TASK_ID: gs=$GS, label=$GS_LABEL ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# Resume from Phase 1 x0 latents — use first 200 for Langevin trajectories
python scripts/run_ssa_da_clip.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --name "sweep_${GS_LABEL}" \
    --prompt "a superrealistic professional photograph of a person" \
    --guidance_scale "$GS" --gen_steps 50 \
    --height 1024 --width 1024 \
    --resume_x0 "results_guidance_sweep/gen_sweep_${GS_LABEL}_quick_x0.npz" \
    --n_samples 200 --n_traj 200 \
    --T 2000 --dt 0.0005 --t_score 1 \
    --snapshot_schedule log_and_fixed \
    --ssa_only \
    --compute_entropy \
    --score_batch_size 16 \
    --clip_batch_size 4 \
    --output_dir results_guidance_sweep

echo "Done task $SLURM_ARRAY_TASK_ID: $(date)"
