#!/bin/bash
#SBATCH --job-name=da_ep
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=results_da_pcavda/da_%a_%j.out
#SBATCH --error=results_da_pcavda/da_%a_%j.err

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

mkdir -p results_da_pcavda

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    # Elderly w=7.5
    PROMPT="a portrait of an elderly person"
    NAME="elderly_gs7p5"
    CHECKPOINT="results_multiprompt_sweep/clip_da_latents_mp_p5_elderly_gs7p5.npz"
    RESUME_X0="results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_quick_x0.npz"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    # Person w=7.5
    PROMPT="a superrealistic professional photograph of a person"
    NAME="person_gs7p5"
    CHECKPOINT="results_guidance_sweep/clip_da_latents_sweep_gs7p5.npz"
    RESUME_X0="results_guidance_sweep/gen_sweep_gs7p5_quick_x0.npz"
fi

echo "=== DA task $SLURM_ARRAY_TASK_ID: $NAME ==="
echo "Prompt: $PROMPT"
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# Try resume from checkpoint first (skips Langevin); fall back to x0
if [ -f "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
    python scripts/run_ssa_da_clip.py \
        --model_id stabilityai/stable-diffusion-xl-base-1.0 \
        --name "$NAME" \
        --prompt "$PROMPT" \
        --guidance_scale 7.5 --gen_steps 50 \
        --height 1024 --width 1024 \
        --resume_checkpoint "$CHECKPOINT" \
        --n_samples 200 --n_traj 200 \
        --T 2000 --dt 0.0005 --t_score 1 \
        --snapshot_schedule log_and_fixed \
        --compute_entropy \
        --score_batch_size 16 \
        --clip_batch_size 4 \
        --output_dir results_da_pcavda
else
    echo "No checkpoint found, resuming from x0: $RESUME_X0"
    python scripts/run_ssa_da_clip.py \
        --model_id stabilityai/stable-diffusion-xl-base-1.0 \
        --name "$NAME" \
        --prompt "$PROMPT" \
        --guidance_scale 7.5 --gen_steps 50 \
        --height 1024 --width 1024 \
        --resume_x0 "$RESUME_X0" \
        --n_samples 200 --n_traj 200 \
        --T 2000 --dt 0.0005 --t_score 1 \
        --snapshot_schedule log_and_fixed \
        --compute_entropy \
        --score_batch_size 16 \
        --clip_batch_size 4 \
        --output_dir results_da_pcavda
fi

echo "Done task $SLURM_ARRAY_TASK_ID: $(date)"
