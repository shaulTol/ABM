#!/bin/bash
#SBATCH --job-name=ps_dt005_T5k
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/results_guidance_sweep/dt005T5k_%a_%j.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/results_guidance_sweep/dt005T5k_%a_%j.err

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

GS_VALUES=(1.0 3.0 7.5 15.0)
GS_LABELS=(1p0 3p0 7p5 15p0)

GS=${GS_VALUES[$SLURM_ARRAY_TASK_ID]}
GS_LABEL=${GS_LABELS[$SLURM_ARRAY_TASK_ID]}

NAME="sweep_gs${GS_LABEL}_dt0p005_T5000"
CKPT="results_guidance_sweep/clip_da_latents_sweep_gs${GS_LABEL}_dt0p005.npz"

echo "=== Task $SLURM_ARRAY_TASK_ID: gs=$GS, name=$NAME ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint: $CKPT"
echo "Start: $(date)"

python scripts/run_ssa_da_clip.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --name "$NAME" \
    --prompt "a superrealistic professional photograph of a person" \
    --guidance_scale "$GS" --gen_steps 50 \
    --height 1024 --width 1024 \
    --resume_x0 "$CKPT" \
    --n_samples 200 --n_traj 200 \
    --T 5000 --dt 0.005 --t_score 1 \
    --ssa_only \
    --snapshot_schedule log_and_fixed \
    --score_batch_size 16 \
    --clip_batch_size 1 \
    --no_save_checkpoint \
    --output_dir results_guidance_sweep

echo "Done task $SLURM_ARRAY_TASK_ID: $(date)"
