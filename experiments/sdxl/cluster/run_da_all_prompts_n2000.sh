#!/bin/bash
#SBATCH --job-name=da2k
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7
#SBATCH --output=results_da_pcavda/da_n2000_%a_%j.out
#SBATCH --error=results_da_pcavda/da_n2000_%a_%j.err

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

# 8 prompts at w=7.5 — N=2000 with outlier filtering
PROMPTS=(
    "a photograph of a cat sitting on a windowsill"
    "a landscape photograph of mountains at sunset"
    "a still life painting of flowers in a vase"
    "a street photograph of a busy city intersection"
    "a photograph of a plate of food at a restaurant"
    "a portrait of an elderly person"
    "a photograph of a car parked on a street"
    "a superrealistic professional photograph of a person"
)
LABELS=(cat mountains flowers street food elderly car person)
X0_PATHS=(
    "results_multiprompt_sweep/gen_mp_p0_cat_gs7p5_x0.npz"
    "results_multiprompt_sweep/gen_mp_p1_mountains_gs7p5_x0.npz"
    "results_multiprompt_sweep/gen_mp_p2_flowers_gs7p5_x0.npz"
    "results_multiprompt_sweep/gen_mp_p3_street_gs7p5_x0.npz"
    "results_multiprompt_sweep/gen_mp_p4_food_gs7p5_x0.npz"
    "results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_x0.npz"
    "results_multiprompt_sweep/gen_mp_p6_car_gs7p5_x0.npz"
    "results_guidance_sweep/gen_sweep_gs7p5_x0.npz"
)

IDX=$SLURM_ARRAY_TASK_ID
PROMPT="${PROMPTS[$IDX]}"
LABEL="${LABELS[$IDX]}"
RESUME_X0="${X0_PATHS[$IDX]}"
NAME="${LABEL}_gs7p5_n2000"

echo "=== DA N=2000 task $IDX: $LABEL ==="
echo "Prompt: $PROMPT"
echo "x0: $RESUME_X0"
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

python scripts/run_ssa_da_clip.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --name "$NAME" \
    --prompt "$PROMPT" \
    --guidance_scale 7.5 --gen_steps 50 \
    --height 1024 --width 1024 \
    --resume_x0 "$RESUME_X0" \
    --n_samples 2000 --n_traj 2000 \
    --T 2000 --dt 0.001 --t_score 1 \
    --snapshot_schedule log_and_fixed \
    --clip_outlier_sigma 2.0 \
    --no_save_checkpoint \
    --clip_batch_size 4 \
    --score_batch_size 16 \
    --output_dir results_da_pcavda

echo "Done task $IDX ($LABEL): $(date)"
