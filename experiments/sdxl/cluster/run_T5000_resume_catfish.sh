#!/bin/bash
#SBATCH --job-name=T5k_res_cat
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-21
#SBATCH --output=/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/results_T5000_uncomplete/T5k_resume_cat_%a_%j.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/results_T5000_uncomplete/T5k_resume_cat_%a_%j.err

# Mirror of run_T5000_resume.sh on the catfish partition. Runs the SAME array
# in parallel with the salmon array (44603227). Each task checks whether the
# target JSON already exists and exits 0 if so — so whichever partition gets
# the task done first wins and the other side skips.
#
# Written 2026-04-28.

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
# Catfish has ~22 GB GPU vs salmon's 48 GB — fight fragmentation:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl

mkdir -p results_T5000_uncomplete

CKPT_DIR=results_T5000

NAMES=(
    sweep_gs1p0_dt0p005_T5000
    sweep_gs3p0_dt0p005_T5000
    sweep_gs7p5_dt0p005_T5000
    sweep_gs15p0_dt0p005_T5000
    sweep_gs1p0_dt0p0025_T5000
    sweep_gs3p0_dt0p0025_T5000
    sweep_gs7p5_dt0p0025_T5000
    sweep_gs15p0_dt0p0025_T5000
    mp_p0_cat_gs1p0_dt0p005_T5000
    mp_p0_cat_gs3p0_dt0p005_T5000
    mp_p0_cat_gs7p5_dt0p005_T5000
    mp_p0_cat_gs15p0_dt0p005_T5000
    mp_p0_cat_gs1p0_dt0p0025_T5000
    mp_p0_cat_gs3p0_dt0p0025_T5000
    mp_p0_cat_gs7p5_dt0p0025_T5000
    mp_p0_cat_gs15p0_dt0p0025_T5000
    mp_p5_elderly_gs1p0_dt0p005_T5000
    mp_p5_elderly_gs3p0_dt0p005_T5000
    mp_p5_elderly_gs7p5_dt0p005_T5000
    mp_p5_elderly_gs15p0_dt0p005_T5000
    mp_p5_elderly_gs1p0_dt0p0025_T5000
    mp_p5_elderly_gs3p0_dt0p0025_T5000
)

PROMPTS=(
    "a superrealistic professional photograph of a person"
    "a superrealistic professional photograph of a person"
    "a superrealistic professional photograph of a person"
    "a superrealistic professional photograph of a person"
    "a superrealistic professional photograph of a person"
    "a superrealistic professional photograph of a person"
    "a superrealistic professional photograph of a person"
    "a superrealistic professional photograph of a person"
    "a photograph of a cat sitting on a windowsill"
    "a photograph of a cat sitting on a windowsill"
    "a photograph of a cat sitting on a windowsill"
    "a photograph of a cat sitting on a windowsill"
    "a photograph of a cat sitting on a windowsill"
    "a photograph of a cat sitting on a windowsill"
    "a photograph of a cat sitting on a windowsill"
    "a photograph of a cat sitting on a windowsill"
    "a portrait of an elderly person"
    "a portrait of an elderly person"
    "a portrait of an elderly person"
    "a portrait of an elderly person"
    "a portrait of an elderly person"
    "a portrait of an elderly person"
)

GS_VALUES=(1.0 3.0 7.5 15.0 1.0 3.0 7.5 15.0 1.0 3.0 7.5 15.0 1.0 3.0 7.5 15.0 1.0 3.0 7.5 15.0 1.0 3.0)
DT_VALUES=(0.005 0.005 0.005 0.005 0.0025 0.0025 0.0025 0.0025 0.005 0.005 0.005 0.005 0.0025 0.0025 0.0025 0.0025 0.005 0.005 0.005 0.005 0.0025 0.0025)

NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
PROMPT=${PROMPTS[$SLURM_ARRAY_TASK_ID]}
GS=${GS_VALUES[$SLURM_ARRAY_TASK_ID]}
DT=${DT_VALUES[$SLURM_ARRAY_TASK_ID]}
CKPT=${CKPT_DIR}/langevin_ckpt_${NAME}.npz
JSON_OUT=results_T5000_uncomplete/clip_da_${NAME}.json

echo "=== [catfish] Resume task $SLURM_ARRAY_TASK_ID: name=$NAME, dt=$DT, gs=$GS ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Ckpt: $CKPT"
echo "JSON: $JSON_OUT"
echo "Start: $(date)"

if [ -f "$JSON_OUT" ]; then
    echo "[skip] JSON already exists — salmon must have finished it first."
    exit 0
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi

python scripts/run_ssa_da_clip.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --name "$NAME" \
    --prompt "$PROMPT" \
    --guidance_scale "$GS" --gen_steps 50 \
    --height 1024 --width 1024 \
    --resume_checkpoint "$CKPT" \
    --n_samples 200 --n_traj 200 \
    --T 5000 --dt "$DT" --t_score 1 \
    --snapshot_schedule log_and_fixed \
    --compute_entropy \
    --score_batch_size 16 \
    --clip_batch_size 1 \
    --output_dir results_T5000_uncomplete

echo "Done task $SLURM_ARRAY_TASK_ID: $(date)"
