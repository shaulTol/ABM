#!/bin/bash
#SBATCH --job-name=T5k_dt0025
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11
#SBATCH --output=/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/results_T5000/T5k_dt0025_%a_%j.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/results_T5000/T5k_dt0025_%a_%j.err

# T=5000, dt=0.0025 (T_phys = 12.5). Three prompts (person, cat, elderly) ×
# four guidance scales (1.0, 3.0, 7.5, 15.0) = 12 jobs.
#
# Reuses generated x0 latents from the original T=2000 runs via --resume_x0.
# Saves the full Langevin checkpoint (snapshots fp16 — see save site in
# run_ssa_da_clip.py). Runs full DA (no --ssa_only) and entropy suite.
#
# Written 2026-04-26.

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

mkdir -p results_T5000

PROMPT_IDX=$((SLURM_ARRAY_TASK_ID / 4))
GS_IDX=$((SLURM_ARRAY_TASK_ID % 4))

GS_VALUES=(1.0 3.0 7.5 15.0)
GS_LABELS=(gs1p0 gs3p0 gs7p5 gs15p0)
GS=${GS_VALUES[$GS_IDX]}
GS_LABEL=${GS_LABELS[$GS_IDX]}

DT=0.0025
DT_LABEL=dt0p0025
T_STEPS=5000

case $PROMPT_IDX in
  0)
    PROMPT="a superrealistic professional photograph of a person"
    X0_PATH="results_guidance_sweep/gen_sweep_${GS_LABEL}_quick_x0.npz"
    NAME="sweep_${GS_LABEL}_${DT_LABEL}_T5000"
    ;;
  1)
    PROMPT="a photograph of a cat sitting on a windowsill"
    X0_PATH="results_multiprompt_sweep/gen_mp_p0_cat_${GS_LABEL}_quick_x0.npz"
    NAME="mp_p0_cat_${GS_LABEL}_${DT_LABEL}_T5000"
    ;;
  2)
    PROMPT="a portrait of an elderly person"
    X0_PATH="results_multiprompt_sweep/gen_mp_p5_elderly_${GS_LABEL}_quick_x0.npz"
    NAME="mp_p5_elderly_${GS_LABEL}_${DT_LABEL}_T5000"
    ;;
  *)
    echo "Bad PROMPT_IDX=$PROMPT_IDX" >&2; exit 1 ;;
esac

echo "=== T=$T_STEPS, dt=$DT, gs=$GS, prompt='$PROMPT' (task $SLURM_ARRAY_TASK_ID) ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "x0:   $X0_PATH"
echo "name: $NAME"
echo "Start: $(date)"

python scripts/run_ssa_da_clip.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --name "$NAME" \
    --prompt "$PROMPT" \
    --guidance_scale "$GS" --gen_steps 50 \
    --height 1024 --width 1024 \
    --resume_x0 "$X0_PATH" \
    --n_samples 200 --n_traj 200 \
    --T "$T_STEPS" --dt "$DT" --t_score 1 \
    --snapshot_schedule log_and_fixed \
    --compute_entropy \
    --score_batch_size 16 \
    --clip_batch_size 4 \
    --output_dir results_T5000

echo "Done task $SLURM_ARRAY_TASK_ID: $(date)"
