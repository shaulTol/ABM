#!/bin/bash
#SBATCH --job-name=da_eld_fc
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/da_eld_fc_%j.out
#SBATCH --error=results/da_eld_fc_%j.err

export PYTHONUNBUFFERED=1
export PATH=<CONDA_BIN>:$PATH
source <CONDA_SH>
conda activate <CONDA_ENV>
NV_BASE=<NV_SITE_PACKAGES>
export LD_LIBRARY_PATH=$NV_BASE/cublas/lib:$NV_BASE/cuda_cupti/lib:$NV_BASE/cuda_nvrtc/lib:$NV_BASE/cuda_runtime/lib:$NV_BASE/cudnn/lib:$NV_BASE/cufft/lib:$NV_BASE/cusolver/lib:$NV_BASE/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=<USER_SITE_PACKAGES>/nccl/lib:<USER_SITE_PACKAGES>/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=<TORCH_LIB>:$LD_LIBRARY_PATH
export HF_HOME=<HF_CACHE>
export HF_TOKEN_PATH=/dev/null
cd <PROJECT_DIR>/experiments/sdxl

mkdir -p results

PROMPT="a portrait of an elderly person"
NAME="elderly_gs7p5_fc_dt0p0025"
RESUME_X0="results/gen_mp_p5_elderly_gs7p5_quick_x0.npz"

echo "=== DA elderly free-conv stopping ==="
echo "Prompt: $PROMPT"
echo "dt=0.0025, T=2000 (max phys time = 5.0), M=1"
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# dt=0.0025 with T=2000 -> 5x further in physical time than the dt=0.0005
# baseline. Reuses the same x0 latents; Langevin is re-run with the new dt.
python run_ssa_da_clip.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --name "$NAME" \
    --prompt "$PROMPT" \
    --guidance_scale 7.5 --gen_steps 50 \
    --height 1024 --width 1024 \
    --resume_x0 "$RESUME_X0" \
    --n_samples 200 --n_traj 200 \
    --T 2000 --dt 0.0025 --t_score 1 \
    --snapshot_schedule log_and_fixed \
    --freeconv_stop_M 1 \
    --compute_entropy \
    --score_batch_size 16 \
    --clip_batch_size 4 \
    --output_dir results

echo "Done: $(date)"
