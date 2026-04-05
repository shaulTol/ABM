#!/bin/bash
#SBATCH --job-name=clip_dt_%a
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=clip_dt_%A_%a.out
#SBATCH --error=clip_dt_%A_%a.err
#SBATCH --array=0-3

# dt=0.001 and dt=0.005 for circle + dps_face (full CLIP-mode SSA+DA).
# Array: 0=circle dt=0.001, 1=face dt=0.001, 2=circle dt=0.005, 3=face dt=0.005
# Compare with 44393548 (dt=0.0001) to find sweet spot for DA convergence.

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
export WANDB_DIR=/tmp/wandb_${SLURM_JOB_ID}
mkdir -p $WANDB_DIR

cd /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl

NAMES=("clip_cn_circle_dt0.001" "clip_cn_dps_face_dt0.001" "clip_cn_circle_dt0.005" "clip_cn_dps_face_dt0.005")
CONTROL_IMAGES=("data/circle_scribble.png" "data/dps_face_scribble.png" "data/circle_scribble.png" "data/dps_face_scribble.png")
DT_VALUES=(0.001 0.001 0.005 0.005)

NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
CTRL=${CONTROL_IMAGES[$SLURM_ARRAY_TASK_ID]}
DT=${DT_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "=== CLIP-mode SSA+DA: ${NAME}, dt=${DT} ==="
echo "Date: $(date)"

python scripts/run_ssa_da_clip.py \
    --model_id "stabilityai/sdxl-turbo" \
    --name "${NAME}" \
    --prompt "a superrealistic professional photograph of" \
    --guidance_scale 0.0 \
    --gen_steps 2 \
    --height 512 --width 512 \
    --n_samples 200 \
    --n_traj 200 \
    --T 500 --dt ${DT} --t_score 1 \
    --score_batch_size 128 \
    --gen_batch_size 4 \
    --controlnet_model_id "xinsir/controlnet-scribble-sdxl-1.0" \
    --control_image_path "${CTRL}" \
    --controlnet_scale 0.5 \
    --clip_model openai/clip-vit-large-patch14 \
    --compute_entropy --entropy_k 10 \
    --also_compute_lc \
    --output_dir results_clip_da \
    --seed 42

echo "=== Done: ${NAME} ==="
echo "Date: $(date)"
