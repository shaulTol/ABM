#!/bin/bash
#SBATCH --job-name=whiten_ssa
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-31
#SBATCH --output=/sci/labs/orzuk/shaulytolk/ABM/experiments/ssa_truncation_redef/results/whiten_%a_%j.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/ABM/experiments/ssa_truncation_redef/results/whiten_%a_%j.err

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

EXPDIR=/sci/labs/orzuk/shaulytolk/ABM/experiments/ssa_truncation_redef
SDXLDIR=/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl
cd /sci/labs/orzuk/shaulytolk/ABM

mkdir -p $EXPDIR/results/whitened

# Map array ID to checkpoint
CKPT_NAMES=(
    "results_guidance_sweep/langevin_ckpt_sweep_gs1p0"
    "results_guidance_sweep/langevin_ckpt_sweep_gs3p0"
    "results_guidance_sweep/langevin_ckpt_sweep_gs7p5"
    "results_guidance_sweep/langevin_ckpt_sweep_gs15p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p0_cat_gs1p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p0_cat_gs3p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p0_cat_gs7p5"
    "results_multiprompt_sweep/langevin_ckpt_mp_p0_cat_gs15p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p1_mountains_gs1p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p1_mountains_gs3p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p1_mountains_gs7p5"
    "results_multiprompt_sweep/langevin_ckpt_mp_p1_mountains_gs15p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p2_flowers_gs1p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p2_flowers_gs3p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p2_flowers_gs7p5"
    "results_multiprompt_sweep/langevin_ckpt_mp_p2_flowers_gs15p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p3_street_gs1p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p3_street_gs3p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p3_street_gs7p5"
    "results_multiprompt_sweep/langevin_ckpt_mp_p3_street_gs15p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p4_food_gs1p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p4_food_gs3p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p4_food_gs7p5"
    "results_multiprompt_sweep/langevin_ckpt_mp_p4_food_gs15p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p5_elderly_gs1p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p5_elderly_gs3p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p5_elderly_gs7p5"
    "results_multiprompt_sweep/langevin_ckpt_mp_p5_elderly_gs15p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p6_car_gs1p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p6_car_gs3p0"
    "results_multiprompt_sweep/langevin_ckpt_mp_p6_car_gs7p5"
    "results_multiprompt_sweep/langevin_ckpt_mp_p6_car_gs15p0"
)

RUN_NAMES=(
    "person_gs1p0" "person_gs3p0" "person_gs7p5" "person_gs15p0"
    "cat_gs1p0" "cat_gs3p0" "cat_gs7p5" "cat_gs15p0"
    "mountains_gs1p0" "mountains_gs3p0" "mountains_gs7p5" "mountains_gs15p0"
    "flowers_gs1p0" "flowers_gs3p0" "flowers_gs7p5" "flowers_gs15p0"
    "street_gs1p0" "street_gs3p0" "street_gs7p5" "street_gs15p0"
    "food_gs1p0" "food_gs3p0" "food_gs7p5" "food_gs15p0"
    "elderly_gs1p0" "elderly_gs3p0" "elderly_gs7p5" "elderly_gs15p0"
    "car_gs1p0" "car_gs3p0" "car_gs7p5" "car_gs15p0"
)

CKPT="${SDXLDIR}/${CKPT_NAMES[$SLURM_ARRAY_TASK_ID]}.npz"
NAME="${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}"

echo "=== Whitened SSA task $SLURM_ARRAY_TASK_ID: $NAME ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint: $CKPT"
echo "Start: $(date)"

python -u $EXPDIR/code/recompute_whitened_ssa.py \
    --checkpoint "$CKPT" \
    --name "$NAME" \
    --dt 0.0005 \
    --alpha 0.95 \
    --clip_batch_size 4 \
    --output_dir "$EXPDIR/results/whitened"

echo "Done task $SLURM_ARRAY_TASK_ID: $(date)"
