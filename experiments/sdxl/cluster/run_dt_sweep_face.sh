#!/bin/bash
#SBATCH --job-name=dt_sweep_%a
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=dt_sweep_%A_%a.out
#SBATCH --error=dt_sweep_%A_%a.err
#SBATCH --array=0-3

# dt sweep on DPS face: 0.0001, 0.0005, 0.001, 0.005
# All with T=500, N=200. Check if DA1 converges at larger dt without SSA artifact.

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

DT_VALUES=(0.0001 0.0005 0.001 0.005)
DT=${DT_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "=== dt sweep: DPS face, dt=${DT} ==="
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Date: $(date)"

python scripts/compute_sdxl_ssa_v2.py \
    --model_id "stabilityai/sdxl-turbo" \
    --name "dt_sweep_face_dt${DT}" \
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
    --control_image_path "data/dps_face_scribble.png" \
    --controlnet_scale 0.5 \
    --compute_entropy --entropy_k 10 \
    --output_dir results_dt_sweep \
    --seed 42

echo "=== Done: dt=${DT} ==="
echo "Date: $(date)"
