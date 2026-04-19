#!/bin/bash
#SBATCH --job-name=da_eld1k
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=results_da_pcavda/da_elderly_n1000_%j.out
#SBATCH --error=results_da_pcavda/da_elderly_n1000_%j.err

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

echo "=== DA elderly N=1000, dt=0.005, T=1000 ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "Start epoch: $(date +%s)"
START_EPOCH=$(date +%s)

python -u scripts/run_ssa_da_clip.py \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --name "elderly_gs7p5_n1000" \
    --prompt "a portrait of an elderly person" \
    --guidance_scale 7.5 --gen_steps 50 \
    --height 1024 --width 1024 \
    --resume_x0 "results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_x0.npz" \
    --n_samples 1000 --n_traj 1000 \
    --T 1000 --dt 0.005 --t_score 1 \
    --snapshot_schedule log_and_fixed \
    --clip_outlier_sigma 2.0 \
    --clip_batch_size 4 \
    --score_batch_size 16 \
    --output_dir results_da_pcavda

END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)
echo ""
echo "=== TIMING SUMMARY ==="
echo "End: $(date)"
echo "Wall time: ${ELAPSED}s (${HOURS}h)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Job ID: $SLURM_JOB_ID"
