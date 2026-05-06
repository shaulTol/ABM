#!/bin/bash
#SBATCH --job-name=sdxl_iso_sanity
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=/sci/labs/orzuk/shaulytolk/ABM/experiments/mc_dip_validation/sdxl_iso_null_via_late_t/results/sanity_%j.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/ABM/experiments/mc_dip_validation/sdxl_iso_null_via_late_t/results/sanity_%j.err

# Single-trajectory-batch sanity check before launching the array job.
# Runs ONE task with full N_per_task=50 trajectories at t_null=999, T=2000,
# guidance_scale=7.5. Output: results/task_999.npz (uses task_id=999 sentinel
# so it doesn't collide with any later array task IDs 0..15).
#
# Pass criteria (verify in .out before launching array):
#   1. Initial latent norm matches sigma_{t_null} * sqrt(65536) ~ 256
#      (logged as "||x_init|| mean ~XX.XX (expected ~YY.YY)").
#   2. Trajectory norm doesn't collapse — final ||x_tau|| should remain in
#      the same ballpark as ||x_0|| (OU-like) rather than growing or shrinking
#      multiple orders of magnitude.
#   3. z_tau CLIP norms ~1.0 (L2-normalized as expected).
#   4. subset lam_max << 9.96e-3 (real DA1 observed). If lam_max is comparable
#      or higher than ~3e-3, the null is broken (still has metastability).

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

cd /sci/labs/orzuk/shaulytolk/ABM/experiments/mc_dip_validation/sdxl_iso_null_via_late_t
mkdir -p results

echo "=== SDXL iso-null sanity (task_id=999) ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

python run_sdxl_iso_null.py \
    --task_id 999 \
    --t_null 999 \
    --N_per_task 50 \
    --T 2000 \
    --dt 0.0025 \
    --guidance_scale 7.5 \
    --base_seed 9000 \
    --output_dir results \
    --score_batch_size 16 \
    --clip_batch_size 4 \
    --var_f_mode sigma_t_null_sq

echo "Done: $(date)"
