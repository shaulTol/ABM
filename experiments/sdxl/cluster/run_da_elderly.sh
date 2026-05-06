#!/bin/bash
#SBATCH --job-name=da_eld_fc
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=results_da_pcavda/da_eld_fc_%j.out
#SBATCH --error=results_da_pcavda/da_eld_fc_%j.err

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
cd /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/elderly_da

mkdir -p results_da_pcavda

PROMPT="a portrait of an elderly person"
NAME="elderly_gs7p5_fc_dt0p0025"
RESUME_X0="../guidance_sweep/results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_quick_x0.npz"

echo "=== DA elderly free-conv stopping ==="
echo "Prompt: $PROMPT"
echo "dt=0.0025, T=2000 (max phys time = 5.0), M=1"
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# dt=0.0025 with T=2000 -> 5x further in physical time than the dt=0.0005
# baseline. Reuses the same x0 latents; Langevin is re-run with the new dt.
python ../guidance_sweep/run_ssa_da_clip.py \
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
    --output_dir results_da_pcavda

echo "Done: $(date)"
