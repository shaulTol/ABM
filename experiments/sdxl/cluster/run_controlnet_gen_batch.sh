#!/bin/bash
#SBATCH --job-name=cn_gen_bat
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --output=cn_gen_bat_%A_%a.out
#SBATCH --error=cn_gen_bat_%A_%a.err

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

# Array config
case $SLURM_ARRAY_TASK_ID in
    0)
        PROMPT="a superrealistic professional photograph of a woman from"
        OUTDIR="output_controlnet_black_woman"
        CONTROL="data/black_512.png"
        CS=0.5
        ;;
    1)
        PROMPT="a superrealistic professional photograph of a man from"
        OUTDIR="output_controlnet_black_man_from"
        CONTROL="data/black_512.png"
        CS=0.5
        ;;
    2)
        PROMPT="a superrealistic professional photograph of a person from"
        OUTDIR="output_controlnet_face_person"
        CONTROL="data/dps_face_scribble.png"
        CS=0.8
        ;;
esac

echo "=== Task $SLURM_ARRAY_TASK_ID: $PROMPT ==="
echo "Control: $CONTROL (cs=$CS)"
echo "Output: $OUTDIR"

python scripts/generate_controlnet_samples.py \
    --model_id "stabilityai/sdxl-turbo" \
    --control_image "$CONTROL" \
    --prompt "$PROMPT" \
    --n 100 \
    --controlnet_scale $CS \
    --guidance_scale 0.0 \
    --steps 4 \
    --output_dir "$OUTDIR" \
    --seed 42

echo "Done: $(date)"
