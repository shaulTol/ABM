#!/bin/bash
#SBATCH --job-name=mammal_v18a
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=mammal_v18a_%j.out
#SBATCH --error=mammal_v18a_%j.err

export PYTHONUNBUFFERED=1
export PATH=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env

NV_BASE=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NV_BASE/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_cupti/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cufft/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cusolver/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export HF_TOKEN_PATH=/dev/null

EXPDIR=/sci/labs/orzuk/shaulytolk/ABM/experiments/autoresearch
cd $EXPDIR

python -u gen_mammal_glasses.py \
    --n_images 200 \
    --no_scribble \
    --turbo_only \
    --prompt "Portrait of an anthropomorphic animal bride or groom" \
    --output_dir results_mammal_v18a

echo "Done: $(date)"
