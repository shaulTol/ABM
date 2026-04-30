#!/bin/bash
#SBATCH --job-name=sdxl_whiten
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --array=0-11
#SBATCH --output=/sci/labs/orzuk/shaulytolk/ABM/experiments_whitening/exp05_sdxl_guidance/slurm_%A_%a.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/ABM/experiments_whitening/exp05_sdxl_guidance/slurm_%A_%a.err

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
export PYTHONPATH=/sci/labs/orzuk/shaulytolk/ABM:$PYTHONPATH
export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export HF_TOKEN_PATH=/dev/null

cd /sci/labs/orzuk/shaulytolk/ABM

python -u experiments_whitening/exp05_sdxl_guidance/run_whitened_langevin.py \
    --cell_idx $SLURM_ARRAY_TASK_ID \
    --x0_dir /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/results_multiprompt_sweep \
    --guidance_sweep_dir /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/results_guidance_sweep \
    --output_dir /sci/labs/orzuk/shaulytolk/ABM/experiments_whitening/exp05_sdxl_guidance/results
