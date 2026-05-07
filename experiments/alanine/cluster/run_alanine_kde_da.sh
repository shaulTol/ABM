#!/bin/bash
#SBATCH --job-name=da_ala_kde
#SBATCH --output=da_ala_kde_%j.out
#SBATCH --error=da_ala_kde_%j.err
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G

export PATH=<CONDA_BIN>:$PATH
source <CONDA_SH>
conda activate <CONDA_ENV>

# Add all nvidia library paths (all in ssa_env now)
NV_BASE=<NV_SITE_PACKAGES>
for d in cublas cuda_cupti cuda_nvrtc cuda_runtime cudnn cufft cusolver cusparse nvtx curand nccl nvjitlink; do
    export LD_LIBRARY_PATH=$NV_BASE/$d/lib:$LD_LIBRARY_PATH
done
export LD_LIBRARY_PATH=<TORCH_LIB>:$LD_LIBRARY_PATH

EXPDIR=<PROJECT_DIR>/experiments/alanine

cd "$EXPDIR"

echo "Running KDE-DA for alanine dipeptide"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

python -u compute_alanine_kde_da.py

echo "End: $(date)"
