#!/bin/bash
#SBATCH --job-name=da_ala_kde
#SBATCH --output=da_ala_kde_%j.out
#SBATCH --error=da_ala_kde_%j.err
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G

export PATH=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env

# Add all nvidia library paths (all in ssa_env now)
NV_BASE=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/nvidia
for d in cublas cuda_cupti cuda_nvrtc cuda_runtime cudnn cufft cusolver cusparse nvtx curand nccl nvjitlink; do
    export LD_LIBRARY_PATH=$NV_BASE/$d/lib:$LD_LIBRARY_PATH
done
export LD_LIBRARY_PATH=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

EXPDIR=/sci/labs/orzuk/shaulytolk/ABM/experiments/alanine

cd "$EXPDIR"

echo "Running KDE-DA for alanine dipeptide"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

python -u compute_alanine_kde_da.py

echo "End: $(date)"
