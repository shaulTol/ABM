#!/bin/bash
#SBATCH --job-name=nf_sdxl_a
#SBATCH --partition=catfish
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=nf_sdxl_a_%j.out
#SBATCH --error=nf_sdxl_a_%j.err

export PATH=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env

REPODIR=/sci/labs/orzuk/shaulytolk/ABM
cd "$REPODIR"

echo "Running SDXL N=1000 noise floor (analytic, no GPU)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

python -u experiments/noise_floor/sdxl/run_sdxl_noise_floor_n1000.py

echo "End: $(date)"
