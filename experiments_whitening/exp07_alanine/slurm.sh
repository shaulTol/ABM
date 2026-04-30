#!/bin/bash
#SBATCH --job-name=ala_whiten
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/ABM/experiments_whitening/exp07_alanine/slurm_%j.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/ABM/experiments_whitening/exp07_alanine/slurm_%j.err

export PATH=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env
NV_BASE=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NV_BASE/cublas/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/sci/labs/orzuk/shaulytolk/ABM:$PYTHONPATH

cd /sci/labs/orzuk/shaulytolk/ABM

python experiments_whitening/exp07_alanine/run.py
