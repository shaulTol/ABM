#!/bin/bash
#SBATCH --job-name=four_mode_flat_top
#SBATCH --partition=glacier
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=four_mode_flat_top_%j.out
#SBATCH --error=four_mode_flat_top_%j.err

# Appendix "Anisotropy, degenerate slow subspaces, and feature visibility":
#  - 10D four-mode circular mixture vs matched isotropic Gaussian (SSA, DA subspace)
#    -> results/p3_scope/p3_{results,arrays,table}_full_d10_a_fc.*
#  - flat-top density exp[-sum (|x_i|/3)^6] in d=10 vs its matched Gaussian
#    (SSA, dip and Silverman tests) -> results/p11_unimodal/
# SSA uses the paper's 1/N first-crossing rule (SSA_STOPPING=first, the default).

export PYTHONUNBUFFERED=1
export SSA_STOPPING=first

export PATH=<CONDA_BIN>:$PATH
source <CONDA_SH>
conda activate <CONDA_ENV>

cd <PROJECT_DIR>/experiments/robustness/geometry

python -u p3_scope_controls.py --checks || exit 1
python -u p3_scope_controls.py --embed_dim 10 --fast_var 0.05 --only a \
    --tag full_d10_a_fc --output_dir results/p3_scope
python -u p11_unimodal_high_ssa.py --checks || exit 1
python -u p11_unimodal_high_ssa.py --dims 10 --cells 10 --tag fc \
    --output_dir results/p11_unimodal
