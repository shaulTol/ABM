#!/bin/bash
#SBATCH --job-name=score_robustness_merge
#SBATCH --partition=glacier
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=score_robustness_merge_%j.out
#SBATCH --error=score_robustness_merge_%j.err

# Run after cluster/run_score_robustness.sh has finished all 50 shards.
#  --da_only: DA1 rotation on the five highest-MI mixtures -> results/score_robustness_full_fc_da.json
#  --merge:   merge the shards, Spearman(SSA, MI) per level -> results/score_robustness_full_fc.{json,md}
# CPU on purpose: no long autocorrelation loop here, and CPU keeps DA bitwise reproducible.

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

export PATH=<CONDA_BIN>:$PATH
source <CONDA_SH>
conda activate <CONDA_ENV>

cd <PROJECT_DIR>/experiments/robustness/score_error

python -u compute_score_robustness.py --da_only --run_tag full_fc
python -u compute_score_robustness.py --merge --run_tag full_fc
