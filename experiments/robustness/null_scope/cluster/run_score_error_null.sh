#!/bin/bash
#SBATCH --job-name=score_error_null
#SBATCH --partition=glacier
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --output=score_error_null_%j.out
#SBATCH --error=score_error_null_%j.err

# Exceedance of the analytic edge under injected score error (appendix "Scope of
# the analytic null", figure fig:rebuttal_null_score_error). Pure numpy, CPU only.
# Grid: (d, N) = (40, 2000) and (768, 200); alpha in {0, 0.1, 0.25, 0.5, 1, 2};
# conservative and nonconservative error fields; 100 replicates per cell.
# First runs compare_overlapping_pairs.py (dependent overlapping pairs vs
# independent pairs, same appendix), whose results/overlapping_pair_null.json
# the score-error grid reads.
# Writes results/score_error_null_spectrum.json, _spectra.npz and _table.md.
# Seeds depend only on (d, replicate, alpha, family), so --workers does not
# change the numbers.

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export PATH=<CONDA_BIN>:$PATH
source <CONDA_SH>
conda activate <CONDA_ENV>

cd <PROJECT_DIR>/experiments/robustness/null_scope

python -u compare_overlapping_pairs.py --out-dir results || exit 1
python -u score_error_null_spectrum.py --checks-only --out-dir results || exit 1
python -u score_error_null_spectrum.py --workers 16 --out-dir results
