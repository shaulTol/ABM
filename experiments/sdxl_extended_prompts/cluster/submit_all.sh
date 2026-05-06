#!/bin/bash
# Submit all three phases with SLURM dependencies.
# Phase 1a (quick gen) and 1b (full gen) run in parallel.
# Phase 2 (SSA) depends on Phase 1a only (needs x0 latents).
#
# Usage: bash cluster/submit_all.sh
# Run from: /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl_extended_prompts/cluster/

set -e

echo "=== Submitting Extended Prompts Sweep ==="
echo "6 prompts × 4 guidance scales = 24 cells per phase"
echo ""

# Phase 1a: Quick gen (200 samples → x0 latents)
JOB_QUICK=$(sbatch --parsable run_ext_quick.sh)
echo "Phase 1a (quick gen):  job $JOB_QUICK"

# Phase 1b: Full gen (2000 samples → entropy)
JOB_GEN=$(sbatch --parsable run_ext_gen.sh)
echo "Phase 1b (full gen):   job $JOB_GEN"

# Phase 2: SSA (depends on Phase 1a completing)
JOB_SSA=$(sbatch --parsable --dependency=afterok:${JOB_QUICK} run_ext_ssa.sh)
echo "Phase 2  (SSA):        job $JOB_SSA (depends on $JOB_QUICK)"

echo ""
echo "Monitor: squeue -u \$USER"
echo "Phase 1a: ~30 min/cell (200 images × 50 steps)"
echo "Phase 1b: ~5.5 h/cell (2000 images × 50 steps)"
echo "Phase 2:  ~13 h/cell (2000 Langevin steps × 200 traj)"
echo ""
echo "Total estimated: ~18.5 h/cell × 24 cells / 24 parallel = ~18.5h wall-clock"
echo "(Phase 1b runs concurrently with Phase 2 once 1a finishes)"
