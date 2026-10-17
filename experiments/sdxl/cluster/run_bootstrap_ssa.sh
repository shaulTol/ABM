#!/bin/bash
#SBATCH --job-name=traj_bootstrap
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=results/logs/bootstrap_%j.out
#SBATCH --error=results/logs/bootstrap_%j.err

# Trajectory bootstrap (B=2000) of the pooled SDXL Shat (eq:c_hat_def, 1/N
# stopping rule) for all 56 guidance-sweep cells (dt=0.005). CPU only; about
# 10 s per cell. Reads the per-trajectory CLIP snapshots written by
# run_traj_averaged.sh (z_clip_traj_<cell>.npz, next to traj_avg_<cell>.json)
# and writes one JSON per cell to results/bootstrap_pooled/.

export PYTHONUNBUFFERED=1
export PATH=<CONDA_BIN>:$PATH
source <CONDA_SH>
conda activate <CONDA_ENV>

cd <PROJECT_DIR>/experiments/sdxl

CLIP_DIR=${CLIP_DIR:-results}          # where z_clip_traj_*.npz and traj_avg_*.json live
OUT_DIR=results/bootstrap_pooled
mkdir -p results/logs "$OUT_DIR"

n=0
for npz in $(ls "$CLIP_DIR"/z_clip_traj_*_dt0p005.npz | sort); do
  cell=$(basename "$npz" .npz); cell=${cell#z_clip_traj_}
  python3 -u traj_averaged_pipeline.py --mode bootstrap --name bootstrap \
      --clip_traj "$npz" --B 2000 --seed 42 \
      --output_dir "$OUT_DIR" --bootstrap_output "bootstrap_${cell}.json" \
      || echo "FAILED $cell"
  n=$((n+1))
done
echo "Done: $n cells"
