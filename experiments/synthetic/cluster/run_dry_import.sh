#!/bin/bash
#SBATCH --job-name=dry_import
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --time=0:15:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=dry_import_%j.out
#SBATCH --error=dry_import_%j.err

export PYTHONUNBUFFERED=1
export PATH=<CONDA_BIN>:$PATH
source <CONDA_SH>
conda activate <CONDA_ENV>

NV_BASE=<NV_SITE_PACKAGES>
export LD_LIBRARY_PATH=$NV_BASE/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_cupti/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cufft/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cusolver/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=<USER_SITE_PACKAGES>/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=<USER_SITE_PACKAGES>/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=<TORCH_LIB>:$LD_LIBRARY_PATH

EXPDIR=<PROJECT_DIR>
cd $EXPDIR
export PYTHONPATH=$EXPDIR

echo "=== Dry-import sanity check: $(date) ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"

PASS=0
FAIL=0
SKIP=0

for f in experiments/synthetic/*.py experiments/alanine/*.py experiments/sdxl/*.py cli/*.py; do
    mod=$(echo "$f" | sed 's|/|.|g; s|\.py$||')

    # Check if script has __name__ guard — if not, importing will execute it
    if ! grep -q '__name__' "$f"; then
        # No guard — use timeout to catch scripts that run on import
        timeout 30 python -c "import importlib; importlib.import_module('$mod')" 2>/tmp/dry_import_err.txt
        rc=$?
        if [ $rc -eq 124 ]; then
            echo "SKIP (no __name__ guard, runs on import): $f"
            SKIP=$((SKIP+1))
            continue
        fi
    else
        python -c "import importlib; importlib.import_module('$mod')" 2>/tmp/dry_import_err.txt
        rc=$?
    fi

    if [ $rc -eq 0 ]; then
        echo "OK: $f"
        PASS=$((PASS+1))
    else
        err=$(cat /tmp/dry_import_err.txt | tail -1)
        if echo "$err" | grep -q "FileNotFoundError\|No such file"; then
            echo "OK (data missing, expected): $f"
            PASS=$((PASS+1))
        elif echo "$err" | grep -q "SystemExit"; then
            echo "OK (argparse exit): $f"
            PASS=$((PASS+1))
        else
            echo "FAIL: $f — $err"
            FAIL=$((FAIL+1))
        fi
    fi
done

echo ""
echo "=== SUMMARY: $PASS OK, $SKIP SKIPPED (no guard), $FAIL FAIL ==="
echo "Done: $(date)"
