#!/bin/bash
#SBATCH --job-name=dry_import
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --time=0:10:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=dry_import_%j.out
#SBATCH --error=dry_import_%j.err

export PYTHONUNBUFFERED=1
export PATH=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env

NV_BASE=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NV_BASE/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_cupti/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cufft/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cusolver/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NV_BASE/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/home/shaulytolk/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

EXPDIR=/sci/labs/orzuk/shaulytolk/ssa-paper-test
cd $EXPDIR
export PYTHONPATH=$EXPDIR

echo "=== Dry-import sanity check: $(date) ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"

PASS=0
FAIL=0

for f in experiments/synthetic/*.py experiments/alanine/*.py experiments/sdxl/*.py cli/*.py; do
    mod=$(echo "$f" | sed 's|/|.|g; s|\.py$||')
    python -c "import importlib; importlib.import_module('$mod')" 2>/tmp/dry_import_err.txt
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "OK: $f"
        PASS=$((PASS+1))
    else
        err=$(cat /tmp/dry_import_err.txt | tail -1)
        # FileNotFoundError for missing data is expected
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
echo "=== SUMMARY: $PASS OK, $FAIL FAIL ==="
echo "Done: $(date)"
