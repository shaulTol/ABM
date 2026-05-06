#!/bin/bash
#SBATCH --job-name=ssa_dt_exp
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --output=ssa_dt_exp_%j.out
#SBATCH --error=ssa_dt_exp_%j.err

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

BASEDIR=/sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl
SCRIPTS=$BASEDIR/scripts
OUTDIR=/sci/labs/orzuk/shaulytolk/ABM/experiments/ssa_truncation_redef/results/sdxl_dt_experiment

mkdir -p $OUTDIR

# ── Config ──
MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"
T=2000

# Prompt definitions matching guidance sweep
declare -A PROMPTS
PROMPTS[person]="A realistic photo of a person"
PROMPTS[cat]="A realistic photo of a cat"
PROMPTS[mountains]="A scenic photo of mountains"
PROMPTS[flowers]="A close-up photo of flowers"
PROMPTS[street]="A photo of a busy street"
PROMPTS[food]="A photo of a delicious meal"
PROMPTS[elderly]="A realistic photo of an elderly person"
PROMPTS[car]="A photo of a car"

# Latent file paths
declare -A LATENT_PATHS
LATENT_PATHS[person_gs1p0]="$BASEDIR/results_guidance_sweep/clip_da_latents_sweep_gs1p0.npz"
LATENT_PATHS[person_gs7p5]="$BASEDIR/results_guidance_sweep/clip_da_latents_sweep_gs7p5.npz"
LATENT_PATHS[cat_gs1p0]="$BASEDIR/results_multiprompt_sweep/clip_da_latents_mp_p0_cat_gs1p0.npz"
LATENT_PATHS[cat_gs7p5]="$BASEDIR/results_multiprompt_sweep/clip_da_latents_mp_p0_cat_gs7p5.npz"
LATENT_PATHS[mountains_gs1p0]="$BASEDIR/results_multiprompt_sweep/clip_da_latents_mp_p1_mountains_gs1p0.npz"
LATENT_PATHS[mountains_gs7p5]="$BASEDIR/results_multiprompt_sweep/clip_da_latents_mp_p1_mountains_gs7p5.npz"
LATENT_PATHS[flowers_gs1p0]="$BASEDIR/results_multiprompt_sweep/clip_da_latents_mp_p2_flowers_gs1p0.npz"
LATENT_PATHS[flowers_gs7p5]="$BASEDIR/results_multiprompt_sweep/clip_da_latents_mp_p2_flowers_gs7p5.npz"
LATENT_PATHS[street_gs1p0]="$BASEDIR/results_multiprompt_sweep/clip_da_latents_mp_p3_street_gs1p0.npz"
LATENT_PATHS[street_gs7p5]="$BASEDIR/results_multiprompt_sweep/clip_da_latents_mp_p3_street_gs7p5.npz"
LATENT_PATHS[food_gs1p0]="$BASEDIR/results_multiprompt_sweep/clip_da_latents_mp_p4_food_gs1p0.npz"
LATENT_PATHS[food_gs7p5]="$BASEDIR/results_multiprompt_sweep/clip_da_latents_mp_p4_food_gs7p5.npz"
LATENT_PATHS[elderly_gs1p0]="$BASEDIR/results_multiprompt_sweep/clip_da_latents_mp_p5_elderly_gs1p0.npz"
LATENT_PATHS[elderly_gs7p5]="$BASEDIR/results_multiprompt_sweep/clip_da_latents_mp_p5_elderly_gs7p5.npz"
LATENT_PATHS[car_gs1p0]="$BASEDIR/results_multiprompt_sweep/clip_da_latents_mp_p6_car_gs1p0.npz"
LATENT_PATHS[car_gs7p5]="$BASEDIR/results_multiprompt_sweep/clip_da_latents_mp_p6_car_gs7p5.npz"

# Guidance scale values
declare -A GS_VALUES
GS_VALUES[gs1p0]=1.0
GS_VALUES[gs7p5]=7.5

cd $SCRIPTS

# Run all combinations: 8 prompts × 2 gs × 2 dt values
for PROMPT_KEY in person cat mountains flowers street food elderly car; do
    for GS_KEY in gs1p0 gs7p5; do
        LATENT_KEY="${PROMPT_KEY}_${GS_KEY}"
        LATENT_FILE="${LATENT_PATHS[$LATENT_KEY]}"
        GS_VAL="${GS_VALUES[$GS_KEY]}"
        PROMPT="${PROMPTS[$PROMPT_KEY]}"

        if [ ! -f "$LATENT_FILE" ]; then
            echo "SKIP: $LATENT_FILE not found"
            continue
        fi

        for DT in 0.002 0.005; do
            DT_LABEL=$(echo $DT | tr '.' 'p')
            NAME="redef_${PROMPT_KEY}_${GS_KEY}_dt${DT_LABEL}"
            echo ""
            echo "=== $NAME ==="
            echo "  Prompt: $PROMPT"
            echo "  GS: $GS_VAL, dt: $DT, T: $T"
            echo "  x0: $LATENT_FILE"

            python -u run_ssa_da_clip.py \
                --model_id "$MODEL_ID" \
                --name "$NAME" \
                --prompt "$PROMPT" \
                --guidance_scale $GS_VAL \
                --height 1024 --width 1024 \
                --resume_x0 "$LATENT_FILE" \
                --dt $DT --T $T \
                --snapshot_schedule log_and_fixed \
                --ssa_only \
                --no_save_checkpoint \
                --output_dir "$OUTDIR" \
                --clip_batch_size 8 \
                --score_batch_size 64
        done
    done
done

echo ""
echo "=== ALL DONE ==="
