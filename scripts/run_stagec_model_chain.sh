#!/bin/bash
# One SLURM job per Stage-C finalist model. Runs its 5 folds x 3 seeds = 15
# full_pipeline runs SEQUENTIALLY inside this single allocation, instead of 15
# separate jobs -- keeps concurrent cluster load at 3 (one per model) instead of 45,
# after 45 concurrent jobs overloaded the shared cluster for other users.
#
# Required env vars: MODEL_NAME, IMAGES_DIR, LABELS_DIR, SHELLS_DIR, BAND_MM,
#                     EPOCHS_P1, LR_P1, LOSS_P1, BW_FLOOR_P1, BW_THR_P1, BW_COEF_P1, GD_COEF_P1,
#                     EPOCHS_P2, LR_P2, LOSS_P2, BW_FLOOR_P2, BW_COEF_P2

#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu_mem:6000
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

set -e

if [ -z "$MODEL_NAME" ]; then
    echo "MODEL_NAME must be set via sbatch --export=..."
    exit 1
fi

source /tmp/set_variables_$SLURM_JOBID  # Necessary to load GPU kernel

# ~/.bashrc returns early when non-interactive, so the conda init block never
# runs in a batch job. Source the hook directly.
source /home/users/2ai12_1/miniconda3/etc/profile.d/conda.sh
conda activate benchseg
echo "python: $(which python)"
echo "Stage C chain: MODEL_NAME=$MODEL_NAME"

SEEDS=(42 43 44)
FOLDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
        RUN_DIR="experiment/stageC/${MODEL_NAME}/fold${FOLD}/seed${SEED}"
        echo ""
        echo "===== ${MODEL_NAME}  fold=${FOLD}  seed=${SEED}  (run_dir=${RUN_DIR}) ====="
        python pipeline.py \
            --images-dir "$IMAGES_DIR" --labels-dir "$LABELS_DIR" \
            --shells-dir "$SHELLS_DIR" \
            --run-dir "$RUN_DIR" \
            --epochs-p1 "$EPOCHS_P1" --lr-p1 "$LR_P1" --loss-p1 "$LOSS_P1" \
            --boundary-weight-floor-p1 "$BW_FLOOR_P1" \
            --boundary-weight-thr-p1 "$BW_THR_P1" \
            --boundary-weight-coef-p1 "$BW_COEF_P1" \
            --gd-coef-p1 "$GD_COEF_P1" \
            --epochs-p2 "$EPOCHS_P2" --lr-p2 "$LR_P2" --loss-p2 "$LOSS_P2" \
            --boundary-weight-floor "$BW_FLOOR_P2" \
            --boundary-weight-coef "$BW_COEF_P2" \
            --band-mm "$BAND_MM" \
            --no-save-preds \
            --seed "$SEED"
    done
done

echo ""
echo "===== ${MODEL_NAME}: all 15 (fold, seed) runs complete ====="
