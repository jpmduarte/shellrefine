#!/bin/bash
# One grid point, dispatched by variables exported via `sbatch --export=...` (see
# submit_grid.sh). Generalises job_boundary_sweep_point.sh's body to the three kinds of
# manifest row: phase1_only (Stage A1/A1b/A1c/A1d), phase2_only (Stage A2/A2b/A3, reusing
# a frozen phase-1 checkpoint/crops), full_pipeline (Stage C, one fold+seed).
#
# Required env vars (all rows): RUN_ID, KIND, RUN_DIR
# phase1_only:    SHELLS_DIR, SPLITS_JSON, EPOCHS_P1, LR_P1, LOSS_P1,
#                 BW_FLOOR_P1, BW_THR_P1, BW_COEF_P1, GD_COEF_P1, [SEED]
# phase2_only:    BASE_RUN, EPOCHS_P2, LR_P2, LOSS_P2, BW_FLOOR_P2, BW_COEF_P2, [SEED]
# full_pipeline:  IMAGES_DIR, LABELS_DIR, SHELLS_DIR, SPLITS_JSON, SEED, and every
#                 phase1/phase2 hyperparameter above

#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu_mem:6000
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

set -e  # a failed conda activate must abort, not silently fall back to base python

if [ -z "$RUN_ID" ] || [ -z "$KIND" ] || [ -z "$RUN_DIR" ]; then
    echo "RUN_ID, KIND and RUN_DIR must be set via sbatch --export=..."
    exit 1
fi

source /tmp/set_variables_$SLURM_JOBID  #Necessary to load GPU Kernel

# ~/.bashrc returns early when non-interactive, so the conda init block never
# runs in a batch job. Source the hook directly.
source /home/users/2ai12_1/miniconda3/etc/profile.d/conda.sh
conda activate benchseg
echo "python: $(which python)"
echo "grid point: RUN_ID=$RUN_ID  KIND=$KIND  RUN_DIR=$RUN_DIR"

SEED_ARGS=()
[ -n "$SEED" ] && SEED_ARGS=(--seed "$SEED")

case "$KIND" in

build_crops)
    # One-time step between A1-family (phase1_only) and A2-family (phase2_only): builds
    # the crops/ that every phase2_only row of a sub-stage shares, from one frozen
    # phase-1 checkpoint. Not itself a grid point.
    python make_crops.py \
        --images-dir "$IMAGES_DIR" --labels-dir "$LABELS_DIR" \
        --shells-dir "$SHELLS_DIR" \
        --out-dir "$RUN_DIR/crops" \
        --checkpoint "$P1_CHECKPOINT" \
        --band-mm "${BAND_MM:-5.0}" \
        --profile-dir "$RUN_DIR/profile"
    cp "$SPLITS_JSON" "$RUN_DIR/splits.json"
    ;;

phase1_only)
    python train.py \
        --epochs "$EPOCHS_P1" --lr "$LR_P1" --loss "$LOSS_P1" \
        --run-dir "$RUN_DIR" \
        --shells-dir "$SHELLS_DIR" \
        --splits-json "$SPLITS_JSON" \
        --profile-dir "$RUN_DIR/profile" \
        --boundary-weight-floor "$BW_FLOOR_P1" \
        --boundary-weight-thr "$BW_THR_P1" \
        --boundary-weight-coef "$BW_COEF_P1" \
        --gd-coef "$GD_COEF_P1" \
        "${SEED_ARGS[@]}"

    python evaluate.py \
        --run-dir "$RUN_DIR" --split val \
        --shells-dir "$SHELLS_DIR" \
        --splits-json "$SPLITS_JSON" \
        --profile-dir "$RUN_DIR/profile"
    ;;

phase2_only)
    python train_phase2.py \
        --epochs "$EPOCHS_P2" --lr "$LR_P2" --loss "$LOSS_P2" \
        --run-dir "$RUN_DIR" \
        --crops-dir "$BASE_RUN/crops" \
        --splits-json "$BASE_RUN/splits.json" \
        --profile-dir "$RUN_DIR/profile" \
        --boundary-weight-floor "$BW_FLOOR_P2" \
        --boundary-weight-coef "$BW_COEF_P2" \
        "${SEED_ARGS[@]}"

    python evaluate_phase2.py \
        --run-dir "$RUN_DIR" --split val \
        --crops-dir "$BASE_RUN/crops" \
        --splits-json "$BASE_RUN/splits.json" \
        --profile-dir "$RUN_DIR/profile" \
        --no-save-preds
    ;;

full_pipeline)
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
        --no-save-preds \
        "${SEED_ARGS[@]}"
    ;;

*)
    echo "Unknown KIND: $KIND (expected phase1_only|phase2_only|full_pipeline)"
    exit 1
    ;;
esac
