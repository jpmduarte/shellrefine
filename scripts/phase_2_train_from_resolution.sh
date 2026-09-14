#!/bin/bash
# Takes one phase-1 resolution all the way to a refined score in a single job: builds the
# crops from that resolution's phase-1 checkpoint, trains phase 2 on them, evaluates.
# Phase 1 itself is assumed already trained, since it is the cheap part and its checkpoint
# is what decides where the crops and the band fall.
#
# Everything except RES defaults to the f2_plain configuration, so the number that comes
# out is directly comparable with experiment/stageC/f2_plain/fold0/seed42.
#
# Usage:
#   sbatch --export=ALL,RES=64 scripts/phase_2_train_from_resolution.sh
#   sbatch --export=ALL,RES=48 scripts/phase_2_train_from_resolution.sh

#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu_mem:6000
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

set -e

if [ -z "$RES" ]; then
    echo "RES must be set via sbatch --export=ALL,RES=64"
    exit 1
fi

IMAGES_DIR=${IMAGES_DIR:-/home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Images}
LABELS_DIR=${LABELS_DIR:-/home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Labels/Head}
SHELLS_DIR=${SHELLS_DIR:-runs/${RES}_3_shells/shells}
PHASE1_RUN=${PHASE1_RUN:-experiment/${RES}_3_phase_1_train}
RUN_DIR=${RUN_DIR:-experiment/${RES}_3_phase_2_train}

BAND_MM=${BAND_MM:-5.0}
EPOCHS_P2=${EPOCHS_P2:-100}
LR_P2=${LR_P2:-0.0003}
LOSS_P2=${LOSS_P2:-dice_bce}
BW_FLOOR_P2=${BW_FLOOR_P2:-1.0}
BW_COEF_P2=${BW_COEF_P2:-1.0}
SEED=${SEED:-42}

source /tmp/set_variables_$SLURM_JOBID  #Necessary to load GPU Kernel

# ~/.bashrc returns early when non-interactive, so the conda init block never runs in a
# batch job. Source the hook directly.
source /home/users/2ai12_1/miniconda3/etc/profile.d/conda.sh
conda activate benchseg
echo "python: $(which python)"
echo "res=${RES}^3  shells=$SHELLS_DIR  run_dir=$RUN_DIR  seed=$SEED"

mkdir -p "$RUN_DIR"

echo "===== crops a partir do checkpoint da fase 1 @ ${RES}^3 ====="
python make_crops.py \
    --images-dir "$IMAGES_DIR" --labels-dir "$LABELS_DIR" \
    --shells-dir "$SHELLS_DIR" \
    --out-dir "$RUN_DIR/crops" \
    --checkpoint "$PHASE1_RUN/phase1/checkpoints/phase1_best.pth" \
    --band-mm "$BAND_MM" \
    --profile-dir "$RUN_DIR/profile"

cp "$PHASE1_RUN/splits.json" "$RUN_DIR/splits.json"

echo "===== treino da fase 2 ====="
python train_phase2.py \
    --epochs "$EPOCHS_P2" --lr "$LR_P2" --loss "$LOSS_P2" \
    --run-dir "$RUN_DIR" \
    --crops-dir "$RUN_DIR/crops" \
    --splits-json "$RUN_DIR/splits.json" \
    --profile-dir "$RUN_DIR/profile" \
    --boundary-weight-floor "$BW_FLOOR_P2" \
    --boundary-weight-coef "$BW_COEF_P2" \
    --seed "$SEED"

echo "===== avaliacao ====="
python evaluate_phase2.py \
    --run-dir "$RUN_DIR" --split val \
    --crops-dir "$RUN_DIR/crops" \
    --splits-json "$RUN_DIR/splits.json" \
    --profile-dir "$RUN_DIR/profile" \
    --no-save-preds

echo "===== ${RES}^3 phase 2 completo ====="
