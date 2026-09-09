#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu_mem:6000
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

set -e  # a failed conda activate must abort, not silently fall back to base python

if [ -z "$FLOOR" ] || [ -z "$COEF" ]; then
    echo "FLOOR and COEF must be set via: sbatch --export=FLOOR=0.2,COEF=1.0 job_boundary_sweep_point.sh"
    exit 1
fi

source /tmp/set_variables_$SLURM_JOBID  #Necessary to load GPU Kernel

# ~/.bashrc returns early when non-interactive, so the conda init block never
# runs in a batch job. Source the hook directly.
source /home/users/2ai12_1/miniconda3/etc/profile.d/conda.sh
conda activate benchseg
echo "python: $(which python)"

# One point of the boundary-weighted-loss coefficient sweep. Reuses the
# shells/phase1 checkpoint/crops from an existing run (fixed, never retrained
# here) so every point in the grid differs ONLY in floor/coef -- phase 1's
# ~0.005 Dice run-to-run seed noise (observed between the two "identical"
# on/off jobs) does not leak into this comparison.
BASE=/home/users/2ai12_1/fetai/boundary_first_then_refine/runs/20260907_163859
RUN_DIR="$BASE/phase2_sweep_f${FLOOR}_c${COEF}"

echo "sweep point: floor=$FLOOR coef=$COEF -> $RUN_DIR"

python train_phase2.py \
    --epochs 100 --lr 3e-4 --loss dice_bce \
    --run-dir "$RUN_DIR" \
    --crops-dir "$BASE/crops" \
    --splits-json "$BASE/splits.json" \
    --profile-dir "$RUN_DIR/profile" \
    --boundary-weight-floor "$FLOOR" \
    --boundary-weight-coef "$COEF"

python evaluate_phase2.py \
    --run-dir "$RUN_DIR" \
    --split val \
    --crops-dir "$BASE/crops" \
    --splits-json "$BASE/splits.json" \
    --profile-dir "$RUN_DIR/profile"
