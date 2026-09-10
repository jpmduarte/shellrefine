#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu_mem:8000
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

set -e  # a failed conda activate must abort, not silently fall back to base python

source /tmp/set_variables_$SLURM_JOBID  #Necessary to load GPU Kernel

# ~/.bashrc returns early when non-interactive, so the conda init block never
# runs in a batch job. Source the hook directly.
source /home/users/2ai12_1/miniconda3/etc/profile.d/conda.sh
conda activate benchseg
echo "python: $(which python)"

# Baseline (control) for the boundary-weighted-loss test: identical config to
# job_boundary_on.sh except --boundary-weight-floor stays at 1.0 (off). Phase 2
# epochs cut from 250 to 100 -- the last run's val_patch_dice plateaued by
# epoch ~35-75 and never moved again through 250, so the extra epochs were
# wasted compute.
python pipeline.py \
    --images-dir /home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Images \
    --labels-dir /home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Labels/Head \
    --epochs-p1 250 --lr-p1 5e-4 \
    --epochs-p2 100 --lr-p2 3e-4 \
    --boundary-weight-floor 1.0
