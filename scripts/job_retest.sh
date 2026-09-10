#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu_mem:10GB
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

set -e  # a failed conda activate must abort, not silently fall back to base python

source /tmp/set_variables_$SLURM_JOBID  #Necessary to load GPU Kernel

# ~/.bashrc returns early when non-interactive, so the conda init block never
# runs in a batch job. Source the hook directly.
source /home/users/2ai12_1/miniconda3/etc/profile.d/conda.sh
conda activate benchseg
echo "python: $(which python)"

# Retest of both phases with a lower lr and a longer cosine schedule. Phase 1
# was still improving steadily at epoch 50 (no plateau, train/val loss close
# together) so this mainly gives it more room to converge. Phase 2's
# val_patch_dice plateaued by epoch 7 and oscillated 0.910-0.917 without a
# clear trend through epoch 50 -- the classic symptom of lr too high for the
# task's fine detail, hence the bigger cut there.
python pipeline.py \
    --images-dir /home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Images \
    --labels-dir /home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Labels/Head \
    --epochs-p1 100 --lr-p1 5e-4 \
    --epochs-p2 100 --lr-p2 3e-4
