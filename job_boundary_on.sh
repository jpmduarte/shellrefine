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

# Boundary-weighted-loss test. Identical to job_boundary_off.sh except the added
# BCE term (weight 1.0 inside the band already computed by make_crops, 0.2
# outside) is switched on. Same phase 1/2 lr and epochs as the control, so this
# isolates the one variable.
python pipeline.py \
    --images-dir /home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Images \
    --labels-dir /home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Labels/Head \
    --epochs-p1 250 --lr-p1 5e-4 \
    --epochs-p2 100 --lr-p2 3e-4 \
    --boundary-weight-floor 0.2 --boundary-weight-coef 0.5
