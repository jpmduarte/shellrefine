#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu_mem:2000
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

set -e  # a failed conda activate must abort, not silently fall back to base python

source /tmp/set_variables_$SLURM_JOBID  #Necessary to load GPU Kernel

# ~/.bashrc returns early when non-interactive, so the conda init block never
# runs in a batch job. Source the hook directly.
source /home/users/2ai12_1/miniconda3/etc/profile.d/conda.sh
conda activate benchseg
echo "python: $(which python)"

python pipeline.py \
    --stage phase1 \
    --images-dir /home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Images \
    --labels-dir /home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Labels/Head
