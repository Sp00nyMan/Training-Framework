#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --output=logs/output-r_base.txt

module load miniconda3
source ~/.bashrc
conda deactivate
conda activate flow

echo "-----------------------------------------------------------------------------------------"

echo "Job ID: " $SLURM_JOB_ID
echo "Job Name: " $SLURM_JOB_NAME

python train.py --model r_base