#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=project_2002026
#SBATCH --time=00:10:00
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --gres=gpu:v100:1,nvme:6G
#SBATCH -o logs/%j%x.out
#SBATCH -e logs/%j%x.err

module purge
module load pytorch

srun python3 predict.py \
    --data=CORE

seff $SLURM_JOBID

