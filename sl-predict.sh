#!/bin/bash
#SBATCH --job-name=test-predict
#SBATCH --account=project_2009199
#SBATCH --time=00:15:00
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=9G
#SBATCH --gres=gpu:a100:1,nvme:6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o new_logs/%x.out
#SBATCH -e new_logs/%x.err


module purge
module load pytorch

#srun python predict.py \
#    --genre_model="/scratch/project_2009199/register-vs-genre/genre/training/models/xlmr-large-model-bce-loss1.pt" \
#    --data=CORE \
#    --results="results/against-large"

srun python predict.py \
    --data="register_oscar" \

seff $SLURM_JOBID
