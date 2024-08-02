#!/bin/bash
#SBATCH --job-name=test-predict
#SBATCH --account=project_2009199
#SBATCH --time=04:15:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=9G
#SBATCH --gres=gpu:a100:1,nvme:6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o predict_logs/%j.out
#SBATCH -e predict_logs/%j.err


module purge
module load pytorch
sample=$1
results="/scratch/project_2009199/register-vs-genre/results/en_${sample}/"
mkdir -p $results

#srun python predict.py \
#    --genre_model="/scratch/project_2009199/register-vs-genre/genre/training/models/xlmr-large-model-bce-loss1.pt" \
#    --data=CORE \
#    --results="results/against-large"

srun python predict.py \
    --data="sampled_reg_oscar" \
    --sample=$sample \
    --results=$results

seff $SLURM_JOBID
