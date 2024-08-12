#!/bin/bash
#SBATCH --job-name=BERTopics
#SBATCH --account=project_2009199
#SBATCH --time=00:15:00
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --gres=gpu:a100:1

clean=$1
source venv/bin/activate

srun python topic_modelling.py $clean

seff $SLURM_JOBID
cp  logs/$SLURM_JOBID.out "result_cleaned_${clean}.txt"
