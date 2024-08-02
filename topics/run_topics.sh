#!/bin/bash
#SBATCH --job-name=topics
#SBATCH --account=project_2009199
#SBATCH --time=01:30:00
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -o %j.out
#SBATCH -e %j.err

num_topics=$1

source tvenv/bin/activate

srun python3 topic_modelling.py $num_topics 10

seff $SLURM_JOBID
cp  $SLURM_JOBID.out "${num_topics}topics_10words_w_support.txt"