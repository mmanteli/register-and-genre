#!/bin/bash
#SBATCH --job-name=topics
#SBATCH --account=project_2009199
#SBATCH --time=04:30:00
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -o %x.out
#SBATCH -e %x.err

num_topics=$1

source tvenv/bin/activate

srun python3 topic_modelling.py $num_topics

seff $SLURM_JOBID