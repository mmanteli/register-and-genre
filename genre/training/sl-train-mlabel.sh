#!/bin/bash
#SBATCH --job-name=genre-train
#SBATCH --account=project_2002026
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:v100:1,nvme:10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

rm logs/current.err
rm logs/current.out
ln -s $SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOBID.out logs/current.out

module load pytorch

splt=$1
rnd=$2

srun python train_multilabel.py \
--split=$splt \
--truncate_texts=True \
--random_excerpt=$rnd \
--result_file="results/multilabel_${splt}_rnd_page_${rnd}_201123.txt"

seff $SLURM_JOBID
