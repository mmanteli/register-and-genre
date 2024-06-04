#!/bin/bash
#SBATCH --job-name=genre-optimisation
#SBATCH --account=project_2002026
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:v100:1,nvme:10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

rm logs/current.err
rm logs/current.out
ln -s $SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOBID.out logs/current.out


mdl=$1
splt=$2

module load pytorch
echo "${mdl} + ${splt}"
srun python optimize_train.py --mod=$mdl --split=$splt --truncate_texts=True --result_file="./results/opt/opt_round3_${mdl}_${splt}_random_page_True.txt" --save_model="opt_models/model_${mdl}_${splt}_"
seff $SLURM_JOBID
