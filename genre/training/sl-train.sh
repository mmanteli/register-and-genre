#!/bin/bash
#SBATCH --job-name=xlmr-l
#SBATCH --account=project_2009199
#SBATCH --time=02:30:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:a100:1,nvme:30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%x%j.out
#SBATCH -e logs/%x%j.err

rm logs/current.err
rm logs/current.out
ln -s $SLURM_JOB_NAME$SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOB_NAME$SLURM_JOBID.out logs/current.out
mkdir -p models
date=$(date +"%Y-%m-%d-%H-%M")
module load pytorch

srun python train_ml_with_defined_classes.py \
    --labels='["POL","ENG","LIT","SMA","MED","EAT"]' \
    --lr=7e-6 \
    --wd=1e-4 \
    --epochs=8 \
    --batch_size=8 \
    --model_name='xlm-roberta-large' \
    --save_model=models/xlmr-large-${date}.pt \
    --result_file=models/xlmr-large-${date}.txt \
    --random_excerpt=False

# xlrm models: 1=lr=1e-5  2=lr=5e-5 3=lr=5e-6 => 3 best results for BASE
# large models: best 3e-6 and 1e-5 => trying 7e-6
