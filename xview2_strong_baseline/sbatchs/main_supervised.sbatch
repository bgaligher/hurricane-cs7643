#!/usr/bin/env bash
#SBATCH --job-name=xv2_wind_train
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00             # <-- enough for full run
#SBATCH --gres=gpu:V100:1
##SBATCH --mem-per-gpu=32G

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kalyono3@gatech.edu

#SBATCH --output=/home/hice1/kalyono3/xview2_strong_baseline/slurm_logs/%j.full.out

nvidia-smi

module load anaconda3/2022.05.0.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/hice1/kalyono3/scratch/conda_venvs/xview2_strong_baseline

cd /home/hice1/kalyono3/xview2_strong_baseline

export WANDB_DISABLED=true     

srun python main.py \
    data=supervised_data \
    trainer.devices=1 \
    trainer.max_epochs=30 \
    group="wind_supervised"
