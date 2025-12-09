#!/usr/bin/env bash
#SBATCH --job-name=xv2_wind_hold
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:V100:1

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kalyono3@gatech.edu

#SBATCH --output=/home/hice1/kalyono3/xview2_strong_baseline/slurm_logs/%j.holdout.out

nvidia-smi

module load anaconda3/2022.05.0.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/hice1/kalyono3/scratch/conda_venvs/xview2_strong_baseline

cd /home/hice1/kalyono3/xview2_strong_baseline

export WANDB_DISABLED=true

export CKPT_PTH="/home/hice1/kalyono3/scratch/weights/checkpoints/wind_supervised_lr0.0002_wd1e-06_bs8_seed42-epoch=26-val_loss=0.26.ckpt"

echo "Evaluating checkpoint:"
echo "$CKPT_PTH"

python eval_main.py \
    data=supervised_data_holdout \
    trainer.devices=1 \
    group="wind_supervised_holdout"

