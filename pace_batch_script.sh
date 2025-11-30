#!/bin/bash
#SBATCH -N 1     
#SBATCH -c 4
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:15:00                  
#SBATCH --gres=gpu:V100:1          
##SBATCH --mem-per-gpu=32G
#SBATCH -J hurricane-train-test    # jobs name
#SBATCH -o ./slurm_outs/slurm_%j.out   # file to write logs, prints, etc

# activate python venv
echo "Activating virtual environment..."
VENV_PYTHON=../final_venv/bin/python
echo "Virtual environment activated!"

# Load modules
echo "Loading modules..."
module load anaconda3/2023.03
module load cuda
echo "Modules loaded!"

# And then some code to run, like
echo "Starting training script..."
$VENV_PYTHON train_hurricane.py
echo "Training complete!"