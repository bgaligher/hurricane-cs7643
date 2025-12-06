#!/bin/bash
#SBATCH -N 1    
#SBATCH -c 16
#SBATCH -p ice-gpu
#SBATCH --ntasks-per-node=1
#SBATCH -t 5:00:00               
#SBATCH --gres=gpu:H200:2
#SBATCH --mem=128G         
#SBATCH -J hurricane-lora    # jobs name
#SBATCH -o ./slurm_outs/hurricane_train-slurm_%j.out   # file to write logs, prints, etc

# activate python venv
echo "Activating virtual environment..."
VENV_PYTHON=../final_venv/bin/python
echo "Virtual environment activated!"

# Load modules
echo "Loading modules..."
module load anaconda3/2023.03
module load cuda
echo "Modules loaded!"

CONFIG_FILE='configs/config_hurricane1.yaml'

# And then some code to run, like
echo "Starting training script..."
$VENV_PYTHON train_hurricane.py --job_id $SLURM_JOB_ID --config_file $CONFIG_FILE
echo "Training complete!"

echo "Starting inference script..."
$VENV_PYTHON inference_hurricane.py --job_id $SLURM_JOB_ID
echo "Inference complete!"