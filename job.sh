#!/usr/bin/env bash
#SBATCH -A cs541
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A30
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --job-name=P3-DQN

set -e

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/mfahmy/miniconda3/envs/myenv
python -u -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0))'

#python -u main.py --train_dqn
python -u main.py --test_dqn --record_video 
