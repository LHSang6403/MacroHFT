#!/bin/bash
#SBATCH --job-name=low_level_slope_1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

module purge

module load python3

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

nohup python3 -u RL/agent/low_level.py --alpha 1 --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' \
    --label 'label_1' >./slope_1.log 2>&1 &

wait

deactivate
