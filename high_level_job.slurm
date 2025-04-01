#!/bin/bash
#SBATCH --job-name=high_level
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

module purge

module load python3

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

nohup python3 -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:0' 2>&1 | tee -a ./ETHUSDT.log &

wait

deactivate
