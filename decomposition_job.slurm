#!/bin/bash
#SBATCH --job-name=decomposition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00

module purge

module load python3

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

nohup python3 -u preprocess/decomposition.py &

wait

deactivate