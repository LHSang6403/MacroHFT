#!/bin/bash
#SBATCH --job-name=decomposition
#SBATCH --output=decomposition_%j.out
#SBATCH --error=decomposition_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00

module purge

module load python/3.10

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

nohup python -u preprocess/decomposition.py &

wait

deactivate
