#!/bin/bash
#SBATCH -o clustering.o%j
#SBATCH --time=24:00:00      # 30 minutes of wall time
#SBATCH -N 1                 # 1 Node
#SBATCH -c 40                 # 1 processor
#SBATCH --mem=200gb             # Assign 2gb memory; default in 1gb
module load Python/3.10.8-GCCcore-12.2.0
python3 peptide_clustering.py