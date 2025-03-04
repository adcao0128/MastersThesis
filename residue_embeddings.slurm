#!/bin/bash
#SBATCH --job-name=esm_embeddings        # Job name
#SBATCH --output=esm_embeddings_%j.out  # Standard output log (%j for job ID)
#SBATCH --error=esm_embeddings_%j.err   # Standard error log
#SBATCH --partition=gpu                 # GPU partition (check with your HPC admin)
#SBATCH --gres=gpu:1                    # Number of GPUs (adjust as needed)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=16G                       # Memory allocation (adjust as needed)
#SBATCH --time=24:00:00                 # Time limit (HH:MM:SS)
#SBATCH --mail-type=ALL            # Email notifications for job completion/failure
#SBATCH --mail-user=adc135@case.edu # Your email address for notifications

# Load necessary modules (adjust to your HPC environment)
module load Python/3.10.8-GCCcore-12.2.0                 # Load Python module (adjust version)
module load CUDA/12.1.1                   # Load CUDA module (adjust version)
pwd
source /home/adc135/Masters-Thesis/venv/bin/activate
echo "Using Python executable: $(which python)"
python --version

python3 get_residue_embeddings.py --batch-size 16 --output-dir Masters-Thesis/MastersThesis/logs
