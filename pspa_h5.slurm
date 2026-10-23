#!/bin/bash
#SBATCH -o pspa_h5.o%j
#SBATCH --time=20:00:00
#SBATCH -N 1
#SBATCH -c 40
#SBATCH --mem=240gb
#SBATCH --output=job_pspa_h5_output.log
#SBATCH --error=job_pspa_h5_error.log
#SBATCH --mail-type=ALL            # Email notifications for job completion/failure
#SBATCH --mail-user=adc135@case.edu # Your email address for notifications

# Load necessary modules (adjust to your HPC environment)
export TQDM_DISABLE=1
module load Python/3.10.8-GCCcore-12.2.0                 # Load Python module (adjust version)
module load CUDA/12.1.1                   # Load CUDA module (adjust version)
source /home/adc135/Masters-Thesis/venv/bin/activate

cp -r create_h5.py cleaned_pspa_filtered_balanced_dataset.csv $PFSDIR
cd $PFSDIR
python3 create_h5.py
