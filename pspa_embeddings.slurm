#!/bin/bash
#SBATCH -o pspa_embeddings.o%j
#SBATCH --time=20:00:00
#SBATCH -N 1
#SBATCH -c 40
#SBATCH --mem=100gb
#SBATCH --output=job_pspa_embeddings_output.log
#SBATCH --error=job_pspa_embeddings_error.log
#SBATCH --mail-type=ALL            # Email notifications for job completion/failure
#SBATCH --mail-user=adc135@case.edu # Your email address for notifications

# Load necessary modules (adjust to your HPC environment)
module load Python/3.10.8-GCCcore-12.2.0                 # Load Python module (adjust version)
module load CUDA/12.1.1                   # Load CUDA module (adjust version)
source /home/adc135/Masters-Thesis/venv/bin/activate

cp -r get_residue_embeddings.py /scratch/pbsjobs/job.2610924.hpc/alphafold_pdb_files $PFSDIR
cd $PFSDIR
python3 get_residue_embeddings.py --batch-size 16 --output-dir Masters-Thesis/MastersThesis/logs
