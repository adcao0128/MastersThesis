#!/bin/bash
#SBATCH -o pspa_gpu.o%j
#SBATCH -p gpu
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=100gb
#SBATCH --output=job_pspa_embeddings_gpu_output.log
#SBATCH --error=job_pspa_error_gpu_embeddings.log
#SBATCH --mail-type=ALL            # Email notifications for job completion/failure
#SBATCH --mail-user=adc135@case.edu # Your email address for notifications
module load Python/3.10.8-GCCcore-12.2.0
source /home/adc135/Masters-Thesis/venv/bin/activate
cp -r read_pspa.py get_residue_embeddings.py Kinase_Substrate_Dataset.txt PSPA_ST.csv $PFSDIR
cd $PFSDIR
# Run PSPA script to fetch AlphaFold structures
echo "Running pspa.py..."
python3 read_pspa.py

# Run embedding extraction script
echo "Running get_residue_embeddings.py..."
mkdir -p residue_embeddings
python3 get_residue_embeddings.py --batch-size 16 --output-dir residue_embeddings