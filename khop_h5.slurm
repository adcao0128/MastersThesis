#!/bin/bash
#SBATCH -o pspa_gnn.o%j
#SBATCH -p gpu
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --mem=240gb
#SBATCH --output=job_pspa_gnn_output_2_psp.log
#SBATCH --error=job_pspa_gnn_error_2_psp.log
#SBATCH --mail-type=ALL            # Email notifications for job completion/failure
#SBATCH --mail-user=adc135@case.edu # Your email address for notifications
module load singularity

# Run PSPA script to fetch AlphaFold structures
echo "Running khop_h5.py..."
singularity exec --nv ../../singularity_ppi.sif bash -c "source /opt/venv/bin/activate && python khop_h5_2.py"