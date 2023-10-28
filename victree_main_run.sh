#!/bin/bash
#
#SBATCH -J K24
#SBATCH -t 30:00:00
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -A naiss2023-5-290
#SBATCH -o /proj/sc_ml/users/x_harme/victree/output/bahlis_10x/patient_MM-29/K24_slurm.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haraldme@kth.se
#

echo "Job started"

# load all required modules and libraries
module load Python/3.10.4-env-nsc1-gcc-2022a-eb
source "/proj/sc_ml/shared/envs/copytree-venv/bin/activate"

echo "Python module and venv loaded."

cd /proj/sc_ml/shared/victree/src
Echo "changed working dir to /proj/sc_ml/shared/victree/src/"

python main.py -i /proj/sc_ml/shared/bahlis_10x/patient_MM-29.h5ad --prior-mutau 1 100000 500 50 --prior-pi 3 -L 5 --qT-temp 500 --split ELBO --step-size 0.1 -n 200 -o /proj/sc_ml/users/x_harme/victree/output/bahlis_10x/patient_MM-29/K24L5i200step0p1splitELBOLambda100kInverseqTTemp500 -K 24

echo "Job finished"