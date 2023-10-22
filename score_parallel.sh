#!/bin/bash
#
#SBATCH -J victree
#SBATCH -t 15:00:00
#SBATCH --mem=96G
#SBATCH --ntasks 32
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zampinetti@gmail.com
#

dat_path="$1"

for file in "$dat_path"/*/*; do
  echo "running config $file"
  source /home/x_vitza/shared/envs/victree-env/bin/activate
  srun --exclusive --ntasks=1 python3 /home/x_vitza/victree/src/experiments/var_tree_experiments.py "$file"&
wait
