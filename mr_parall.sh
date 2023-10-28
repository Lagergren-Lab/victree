#!/bin/bash
#
#SBATCH -J mrvictree
#SBATCH -A naiss2023-5-290
#SBATCH -t 10:00:00
#SBATCH --mem 80G
#SBATCH --ntasks 10
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zampinetti@gmail.com
#

dat_path="$1"

export PYTHONPATH=/home/x_vitza/victree/src
for file in ${dat_path}/K*mr*/*.png; do
  echo "running config $file"
  # srun --exclusive --ntasks 1 --mem 100M --cpus-per-task 1 sleep 3
  source /home/x_vitza/shared/envs/victree-env/bin/activate
  srun --exclusive --ntasks=1 --mem 8G --cpus-per-task 1 python3 /home/x_vitza/victree/src/experiments/var_tree_experiments/mut_rate_single_dataset.py "${dat_path}/${file}"&
done;

wait

