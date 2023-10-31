#!/bin/bash
#
#SBATCH -J scoremissK12
#SBATCH -A naiss2023-5-290
#SBATCH -t 10:00:00
#SBATCH --mem 84G
#SBATCH --ntasks 7
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zampinetti@gmail.com
#

dat_path="$1"
miss="/home/x_vitza/victree/score_res/missing_score_files.txt"

export PYTHONPATH=/home/x_vitza/victree/src
while IFS= read -r file; do
  echo "running config $file"
  source /home/x_vitza/shared/envs/victree-env/bin/activate
  srun --exclusive --ntasks=1 --mem 12G --cpus-per-task 1 python3 /home/x_vitza/victree/src/experiments/var_tree_experiments/run_single_dataset.py "${dat_path}/${file}"&
done < "$miss";

wait

