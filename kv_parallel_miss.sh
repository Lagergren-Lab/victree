#!/bin/bash
#
#SBATCH -J victree
#SBATCH -A naiss2023-5-290
#SBATCH -t 10:00:00
#SBATCH --mem 80G
#SBATCH --ntasks 10
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zampinetti@gmail.com
#

dat_path="$1"
miss="/home/x_vitza/victree/score_res/missing_score_vk_files.txt"

export PYTHONPATH=/home/x_vitza/victree/src
source /home/x_vitza/shared/envs/victree-env/bin/activate
while IFS= read -r file; do
  echo "running config $file"
  srun --exclusive --ntasks=1 --cpus-per-task 1 --mem 8G python3 /home/x_vitza/victree/src/experiments/var_tree_experiments/run_dataset_neigh_K.py "${dat_path}${file}"&
done < "$miss";

wait

