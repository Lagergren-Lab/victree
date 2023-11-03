#!/bin/bash
#
#SBATCH -J vic3gtempK9
#SBATCH -A naiss2023-5-290
#SBATCH -t 5:00:00
#SBATCH --mem 240G
#SBATCH --ntasks 40
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zampinetti@gmail.com
#

dat_path="$1"
# qt temperature extend param
qtempextend_list="0.5 0.8 1.0 1.3 2.0 5.0"
# gt multiplier
gtm_list="1.0 1.5 2.0"

export PYTHONPATH=/home/x_vitza/victree/src
source /home/x_vitza/shared/envs/victree-env/bin/activate
for file in $(ls ${dat_path}/K9*/*.png | grep -v "mr[0-9]\+" | sort -r); do
  for qtempextend in ${qtempextend_list}; do
    for gtm in ${gtm_list}; do
      echo "running config $file with qtempext $qtempextend gt mul $gtm and final step"
      srun --exclusive --ntasks=1 --mem 6G --cpus-per-task 1 python3 /home/x_vitza/victree/src/experiments/var_tree_experiments/run_single_dataset.py "${file}" $qtempextend $gtm &
    done;
  done;
done;

wait
