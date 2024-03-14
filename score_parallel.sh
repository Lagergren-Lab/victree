#!/bin/bash
#
#SBATCH -J perf_victree
#SBATCH -A naiss2023-5-290
#SBATCH -t 2:00:00
#SBATCH --mem 80G
#SBATCH --ntasks 10
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zampinetti@gmail.com
#

dat_path="$1"
qt_extend="1.3"
gt_rel="2.0"

export PYTHONPATH=/home/x_vitza/victree/src
source /home/x_vitza/shared/envs/victree-env/bin/activate
for file in $(ls ${dat_path}/K9*/*.png | grep -v "mr[0-9]\+" | sort -r); do
  echo "running config $file"
  srun --exclusive --ntasks=1 --mem 8G --cpus-per-task 1 python3 /home/x_vitza/victree/src/experiments/var_tree_experiments/run_single_dataset.py "${file}" ${qt_extend} ${gt_rel} &
done;

wait

