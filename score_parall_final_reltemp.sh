#!/bin/bash
#
#SBATCH -J vic3gtemp
#SBATCH -A naiss2023-5-290
#SBATCH -t 8:00:00
#SBATCH --mem 90G
#SBATCH --ntasks 10
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zampinetti@gmail.com
#

dat_path="$1"
n_iter="$2"
# gt rel temperatures for final samples
rel_temp_g="1.3,1.5,2.0,5.0,10"

export PYTHONPATH=/home/x_vitza/victree/src
source /home/x_vitza/shared/envs/victree-env/bin/activate
for file in $(ls ${dat_path}/K12*/*.png | grep -v "mr[0-9]\+" | sort -r); do
  echo "running config $file with rel temp g final $rel_temp_g"
  srun --exclusive --ntasks=1 --mem 9G --cpus-per-task 1 python3 /home/x_vitza/victree/src/experiments/var_tree_experiments/single_dat_final_sample_gt.py "${file}" $rel_temp_g $n_iter &
done;

wait
