#!/bin/bash
#
#SBATCH -J victree
#SBATCH -t 20:00:00
#SBATCH --mem=85G
#SBATCH --ntasks 15
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haraldme@kth.se
#

# Function to process each YAML file
process_yaml() {
  local yaml_file="$1"

  echo "running config $1"
  srun --exclusive --ntasks=1 --cpus-per-task=1 --mem=5G \
    run_victree_analysis.sh "${yaml_file}"&
}

# Function to recursively search for YAML files
search_yaml() {
  local dir="$1"

  # Loop through all files and directories in the given directory
  for file in "$dir"/*; do
    if [ -f "$file" ]; then
      # Check if the file is a YAML file
      if [[ "$file" == *.yaml || "$file" == *.yml ]]; then
        # Check if the YAML file contains an 'input' key
        if grep -q '^[[:space:]]*input:' "$file"; then
          process_yaml "$file"
        fi
      fi
    elif [ -d "$file" ]; then
      # Recursively search the subdirectory
      search_yaml "$file"
    fi
  done
}

# validate input
if [ $# -ne 1 ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# load all required modules and libraries
module add Python/3.10.4-env-nsc1-gcc-2022a-eb
source "/proj/sc_ml/shared/envs/victree-env/bin/activate"
module add R/4.2.2-nsc1-gcc-11.3.0-bare
export R_LIBS="/proj/sc_ml/shared/envs/r-libs"
#export R_LIBS_USER="/proj/sc_ml/shared/envs/r-libs"
export NUMEXPR_MAX_THREADS=32

directory="$1"

# Check if the provided directory exists
if [ ! -d "$directory" ]; then
  echo "Error: Directory '$directory' not found."
  exit 1
fi

search_yaml "$directory"
wait

