#!/bin/bash

create_output_dir() {
  local yaml_file="$1"

  local yaml_dir="$(dirname "$yaml_file")"
  local output_dir="$yaml_dir/$(basename "${yaml_file%.*}")"

  if [ ! -d "$output_dir" ]; then
    mkdir "$output_dir"
  else
    echo "ERROR: Output dir was already present. Exit." >&2
    # exit 1
  fi

  echo "${output_dir}"
}

parse_yaml_args() {
  local yaml_file="$1"
  local output_dir="$2"

  local yaml_dir="$(dirname "$yaml_file")"
  local args=("--output" "${output_dir}")

  while IFS="=" read -r key value; do 
    # replace underscores with dashes
    key=${key//_/-}
    # flag param
    if [[ "${value}" == "true" ]]; then
      args+=("--${key}")
    elif [[ "${key}" == "input" ]]; then
      args+=("--${key} ${yaml_dir}/${value}")
    else
      args+=("--${key} ${value}")
    fi
  done < <(
    yq 'to_entries | map([.key, .value] | join("=")) | .[]' "${yaml_file}"
  )

  echo "${args[@]}"
}

copytree_dir="/home/x_vitza/coPy-tree"

# validate input
if [ $# -ne 1 ]; then
  if [ $# -eq 2 ]; then
    copytree_dir="$2"
  else
    echo "Usage: $0 <config.yml> [ <copytree_dir> ]"
    exit 1
  fi
fi

victree="${copytree_dir}/src/main.py"
# analysis="./r_analysis.R"

yaml_file="$1"
yaml_dir="$(dirname "$yaml_file")"
start_dir="$(pwd)"

echo "Setting up output directory"
output_dir=$(create_output_dir "${yaml_file}")
echo "OUTPUT: ${output_dir}"
# echo "Parsing config"
# args=($(parse_yaml_args "${yaml_file}" "${output_dir}"))
# echo "Running VI"
# "${victree}" "${args[@]}" || {
#   echo "Something went wrong. Cleaning up output dir"
#   rm -r "${output_dir}"
#   exit 1
# }

checkpoint="$(realpath "$(find "${output_dir}" -maxdepth 1 -type f -name "checkpoint*.h5")")"
if [ -f "$checkpoint" ]; then
  echo "checkpoint found! $checkpoint"
else
  echo "checkpoint not found. skipping analysis"
  exit
fi

input_file="$(realpath "${yaml_dir}/$(yq eval '.input' ${yaml_file})")"

# go in copytree_dir so to use catch scripts dependencies
echo "moving into ${copytree_dir}"
cd "${copytree_dir}" || exit 1

echo "Running R analysis"
./r_analysis.R -gt "$input_file" -m -p 2 "$checkpoint" 
echo "Finish."

# go back
echo "moving back to ${start_dir}"
cd "${start_dir}"

