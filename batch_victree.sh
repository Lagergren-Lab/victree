#!/bin/bash

# Parse command line arguments
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config_dir>"
  echo "run the script in the folder relative to the input file specification"
  exit 1
fi

config_dir="$1"
input_file="$2"
# remote_dir="$3"

# Set output directory prefix
output_prefix="output_batch"
mkdir -p "${output_prefix}"

# Create zip directory if it doesn't exist
zip_dir="./zip"
mkdir -p "${zip_dir}"

# Loop over configuration files in directory
for config_file in "${config_dir}"/*.yml; do
  # Load parameters from YAML file
  # eval $(yq -r 'to_entries | .[] | "export \(.key)=\(.value)"' "${config_file}")
  
  # Set output directory for this run
  output_dir="${output_prefix}/$(basename "${config_file}" .yml)"

  args=("--output" "${output_dir}")

  while IFS="=" read -r key value; do 
    # replace underscores with dashes
    key=${key//_/-}
    # flag param
    if [[ "$value" == "true" ]]; then
      args+=("--$key")
    elif [[ "$key" == "input" ]]; then
      args+=("--$key $(pwd)/$value")
    else
      args+=("--$key $value")
    fi
  done < <(
    yq 'to_entries | map([.key, .value] | join("=")) | .[]' "${config_file}"
  )

  # Run script with parameters
  full_cmd="/Users/zemp/phd/scilife/coPyTree/src/main.py ${args[@]}"
  echo "executing: ${full_cmd}"
  eval "$full_cmd"
  
  # Zip output directory and delete it
  # zip -r "${output_dir}.zip" "$(basename "${output_dir}")"
  # rm -rf "${output_dir}"

  # # Copy zip file to remote directory
  # scp "${zip_dir}/${output_dir}.zip" "${remote_dir}/$(basename "${output_dir}").zip" && rm -f "${zip_dir}/${output_dir}.zip"
  
done

