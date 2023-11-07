# VICTree: Variational Inference for Clonal Tree reconstruction
> Implementation of VI method on clonal trees probabilistic model

## Requirements

For reproducibility, we include the `environment.yml` file generated from the
conda environment inside which all experiments have been run.

## Run

To run the script, just execute it with `python3`

``` bash
python ./src/main.py --input input_data.h5ad --output out_path -n-nodes 10 --n-iter 100
```

### Options

Arguments such as input and VI parameters can be set through command line interface.
Run the main script with the `-h` flag to view all accepted arguments with their description.

## Synthetic data

To generate synthetic data, run the `simul.py` script with arguments passed from CLI (as for `main.py`,
check the help page of the script for accepted arguments).

