# VICTree: Variational Inference for Clonal Tree reconstruction
> Implementation of VI method on clonal trees probabilistic model

## Requirements

For reproducibility, we include the `environment.yml` file generated from the
conda environment inside which all experiments have been run.

## Run

To run the script, just execute it with `python3`

``` bash
python ./src/main.py
```

### Options

Arguments such as input and VI parameters can be set through command line interface.
Run the main script with the `-h` flag to view all accepted arguments with their description.

## Synthetic data

To generate synthetic data, run the `simul.py` script with arguments passed from CLI (as for `main.py`,
check the help page of the script for accepted arguments).

## Visualization

### VI
For VI results visualization, execute the main script with the `--diagnostics` flag;
this will save information about inference at each iteration and write it on a `checkpoint*.h5` file.
This file can be inspected by running the `r_analysis.R` script.

An example would be
```bash
Rscript ./r_analysis.R output/checkpoint_k5a7n300m1000.h5 -gt datasets/simul_k5a7n300m1000e1-50d10mt1-10-500-50.h5 -m -p 2
```
which writes a PDF with several plots showing the output of inference. If run with simulated data, `-gt` option provides
ground truth information to the plots and with `-m` the script will attempt a remapping of the variational inference
clone names to the ground truth (although not guaranteed, in case, for instance, of different values of clones between
inferred and ground truth).

### Synthetic dataset
To quickly inspect how the generated data looks like, it is enough to run the `data_vis.R` script.

E.g.
```bash
Rscript ./data_vis.R ./dataset/simul_k5a7n300m1000e1-50d10mt1-10-500-50.h5
```
which will generate a single page PDF in the same folder of the input data.
