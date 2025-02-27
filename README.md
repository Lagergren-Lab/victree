# VICTree: Variational Inference for Clonal Tree reconstruction
> Implementation of VI method on clonal trees probabilistic model

## Requirements

Install the required packages with the following command:

```bash
python3.10 -m venv 'victree-env'
source victree-env/bin/activate
pip install -r requirements.txt
pip install . # install victree locally to access commands `victree` and `victree-simul`
```

or with conda

```
conda create -n victree-env python=3.10 -c conda-forge -c bioconda --file requirements.txt
conda activate victree-env
pip install .
```

## Run

To run the main program you can call `victree` in the environment

``` bash
victree --input input_data.h5ad --output out_path -n-nodes 10 --n-iter 100
```

### Options

Arguments such as input and VI parameters can be set through command line interface.
Run the main script with the `-h` flag to view all accepted arguments with their description.

## Synthetic data

To generate synthetic data, run the `victree-simul` script with arguments passed from CLI.

```
victree-simul --help
```

Note: the simulation script uses functions from the [`scgenome`](https://github.com/mondrian-scwgs/scgenome)
library. However, `scgenome` is not in the bioconda channel,
so you will need to install it with `pip install scgenome` separately.
We acknowledge that this installation might fail for compatibility issues on some
systems and we are working on a solution. You will still be able to run the main
inference algorithm with `victree` command.

