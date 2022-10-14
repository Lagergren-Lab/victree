# CopyTree: copy number over tree inference
> Pyro implementation for copy number inference over cancer clones trees

## Requirements

For reproducibility, refer to the following versions of the required libraries:

``` txt
pyro-ppl=1.8.1
python-graphviz
torch=1.11.0
```

## Options

``` 
usage: main.py [-h] [--seed SEED] [--cuda] [--n-iter N_ITER]

options:
  -h, --help   show this help message and exit
  --seed SEED
  --cuda
  --n-iter N_ITER variational inference maximum steps
  --log LEVEL any of DEBUG, INFO, ERROR, WARNING
```

## Run

To run the script, just execute it with `python3`

``` bash
python3 main.py
```
## Tests

To run tests, make sure you have `pytest` installed, then run:

```bash
pytest -v # for verbose test results
```
