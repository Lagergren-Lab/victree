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
usage: main.py [-h] [--seed SEED] [--cuda] [--tmc-num-samples TMC_NUM_SAMPLES]

MAP Baum-Welch learning Bach Chorales

options:
  -h, --help            show this help message and exit
  --seed SEED
  --cuda
```


## Run

To run the script, just execute it with `python3`

``` bash
python3 main.py
```
