#!/bin/bash

export PYTHONPATH="/zfsauton2/home/mingzhul/time-series-prompt/moment-research"


# export WANDB_MODE="offline"


### 
python3 scripts/baselines/gpt4ts_classification.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 2\
 --random_seed 0\

python3 scripts/baselines/gpt4ts_classification.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 2\
 --random_seed 1\

python3 scripts/baselines/gpt4ts_classification.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 2\
 --random_seed 2\

python3 scripts/baselines/gpt4ts_classification.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 2\
 --random_seed 3\

python3 scripts/baselines/gpt4ts_classification.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 2\
 --random_seed 4\
