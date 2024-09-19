#!/bin/bash

export PYTHONPATH="/zfsauton2/home/mingzhul/time-series-prompt/moment-research"


# export WANDB_MODE="offline"


### 
python3 scripts/baselines/gpt4ts_classification.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 0\
 --random_seed 0\
 --lora 1\

python3 scripts/baselines/gpt4ts_classification.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 0\
 --random_seed 1\
 --lora 1\

python3 scripts/baselines/gpt4ts_classification.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 0\
 --random_seed 2\
 --lora 1\

python3 scripts/baselines/gpt4ts_classification.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 0\
 --random_seed 3\
 --lora 1\

python3 scripts/baselines/gpt4ts_classification.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 0\
 --random_seed 4\
 --lora 1\
