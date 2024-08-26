#!/bin/bash

export PYTHONPATH="/home/scratch/mingzhul/moment-research"


export WANDB_MODE="offline"


### 
python3 scripts/baselines/gpt4ts_classification.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 3\
 --random_seed 0\


