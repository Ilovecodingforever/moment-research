#!/bin/bash

export PYTHONPATH="/zfsauton2/home/mingzhul/time-series-prompt/moment-research"


# export WANDB_MODE="offline"


### 
python3 scripts/prompt/gpt4ts_classification_prompt.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 1\
 --random_seed 0\
 --multivariate_projection 'attention'\


python3 scripts/prompt/gpt4ts_classification_prompt.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 1\
 --random_seed 1\
 --multivariate_projection 'attention'\

python3 scripts/prompt/gpt4ts_classification_prompt.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 1\
 --random_seed 2\
 --multivariate_projection 'attention'\

python3 scripts/prompt/gpt4ts_classification_prompt.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 1\
 --random_seed 3\
 --multivariate_projection 'attention'\

python3 scripts/prompt/gpt4ts_classification_prompt.py\
 --config 'configs/classification/gpt4ts.yaml'\
 --gpu_id 1\
 --random_seed 4\
 --multivariate_projection 'attention'\
