#!/bin/bash




# has to do this here not in .env, because dotenv is loaded after importing moment
export PYTHONPATH="/home/scratch/mingzhul/moment-research"

# TODO
export WANDB_MODE="offline"

python3 scripts/baselines/timesnet_classification.py\
 --config 'configs/classification/timesnet.yaml'\
 --gpu_id 2\
 --d_model 64\
 --d_ff 64\
 --random_seed 0

 
#  --n_channels 7\
#  --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/national_illness.csv'\


