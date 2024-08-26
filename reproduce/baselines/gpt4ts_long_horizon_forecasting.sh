#!/bin/bash

# has to do this here not in .env, because dotenv is loaded after importing moment
export PYTHONPATH="/home/scratch/mingzhul/moment-research"

# TODO
export WANDB_MODE="offline"

### 
python3 scripts/baselines/gpt4ts_long_horizon_forecasting.py\
 --config 'configs/forecasting/gpt4ts_long_horizon.yaml'\
 --gpu_id 3\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/national_illness.csv'\
 --random_seed 0\
 --forecast_horizon 24\

