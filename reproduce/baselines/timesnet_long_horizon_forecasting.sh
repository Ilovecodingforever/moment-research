#!/bin/bash


# has to do this here not in .env, because dotenv is loaded after importing moment
export PYTHONPATH="/Users/crl/Library/CloudStorage/Box-Box/research/Auton/LLM/moment-research/"

# TODO
export WANDB_MODE="offline"

python3 scripts/baselines/timesnet_long_horizon_forecasting.py\
 --config 'configs/forecasting/timesnet_long_horizon.yaml'\
 --gpu_id 3\
 --d_model 64\
 --d_ff 64\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/national_illness.csv'\
 --random_seed 4\
 --forecast_horizon 60\

