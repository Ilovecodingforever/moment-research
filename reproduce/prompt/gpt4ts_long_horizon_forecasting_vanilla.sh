#!/bin/bash

# has to do this here not in .env, because dotenv is loaded after importing moment
export PYTHONPATH="/zfsauton2/home/mingzhul/time-series-prompt/moment-research/"

# TODO
# export WANDB_MODE="offline"


###

# etth1
python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/ETTh1.csv'\
 --random_seed 0\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\


python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/ETTh1.csv'\
 --random_seed 1\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/ETTh1.csv'\
 --random_seed 2\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/ETTh1.csv'\
 --random_seed 3\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/ETTh1.csv'\
 --random_seed 4\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\


# etth2
python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/ETTh2.csv'\
 --random_seed 0\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/ETTh2.csv'\
 --random_seed 1\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/ETTh2.csv'\
 --random_seed 2\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/ETTh2.csv'\
 --random_seed 3\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/ETTh2.csv'\
 --random_seed 4\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\


# exchange_rate
python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 8\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/exchange_rate.csv'\
 --random_seed 0\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 8\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/exchange_rate.csv'\
 --random_seed 1\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 8\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/exchange_rate.csv'\
 --random_seed 2\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 8\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/exchange_rate.csv'\
 --random_seed 3\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 8\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/exchange_rate.csv'\
 --random_seed 4\
 --forecast_horizon 96\
 --multivariate_projection 'vanilla'\


# national_illness
python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/national_illness.csv'\
 --random_seed 0\
 --forecast_horizon 60\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/national_illness.csv'\
 --random_seed 1\
 --forecast_horizon 60\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/national_illness.csv'\
 --random_seed 4\
 --forecast_horizon 60\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/national_illness.csv'\
 --random_seed 4\
 --forecast_horizon 60\
 --multivariate_projection 'vanilla'\

python3 scripts/prompt/gpt4ts_long_horizon_forecasting_prompt.py\
 --config 'configs/prompt/gpt4ts_long_horizon.yaml'\
 --gpu_id 0\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/national_illness.csv'\
 --random_seed 4\
 --forecast_horizon 60\
 --multivariate_projection 'vanilla'\
