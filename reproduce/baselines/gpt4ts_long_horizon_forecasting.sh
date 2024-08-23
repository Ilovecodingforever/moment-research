#!/bin/bash


export PYTHONPATH="/home/scratch/mingzhul/moment-research"
export HF_HOME="/home/scratch/mingzhul/.cache/huggingface"

### 
python3 scripts/baselines/gpt4ts_long_horizon_forecasting.py\
 --config 'configs/forecasting/gpt4ts_long_horizon.yaml'\
 --gpu_id 3\
 --n_channels 7\
 --dataset_names '/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/national_illness.csv'\
 --random_seed 0\
 --forecast_horizon 24\

