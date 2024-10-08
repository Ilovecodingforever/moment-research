#!/bin/bash

export PYTHONPATH="/zfsauton2/home/mingzhul/time-series-prompt/moment-research"


export WANDB_MODE="offline"


### 
python3 scripts/baselines/gpt4ts_imputation.py\
 --config 'configs/imputation/gpt4ts_train.yaml'\
 --gpu_id 3\
 --n_channels 7\
 --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/national_illness.csv'\
 --random_seed 0\



# ### ETTh1
# python3 scripts/baselines/gpt4ts_imputation.py\
#  --config 'configs/imputation/gpt4ts_train.yaml'\
#  --gpu_id 0\
#  --n_channels 7\
#  --dataset_names 'data/Timeseries-PILE/forecasting/autoformer/ETTh1.csv'

# ### ETTh2
# python3 ../../scripts/baselines/gpt4ts_imputation.py\
#  --config '../../configs/imputation/gpt4ts_train.yaml'\
#  --gpu_id 0\
#  --n_channels 7\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTh2.csv'

# ### ETTm1
# python3 ../../scripts/baselines/gpt4ts_imputation.py\
#  --config '../../configs/imputation/gpt4ts_train.yaml'\
#  --gpu_id 0\
#  --n_channels 7\
#  --dataset_names '/TimeseriesDatasets/forecasting/autoformer/ETTm1.csv'
