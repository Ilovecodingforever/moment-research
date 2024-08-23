#!/bin/bash

export PYTHONPATH="/home/scratch/mingzhul/moment-research/"

python3 scripts/pretraining/pretraining.py \
  --config configs/pretraining/pretrain.yaml \
  --gpu_id 0
