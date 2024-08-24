#!/bin/bash

export PYTHONPATH="/home/scratch/mingzhul/moment-research/"
export PYTHONPATH="/Users/crl/Library/CloudStorage/Box-Box/research/Auton/LLM/moment-research/"

python3 scripts/pretraining/pretraining.py \
  --config configs/pretraining/pretrain.yaml \
  --gpu_id 0
