

from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={"task_name": "reconstruction"},
)
model.init()


import os
import torch

path = '/home/scratch/mingzhul/moment-research/results/moment_checkpoints/fearless-planet-52'
model_name = 'MOMENT_checkpoint_0'


model.save_pretrained(os.path.join(path, f"{model_name}"), from_pt=True)

# with open(os.path.join(path, f"{model_name}.pth"), "wb") as f:
#     torch.save(model, f)



import os
import torch
# load from results/moment_checkpoints/fearless-planet-52/MOMENT_checkpoint_0/pytorch_model.bin
with open('/home/scratch/mingzhul/moment-research/results/moment_checkpoints/fearless-planet-52/MOMENT_checkpoint_0/pytorch_model.bin', 'rb') as f:
    model = torch.load(f)

copy = {}

copy['model_state_dict'] = model

# get rid of key
copy['model_state_dict'].pop('patch_embedding.position_embedding.pe')

path = '/home/scratch/mingzhul/moment-research/results/moment_checkpoints/fearless-planet-52'
model_name = 'MOMENT_checkpoint_0'

with open(os.path.join(path, f"{model_name}.pth"), "wb") as f:
    torch.save(copy, f)

