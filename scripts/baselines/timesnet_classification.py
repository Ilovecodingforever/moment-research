import argparse
import datetime
import os


import sys
sys.path.append("/home/scratch/mingzhul/moment-research")


import pickle as pkl

import numpy as np
import torch
from tqdm import tqdm
from yaml import dump

from moment.common import PATHS
from moment.data.base import ClassificationResults
from moment.data.dataloader import get_timeseries_dataloader
from moment.models.base import BaseModel
from moment.models.moment import MOMENT
from moment.models.statistical_classifiers import fit_svm
from moment.utils.config import Config
from moment.utils.uea_classification_datasets import uea_classification_datasets
from moment.utils.utils import control_randomness, parse_config


from moment.tasks.classify import Classification
from typing import Optional


DEFAULT_CONFIG_PATH = "configs/default.yaml"

SMALL_IMAGE_DATASETS = [
    "Crop",
    "MedicalImages",
    "SwedishLeaf",
    "FacesUCR",
    "FaceAll",
    "Adiac",
    "ArrowHead",
]
SMALL_SPECTRO_DATASETS = ["Wine", "Strawberry", "Coffee", "Ham", "Meat", "Beef"]


NOTES = "Training TimesNet for classification"



def run_experiment(
    # experiment_name: str = "unsupervised_representation_learning",
    config_path: str = None,
    gpu_id: str = "0",
    default_config_path: str = "configs/default.yaml",
    train_batch_size: int = 64,
    val_batch_size: int = 256,
    d_model: int = 16,
    d_ff: int = 16,
    init_lr: Optional[float] = None,
    random_seed: int = 13,
):
    # Load arguments and parse them
    config = Config(
        config_file_path=config_path, default_config_file_path=DEFAULT_CONFIG_PATH
    ).parse()

    config["device"] = (
        torch.device("cuda:{}".format(gpu_id)) if torch.cuda.is_available() else "cpu"
    )



    config["checkpoint_path"] = PATHS.CHECKPOINTS_DIR



    args = parse_config(config)

    # Set-up parameters and defaults
    args.config_file_path = config_path
    args.shuffle = False
    args.run_datetime = datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")



    # Setup arguments
    args.train_batch_size = train_batch_size
    args.val_batch_size = val_batch_size
    args.finetuning_mode = "end-to-end"
    args.d_model = d_model
    args.d_ff = d_ff
    if init_lr is not None:
        args.init_lr = init_lr



    # Set all randomness
    control_randomness(seed=random_seed)


    all_classification_datasets = uea_classification_datasets
    pbar = tqdm(all_classification_datasets, total=len(all_classification_datasets))
    for full_file_path_and_name in pbar:
        dataset_name = full_file_path_and_name #.split("/")[-2]
        pbar.set_postfix({"Dataset": dataset_name})

        if (dataset_name in SMALL_IMAGE_DATASETS) or (
            dataset_name in SMALL_SPECTRO_DATASETS
        ):
            args.upsampling_type = "interpolate"
        else:
            args.upsampling_type = "pad"

        args.dataset_names = dataset_name


        print(f"Running experiments with config:\n{args}\n")

        task_obj = Classification(args=args)

        # Setup a W&B Logger
        task_obj.setup_logger(notes=NOTES)
        task_obj.train()

        # End the W&B Logger
        task_obj.end_logger()




if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--experiment_name", type=str, default="unsupervised_representation_learning"
    # )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/classification/timesnet.yaml",
    )
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=256, help="Validation batch size"
    )
    parser.add_argument(
        "--init_lr", type=float, default=0.00005, help="Peak learning rate"
    )
    parser.add_argument("--d_model", type=int, default=16, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=16, help="Model dimension")

    parser.add_argument(
        "--random_seed", type=int, default=13, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    run_experiment(
        # experiment_name=args.experiment_name,
        config_path=args.config_path,
        gpu_id=args.gpu_id,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        init_lr=args.init_lr,
        d_model=args.d_model,
        d_ff=args.d_ff,
        random_seed=args.random_seed,
    )

