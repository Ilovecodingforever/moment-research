import argparse
import datetime
import os
import pickle as pkl



import sys
sys.path.append("/zfsauton2/home/mingzhul/time-series-prompt/moment-research")





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



NOTES = "Training gpt4ts for classification"



def run_experiment(
    experiment_name: str = "gpt4ts_classification",
    config_path: str = None,
    gpu_id: str = "0",
    random_seed: int = 13,
):
    # Load arguments and parse them
    config = Config(
        config_file_path=config_path, default_config_file_path=DEFAULT_CONFIG_PATH
    ).parse()

    config["device"] = gpu_id if torch.cuda.is_available() else "cpu"



    config["checkpoint_path"] = PATHS.CHECKPOINTS_DIR

    PATHS.RESULTS_DIR = PATHS.RESULTS_DIR + "/" + str(random_seed)



    args = parse_config(config)

    # Set-up parameters and defaults
    args.config_file_path = config_path
    args.shuffle = False
    args.run_datetime = datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")

    # Set all randomness
    # control_randomness(seed=args.random_seed)
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
        default="configs/classification/gpt4ts.yaml",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")


    parser.add_argument(
        "--random_seed", type=int, default=13, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--lora", type=str, default=False
    )



    args = parser.parse_args()

    run_experiment(
        # experiment_name=args.experiment_name,
        config_path=args.config_path,
        gpu_id=args.gpu_id,
        random_seed=args.random_seed,
    )
