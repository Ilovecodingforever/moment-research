import argparse
import datetime
import os
import pickle as pkl



import sys
sys.path.append("/Users/crl/Library/CloudStorage/Box-Box/research/Auton/LLM/moment-research/")



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

from moment.utils.dtw_metric import dtw, accelerated_dtw



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



def get_dataloaders(args):
    args.dataset_names = args.full_file_path_and_name
    args.data_split = "train"
    train_dataloader = get_timeseries_dataloader(args=args)
    args.data_split = "test"
    test_dataloader = get_timeseries_dataloader(args=args)
    args.data_split = "val"
    val_dataloader = get_timeseries_dataloader(args=args)
    return train_dataloader, test_dataloader, val_dataloader



def _create_results_dir(experiment_name):
    results_path = os.path.join(PATHS.RESULTS_DIR, experiment_name)
    os.makedirs(results_path, exist_ok=True)
    return results_path


def _save_config(args, results_path):
    with open(os.path.join(results_path, "config.yaml"), "w") as f:
        dump(vars(args), f)


def run_experiment(
    experiment_name: str = "unsupervised_representation_learning",
    config_path: str = None,
    gpu_id: str = "0",
):
    # Load arguments and parse them
    config = Config(
        config_file_path=config_path, default_config_file_path=DEFAULT_CONFIG_PATH
    ).parse()

    config["device"] = (
        torch.device("cuda:{}".format(gpu_id)) if torch.cuda.is_available() else "cpu"
    )
    args = parse_config(config)

    # Set-up parameters and defaults
    args.config_file_path = config_path
    args.shuffle = False
    args.run_datetime = datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")

    # Set all randomness
    control_randomness(seed=args.random_seed)

    # Save the experiment arguments and metadata
    results_path = _create_results_dir(experiment_name)
    _save_config(args, results_path)

    all_classification_datasets = uea_classification_datasets

    datasets_with_failed_experiments = []

    pbar = tqdm(all_classification_datasets, total=len(all_classification_datasets))
    for full_file_path_and_name in pbar:
        dataset_name = full_file_path_and_name.split("/")[-2]
        pbar.set_postfix({"Dataset": dataset_name})

        if (dataset_name in SMALL_IMAGE_DATASETS) or (
            dataset_name in SMALL_SPECTRO_DATASETS
        ):
            args.upsampling_type = "interpolate"
        else:
            args.upsampling_type = "pad"

        args.full_file_path_and_name = full_file_path_and_name
        args.task_name = "classification"
        args.dataset_names = dataset_name

        print(f"Running experiments with config:\n{args}\n")

        # train_dataloader, test_dataloader, val_dataloader = get_dataloaders(args)


        # TODO: hyperparams?
        # TODO: make sure no padding and subsampling. maybe directly read from files

        from moment.data.load_data import load_from_tsfile

        dataset_name = full_file_path_and_name.split("/")[-2]
        series = full_file_path_and_name.split("/")[-1].split("_")[0]
        root_path = full_file_path_and_name.split("/")[:-1]

        path = os.path.join("/".join(root_path), series + "_TRAIN.ts")
        train_dataset, train_labels = load_from_tsfile(path)

        path = os.path.join("/".join(root_path), series + "_TEST.ts")
        test_dataset, test_labels = load_from_tsfile(path)


        trues = []
        preds = []
        for test_data, test_label in tqdm(zip(test_dataset, test_labels)):

            trues.append(test_label)
            best_scores = np.inf

            for train_data, train_label in tqdm(zip(train_dataset, train_labels)):

                manhattan_distance = lambda x, y: np.abs(x - y)

                x = train_data.reshape(-1, 1)
                y = test_data.reshape(-1, 1)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                if d < best_scores:
                    best_scores = d
                    label = train_label

            preds.append(label)

        accuracy = np.mean(np.array(trues) == np.array(preds))

        with open(os.path.join(results_path, "results.txt"), "a") as f:
            f.write(f"{dataset_name} accuracy: {accuracy}\n")




if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, default="unsupervised_classification_dtw"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/classification/unsupervised_representation_learning.yaml",
    )
    # parser.add_argument("--gpu_id", type=str, default="0")

    args = parser.parse_args()

    run_experiment(
        experiment_name=args.experiment_name,
        config_path=args.config_path,
        # gpu_id=args.gpu_id,
    )
