import argparse
from typing import Optional

import sys
sys.path.append("/zfsauton2/home/mingzhul/time-series-prompt/moment-research")


import torch

from moment.common import PATHS
from moment.tasks.forecast_finetune import ForecastFinetuning
from moment.utils.config import Config
from moment.utils.utils import control_randomness, make_dir_if_not_exists, parse_config

NOTES = "Supervised finetuning on forecasting datasets"


def forecasting(
    config_path: str = "configs/forecasting/linear_probing.yaml",
    default_config_path: str = "configs/default.yaml",
    gpu_id: int = 0,
    forecast_horizon: int = 96,
    train_batch_size: int = 64,
    val_batch_size: int = 256,
    finetuning_mode: str = "linear-probing",
    init_lr: Optional[float] = None,
    max_epoch: int = 3,
    dataset_names: str = "/TimeseriesDatasets/forecasting/autoformer/electricity.csv",
) -> None:
    config = Config(
        config_file_path=config_path, default_config_file_path=default_config_path
    ).parse()

    # Control randomness
    control_randomness(config["random_seed"])

    # Set-up parameters and defaults
    config["pred_len"] = forecast_horizon
    config["device"] = gpu_id if torch.cuda.is_available() else "cpu"
    config["checkpoint_path"] = PATHS.CHECKPOINTS_DIR
    args = parse_config(config)
    make_dir_if_not_exists(config["checkpoint_path"])

    # Setup arguments
    args.forecast_horizon = forecast_horizon
    args.train_batch_size = train_batch_size
    args.val_batch_size = val_batch_size
    args.finetuning_mode = finetuning_mode
    args.max_epoch = max_epoch
    args.dataset_names = dataset_names
    if init_lr is not None:
        args.init_lr = init_lr

    print(f"Running experiments with config:\n{args}\n")
    task_obj = ForecastFinetuning(args=args)

    # Setup a W&B Logger
    task_obj.setup_logger(notes=NOTES)
    task_obj.train()

    # End the W&B Logger
    task_obj.end_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--forecast_horizon", type=int, default=96, help="Forecasting horizon"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=256, help="Validation batch size"
    )
    parser.add_argument(
        "--finetuning_mode", type=str, default="linear-probing", help="Fine-tuning mode"
    )  # linear-probing end-to-end-finetuning
    parser.add_argument(
        "--init_lr", type=float, default=0.00005, help="Peak learning rate"
    )
    parser.add_argument(
        "--max_epoch", type=int, default=3, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        help="Name of dataset(s)",
        default="data/Timeseries-PILE/forecasting/autoformer/national_illness.csv",
    )

    args = parser.parse_args()

    forecasting(
        config_path=args.config,
        gpu_id=args.gpu_id,
        forecast_horizon=args.forecast_horizon,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        finetuning_mode=args.finetuning_mode,
        init_lr=args.init_lr,
        max_epoch=args.max_epoch,
        dataset_names=args.dataset_names,
    )
