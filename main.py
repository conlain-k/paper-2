import os

# os.environ["WANDB_SILENT"] = "true"
from helpers import *

import wandb
import torch
from torch.utils.data import DataLoader

from dataclasses import asdict

from config import Config, DELIM
from train import train_model
from loaders import LocalizationDataset
from solvers import make_localizer
import constlaw

from math import ceil

import pprint


import argparse

parser = argparse.ArgumentParser(prog="main.py", description="Train localization NN")

parser.add_argument("-c", "--config_override", default=None)
parser.add_argument("-E", "--use_ema", action="store_true")
parser.add_argument(
    "--init_weight_scale",
    help="Initial weight scale to use",
    default=None,
    type=float,
)
parser.add_argument(
    "--lr_max", help="Initial learning rate to use", default=None, type=float
)

parser.add_argument(
    "--num_layers", help="How many FNO layers to use", default=None, type=int
)

parser.add_argument(
    "--latent_channels",
    help="Number of latent channels to use with FNO",
    default=None,
    type=int,
)

parser.add_argument(
    "--ds_type",
    help="Which dataset to train on",
    default="fixed32",
    type=str,
    choices=["fixed32", "fixed16", "fixed16_u2", "randbc32", "randcr32"],
)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

dataset_info = {
    "fixed16": {
        "m_base": "paper2_16",
        "r_base": "paper2_16_u1_responses",
        "upsamp_micro_fac": 1,
        "use_constlaw": True,
    },
    "fixed32": {
        "m_base": "paper2_32",
        "r_base": "paper2_32_u1_responses",
        "upsamp_micro_fac": 1,
        "use_constlaw": True,
    },
    "fixed16_u2": {
        "m_base": "paper2_16",
        "r_base": "paper2_16_u2_responses",
        "upsamp_micro_fac": 2,
        "use_constlaw": True,
    },
    "randbc32": {
        "m_base": "paper2_32_randbc",
        "r_base": "paper2_32_randbc_u1_responses",
        "upsamp_micro_fac": 1,
        "use_constlaw": False,
    },
    "randcr32": {
        "m_base": "paper2_32_randcr",
        "r_base": "paper2_32_randcr_u1_responses",
        "upsamp_micro_fac": 1,
        "use_constlaw": False,
    },
    "hiCR32": {
        "m_base": "paper2_32_hiCR",
        "r_base": "paper2_32_hiCR_u1_responses",
        "upsamp_micro_fac": 1,
        "use_constlaw": False,
    },
    "poly64": {
        "m_base": "cubic_combined",
        "r_base": "cubic_combined_u1_responses",
        "upsamp_micro_fac": 1,
        "use_constlaw": False,
        "is_poly": True,
    },
}


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

datasets = None


# load train and validation sets from given file base
def load_data(ds_info, mode, constlaw=None, loader_args={}):

    datasets, _ = collect_datasets(ds_info["m_base"], r_base=ds_info["r_base"])

    print(f"Currently loading files {datasets[mode]} for mode {mode}")

    dataset = LocalizationDataset(
        **datasets[mode],
        upsamp_micro_fac=ds_info["upsamp_micro_fac"],
        is_poly=ds_info.get("is_poly", False),
    )

    if ds_info["use_constlaw"]:
        dataset.assignConstlaw(constlaw)

    # dump into dataloader
    loader = DataLoader(
        dataset, pin_memory=True, persistent_workers=False, **loader_args
    )

    print(f"Dataset {mode} has {len(dataset)} instances!")
    print("Data type is", dataset[0][0].dtype)
    return loader


if __name__ == "__main__":
    args = parser.parse_args()

    if args.config_override:
        conf_args = load_conf_override(args.config_override)
        config = Config(**conf_args, use_EMA=args.use_ema)
    else:
        config = Config()

    if args.lr_max:
        config.lr_max = args.lr_max
    if args.init_weight_scale:
        config.fno_args["init_weight_scale"] = args.init_weight_scale
    if args.num_layers:
        config.fno_args["modes"] = [-1] * args.num_layers
    if args.latent_channels:
        config.fno_args["latent_channels"] = args.latent_channels

    config.train_dataset_name = args.ds_type

    # TODO this assumes CR is hard coded in constlaw
    E_VALS = [120.0, 100.0 * 120.0]
    NU_VALS = [0.3, 0.3]
    E_BAR = [0.001, 0, 0, 0, 0, 0]

    constlaw = constlaw.StrainToStress_2phase(E_VALS, NU_VALS)

    ds_info = dataset_info[args.ds_type]

    # get train and validation datasets
    train_loader = load_data(
        ds_info,
        DataMode.TRAIN,
        constlaw=constlaw,
        loader_args=config.loader_args[DataMode.TRAIN],
    )
    valid_loader = load_data(
        ds_info,
        DataMode.VALID,
        constlaw=constlaw,
        loader_args=config.loader_args[DataMode.VALID],
    )

    # NOTE assumes that second-to-last strain dimension is a spatial one
    num_voxels = train_loader.dataset[0][0].shape[-2]
    config.num_voxels = num_voxels

    # config.fno_args["modes"] = modes_new
    # cache training data name for future reference

    model = make_localizer(config, constlaw)
    # set scalings using reference stiffness and constlaw
    model.compute_scalings(E_BAR)

    model = model.to(DEVICE)
    model.inf_device = DEVICE

    print(DELIM)
    print(model)
    print(count_parameters(model))
    print(DELIM)

    print(f"\nConfig is:\n{pprint.pformat(asdict(config))}\n")

    bigdict = {}
    bigdict.update(asdict(config))
    bigdict.update({"model_arch": repr(model), "num_params": count_parameters(model)})

    # set up logging
    wandb.init(
        # set the wandb project where this run will be logged
        project="paper-2",
        # track hyperparameters and run metadata
        config=bigdict,
        name=config.arch_str,
    )

    train_model(model, config, train_loader, valid_loader)

    # finalize logging
    wandb.finish()
