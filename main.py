import os
# os.environ["WANDB_SILENT"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

import wandb
import torch
from torch.utils.data import DataLoader

from dataclasses import asdict

from config import Config, DELIM
from helpers import *
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

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

# this also defines the dataset we are loading
CR_str = "100.0"
m_base = "paper2_smooth"
r_base = None
USING_ABAQUS_DATASET = True
USING_ABAQUS_DATASET = True
UPSAMP_MICRO_FAC = None

m_base = "paper2_16"
r_base = "paper2_16_u2_responses"
USING_ABAQUS_DATASET = False

UPSAMP_MICRO_FAC = 2

# m_base = "paper2_16"
# r_base = "paper2_16_u1_responses"
# UPSAMP_MICRO_FAC = 1

# m_base = "paper2_32"
# r_base = "paper2_32_u1_responses"
# UPSAMP_MICRO_FAC = 1


datasets, CR = collect_datasets(m_base, CR_str, r_base=r_base)


E_VALS = [120.0, CR * 120.0]
NU_VALS = [0.3, 0.3]
E_BAR = [0.001, 0, 0, 0, 0, 0]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# load train and validation sets from given file base
def load_data(config, mode):
    global ref_val
    print(f"Currently loading files {datasets[mode]} for mode {mode}")

    dataset = LocalizationDataset(
        **datasets[mode],
        upsamp_micro_fac=UPSAMP_MICRO_FAC,
        swap_abaqus=USING_ABAQUS_DATASET,
    )

    # dump into dataloader
    loader = DataLoader(dataset, pin_memory=True, **config.loader_args[mode])

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

    # get train and validation datasets
    train_loader = load_data(config, DataMode.TRAIN)
    valid_loader = load_data(config, DataMode.VALID)

    # NOTE assumes that second-to-last strain dimension is a spatial one
    num_voxels = train_loader.dataset[0][0].shape[-2]
    config.num_voxels = num_voxels

    # config.fno_args["modes"] = modes_new
    # cache training data name for future reference
    config.train_dataset_name = datasets[DataMode.TRAIN]

    model = make_localizer(config)

    constlaw=constlaw.StrainToStress_2phase(E_VALS, NU_VALS, E_BAR)

    # now we can set constitutive parameters
    model.setConstlaw(constlaw)

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
