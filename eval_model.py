import torch
from torch.utils.data import DataLoader

from dataclasses import asdict

from config import Config, DELIM
from helpers import *
from solvers import make_localizer
from main import load_data
from train import eval_pass

import pprint

import os

import argparse

from torch.nn.parallel import DistributedDataParallelCPU

parser = argparse.ArgumentParser(prog="main.py", description="Train localization NN")

parser.add_argument("-c", "--config_override", default=None)
parser.add_argument("-f", "--checkpoint_file", required=True)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


# this also defines the dataset we are loading
CR_str = "50.0"
m_base = "paper2"

datasets, CR = collect_datasets(m_base, CR_str)

E_VALS = [120.0, CR * 120.0]
NU_VALS = [0.3, 0.3]
E_BAR = [0.001, 0, 0, 0, 0, 0]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    args = parser.parse_args()

    if args.config_override:
        conf_args = load_conf_override(args.config_override)
        # print(conf_args)
        config = Config(**conf_args)
    else:
        config = Config()

    config.device = DEVICE

    model = make_localizer(config)
    model.setConstParams(E_VALS, NU_VALS, E_BAR)

    print(DELIM)
    print(model)
    print(count_parameters(model))
    print(DELIM)

    # train_loader = load_data(config, DataMode.TRAIN)

    test_loader = load_data(config, DataMode.TEST)

    print(f"\nConfig is:\n{pprint.pformat(asdict(config))}\n")

    # load in saved weigths
    load_checkpoint(args.checkpoint_file, model)

    model = model.to(config.device)
    model.eval()

    # model = torch.compile(model)

    print(f"Using {torch.get_num_threads()} threads!")
    print(f"Using {torch.get_num_interop_threads()} interop threads!")

    # if not torch.cuda.is_available():
    #     model = DistributedDataParallelCPU(model)

    torch.set_num_threads(1)

    eval_pass(model, -1, test_loader, DataMode.TEST)
