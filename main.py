import torch
from torch.utils.data import DataLoader

from dataclasses import asdict

from config import Config, DELIM
from helpers import *
from train import train_model
from loaders import LocalizationDataset
from solvers import make_localizer

import pprint

import wandb, os

import argparse

parser = argparse.ArgumentParser(prog="main.py", description="Train localization NN")

parser.add_argument("-c", "--config_override", default=None)
parser.add_argument("-E", "--use_ema", action="store_true")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

os.environ["WANDB_SILENT"] = "true"

import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

# this also defines the dataset we are loading
CR_str = "100.0"
m_base = "paper2_smooth"

datasets, CR = collect_datasets(m_base, CR_str)

E_VALS = [120.0, CR * 120.0]
NU_VALS = [0.3, 0.3]
E_BAR = [0.001, 0, 0, 0, 0, 0]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# load train and validation sets from given file base
def load_data(config, mode):
    global ref_val
    print("Loading {mode} data! This may take a bit ...")
    print(f"Currently loading files {datasets[mode]} for mode {mode}")

    dataset_train = LocalizationDataset(**datasets[mode])

    # dump into dataloader
    loader_train = DataLoader(
        dataset_train, pin_memory=True, **config.loader_args[mode]
    )

    print("Data loaded!")
    print(f"Training on {len(dataset_train)} instances!")
    # print(f"Validating on {len(dataset_valid)} instances!")
    print("Data type is", dataset_train[0][0].dtype)
    return loader_train  # , loader_valid


def tiny_forward(model, micros):
    for i in range(5):
        y = model(micros)
        loss = (y**2).mean()
        # loss.backward()


def profile_forward(model):
    from torch.profiler import profile, record_function, ProfilerActivity

    model.config.return_resid = False

    # generate random set of microstructures
    micros = torch.randn(128, 2, 31, 31, 31).cuda()
    model.eval()
    model = model.cuda()

    # print(torch.cuda.memory_summary())
    torch.cuda.synchronize()

    tiny_forward(model, micros)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        with record_function("model_inference"):
            # with torch.no_grad():
            tiny_forward(model, micros)
            # loss.backward()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print(torch.cuda.memory_summary())


if __name__ == "__main__":
    args = parser.parse_args()

    if args.config_override:
        conf_args = load_conf_override(args.config_override)
        # print(conf_args)
        config = Config(**conf_args, use_EMA=args.use_ema)
    else:
        config = Config()

    model = make_localizer(config)
    model.setConstParams(E_VALS, NU_VALS, E_BAR)

    model = model.to(DEVICE)
    # profile_forward(model)

    print(DELIM)
    print(model)
    print(count_parameters(model))
    print(DELIM)

    train_loader = load_data(config, DataMode.TRAIN)

    valid_loader = load_data(config, DataMode.VALID)

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
