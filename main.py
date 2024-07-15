import torch
from torch.utils.data import DataLoader

from dataclasses import asdict

from config import Config, DELIM
from helpers import *
from train import train_model
from loaders import LocalizationDataset
from solvers import make_localizer

from math import ceil

import pprint

import wandb, os

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

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

os.environ["WANDB_SILENT"] = "true"

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

m_base = "paper2_16"
r_base = "paper2_16_u1_responses"
UPSAMP_MICRO_FAC = 1


datasets, CR = collect_datasets(m_base, CR_str, r_base=r_base)


E_VALS = [120.0, CR * 120.0]
NU_VALS = [0.3, 0.3]
E_BAR = [0.001, 0, 0, 0, 0, 0]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# load train and validation sets from given file base
def load_data(config, mode):
    global ref_val
    print("Loading {mode} data! This may take a bit ...")
    print(f"Currently loading files {datasets[mode]} for mode {mode}")

    dataset_train = LocalizationDataset(
        **datasets[mode],
        upsamp_micro_fac=UPSAMP_MICRO_FAC,
        swap_abaqus=USING_ABAQUS_DATASET,
    )

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

    if args.lr_max:
        config.lr_max = args.lr_max
    if args.init_weight_scale:
        config.fno_args["init_weight_scale"] = args.init_weight_scale

    # profile_forward(model)

    train_loader = load_data(config, DataMode.TRAIN)
    valid_loader = load_data(config, DataMode.VALID)

    # NOTE assumes that second-to-last strain dimension is a spatial one
    num_voxels = train_loader.dataset[0][1].shape[-2]
    config.num_voxels = num_voxels

    modes = config.fno_args["modes"]

    # if # modes is negative or too big for given data, only keep amount that data can provide
    full_num_modes = ceil(num_voxels / 2)

    modes_new = [
        full_num_modes if (m == -1 or m > full_num_modes) else m for m in modes
    ]

    config.fno_args["modes"] = modes_new

    model = make_localizer(config)

    # now we can set constitutive parameters
    model.setConstParams(E_VALS, NU_VALS, E_BAR)

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
