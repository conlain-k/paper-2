import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from inspect import currentframe, getframeinfo

import inspect

import h5py

from enum import Enum


# Python 3.9 is weird about StrEnum
class DataMode(str, Enum):
    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"


class StructureType(str, Enum):
    TWO_PHASE = "TWO_PHASE"
    CUBIC_CRYSTAL = "CRYSTAL"


SCRATCH_DIR = "/storage/home/hcoda1/3/ckelly84/scratch/"

CHECKPOINT_DIR = "checkpoints"


def mean_L1_error(true, pred):
    return (pred.detach() - true.detach()).abs().mean(dim=(-3, -2, -1))


def sync():
    # force cuda sync if cuda is available, otherwise skip
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def upsample_field(f, fac):
    # upsample z then x then y
    return (
        f.repeat_interleave(fac, dim=-3)
        .repeat_interleave(fac, dim=-2)
        .repeat_interleave(fac, dim=-1)
    )


def write_dataset_to_h5(dataset, name, h5_file):

    if isinstance(dataset, torch.Tensor):
        dataset = dataset.detach().cpu().numpy()

    # assumes batched, so that second channel is features
    chunk_size = (1,) + dataset[0].shape
    # print("chunk size is", chunk_size)

    # now make the actual datasets
    h5_file.create_dataset(
        name,
        data=dataset,
        dtype=dataset.dtype,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        chunks=chunk_size,
    )


def collect_datasets(m_base, CR):
    # get response file base
    r_base = f"{m_base}_cr{CR}_bc0_responses"

    # build up save strings from this
    MICRO_TRAIN = SCRATCH_DIR + f"micros/{m_base}_train.h5"
    MICRO_VALID = SCRATCH_DIR + f"micros/{m_base}_valid.h5"
    MICRO_TEST = SCRATCH_DIR + f"micros/{m_base}_test.h5"
    RESP_TRAIN = SCRATCH_DIR + f"outputs/{r_base}_train.h5"
    RESP_VALID = SCRATCH_DIR + f"outputs/{r_base}_valid.h5"
    RESP_TEST = SCRATCH_DIR + f"outputs/{r_base}_test.h5"
    datasets = {
        DataMode.TRAIN: {"micro_file": MICRO_TRAIN, "resp_file": RESP_TRAIN},
        DataMode.VALID: {"micro_file": MICRO_VALID, "resp_file": RESP_VALID},
        DataMode.TEST: {"micro_file": MICRO_TEST, "resp_file": RESP_TEST},
    }

    return datasets, float(CR)


# source: https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings
def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


# take a batched norm-of-average of a batch of 6-vectors corresponding to rank-2 symmetric tensors (strain, stress)
def batched_vec_avg_norm(field):
    # first take volume average, then take L2 norm for each batch entry
    # keep old shape around for broadcasting
    return (
        (field.mean(dim=(-3, -2, -1), keepdim=True) ** 2)
        .sum(dim=1, keepdim=True)
        .sqrt()
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_conf_override(conf_file):
    def load_conf_args(fname):
        with open(fname, "r") as f:
            return json.load(f)

    conf_args = load_conf_args(conf_file)
    if conf_args.get("_parent", None) is not None:
        # assumes correct relative/abs path to current dir
        # go to root and then back down
        parent_args = load_conf_override(conf_args["_parent"])
        # overwrite entries in parent
        parent_args.update(conf_args)
        # now use updated parent's version
        conf_args = parent_args

    # also stash the config file
    conf_args["_conf_file"] = conf_file

    # store this in the conf dict for later
    return conf_args


def print_activ_map(x):
    with torch.no_grad():
        x = x.detach()
        # print channel-wise power of an intermediate state x, along with line #
        cf = currentframe()
        line_num = cf.f_back.f_lineno
        filename = getframeinfo(cf.f_back).filename

        function_name = inspect.stack()[1].function

        xmin = x.min(dim=(-3, -2, -1, 0))
        xmax = x.max(dim=(-3, -2, -1, 0))
        xmean = x.mean(dim=(-3, -2, -1, 0))
        xstd = x.std(dim=(-3, -2, -1, 0))

        print(
            f"File {filename}:{line_num} ({function_name}) min is {xmin}, max is {xmax}, mean is {xmean}, std is {xstd},"
        )

        del x


def save_checkpoint(model, optim, sched, epoch, loss, best=False, ema_model=None):
    print(f"Saving model for epoch {epoch}!")
    path = model.config.get_save_str(model, epoch, best=best)

    eval_model = ema_model or model

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": eval_model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "last_train_loss": loss,
            "conf_file": model.config._conf_file,
        },
        path,
    )


def load_checkpoint(path, model, optim=None, sched=None, strict=True):
    # loads checkpoint into a given model and optimizer
    checkpoint = torch.load(path, map_location=model.config.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optim is not None:
        optim.load_state_dict(checkpoint["optimizer_state_dict"], strict=strict)
    if sched is not None:
        sched.load_state_dict(checkpoint["scheduler_state_dict"], strict=strict)

    epoch = checkpoint["epoch"]
    loss = checkpoint["last_train_loss"]

    print(f"Loading model for epoch {epoch}! Last loss was {loss:.3f}")


def plot_pred(epoch, micro, y_true, y_pred, field_name, image_dir):
    vmin_t, vmax_t = y_true.min(), y_true.max()
    vmin_p, vmax_p = y_pred.min(), y_pred.max()
    vmin = min(vmin_t, vmin_p)
    vmax = max(vmax_t, vmax_p)

    fig = plt.figure(figsize=(6, 6))

    def prep(im):
        # given an 2D array in Pytorch, prepare to plot it
        return im.detach().cpu().numpy().T

    grid = AxesGrid(
        fig,
        111,
        nrows_ncols=(2, 2),
        axes_pad=0.4,
        # share_all=True,
        # label_mode="1",
        cbar_location="right",
        cbar_mode="edge",
        cbar_pad="5%",
        cbar_size="15%",
        # direction="column"
    )

    # plot fields on top
    grid[0].imshow(
        prep(y_true),
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="lower",
    )
    grid[0].set_title("True")
    im2 = grid[1].imshow(
        prep(y_pred),
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="lower",
    )
    grid[1].set_title("Predicted")

    grid[2].imshow(prep(micro), origin="lower")
    grid[2].set_title("Micro")

    im4 = grid[3].imshow(prep((y_true - y_pred).abs()), cmap="turbo", origin="lower")
    grid[3].set_title("Absolute Residual")

    # add colorbars
    grid.cbar_axes[0].colorbar(im2)
    grid.cbar_axes[1].colorbar(im4)

    fig.suptitle(f"{field_name}", y=0.95)
    fig.tight_layout()

    os.makedirs(image_dir, exist_ok=True)

    plt.savefig(f"{image_dir}/epoch_{epoch}_{field_name}.png", dpi=300)

    plt.close(fig)
