import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt

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

def write_dataset_to_h5(dataset, name, h5_file):

    if isinstance(dataset, torch.Tensor):
        dataset = dataset.detach().cpu().numpy()

    # assumes batched, so that second channel is features
    chunk_size = (1,) + dataset[0].shape
    print("chunk size is", chunk_size)

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
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optim is not None:
        optim.load_state_dict(checkpoint["optimizer_state_dict"], strict=strict)
    if sched is not None:
        sched.load_state_dict(checkpoint["scheduler_state_dict"], strict=strict)

    epoch = checkpoint["epoch"]
    loss = checkpoint["last_train_loss"]

    print(f"Loading model for epoch {epoch}! Last loss was {loss:.3f}")


# def plot_cube(im, filepath, elev=34, azim=-30):
#     # make sure we have a numpy array
#     if isinstance(im, torch.Tensor):
#         im = im.detach().numpy()

#     Ix = im[0, :, :]
#     Iy = im[:, 0, :]
#     Iz = im[:, :, 0]

#     vmin = np.min([Ix, Iy, Iz])
#     vmax = np.max([Ix, Iy, Iz])
#     norm = plt.Normalize(vmin=vmin, vmax=vmax)

#     colors = plt.cm.turbo(norm(im))

#     Cx = colors[0, :, :]
#     Cy = colors[:, 0, :]
#     Cz = colors[:, :, 0]

#     # print(Ix, Iy, Iz)

#     # print(im.shape, Ix.shape)

#     xp, yp = Ix.shape

#     # print(xp, yp)
#     x = np.arange(0, xp, 1 - 1e-13)
#     y = np.arange(0, yp, 1 - 1e-13)
#     Y, X = np.meshgrid(y, x)

#     # print(x)

#     fig = plt.figure(figsize=(12, 9))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.dist = 6.2
#     ax.view_init(elev=elev, azim=azim)
#     ax.axis("off")

#     # print(X.shape, Y.shape, np.rot90(Ix, k=1).shape, (X - X + yp).shape)

#     # print()

#     ax.plot_surface(
#         X,
#         Y,
#         X - X + yp,
#         facecolors=np.rot90(Cx, k=1),
#         rstride=1,
#         cstride=1,
#         antialiased=True,
#         shade=False,
#         vmin=vmin,
#         vmax=vmax,
#         cmap="turbo",
#     )

#     ax.plot_surface(
#         X,
#         X - X,
#         Y,
#         facecolors=np.rot90(Cy.transpose((1, 0, 2)), k=2),
#         rstride=1,
#         cstride=1,
#         antialiased=True,
#         shade=False,
#         vmin=vmin,
#         vmax=vmax,
#         cmap="turbo",
#     )

#     ax.plot_surface(
#         X - X + xp,
#         X,
#         Y,
#         facecolors=np.rot90(Cz, k=-1),
#         rstride=1,
#         cstride=1,
#         antialiased=True,
#         shade=False,
#         vmin=vmin,
#         vmax=vmax,
#         cmap="turbo",
#     )
#     fig.tight_layout()

#     # make colorbar directly from normalization code
#     m = plt.cm.ScalarMappable(cmap=plt.cm.turbo, norm=norm)
#     m.set_array([])
#     fig.colorbar(m, ax=ax)
#     plt.savefig(filepath, transparent=True, dpi=300)
