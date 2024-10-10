from numpy import s_, unravel_index

import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from inspect import currentframe, getframeinfo

import inspect

import h5py

import shutil

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


def collect_datasets(m_base, CR, r_base=None):
    # get response file base
    if r_base is None:
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


def print_activ_map(x):
    with torch.no_grad():
        x = x.detach()
        # print channel-wise power of an intermediate state x, along with line #
        cf = currentframe()
        line_num = cf.f_back.f_lineno
        filename = getframeinfo(cf.f_back).filename

        function_name = inspect.stack()[1].function

        xmean = x.mean(dim=(-3, -2, -1, 0))
        xstd = x.std(dim=(-3, -2, -1, 0))
        xrms = (x**2).sum(dim=(-3, -2, -1, 0)).sqrt()

        print(
            f"File {filename}:{line_num} ({function_name}) rms is {xrms}, mean is {xmean}, std is {xstd},"
        )

        del x


def save_checkpoint(
    model,
    optim,
    sched,
    epoch,
    loss,
    best=False,
    ema_model=None,
    path_override=None,
    backup_prev=True,
):
    print(f"Saving model for epoch {epoch}!")
    if path_override is None:
        path = model.config.get_save_str(model, epoch, best=best)
    else:
        path = path_override

    if os.path.isfile(path) and backup_prev:
        # always save at least one backup
        root, ext = os.path.splitext(path)
        backup_path = f"{root}_prev{ext}"
        shutil.move(path, backup_path)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict() if optim is not None else {},
            "scheduler_state_dict": sched.state_dict() if sched is not None else {},
            "last_train_loss": loss,
            "conf_file": model.config._conf_file,
            # save ema model in addition (if one exists)
            "ema_state_dict": ema_model.state_dict() if ema_model is not None else {},
        },
        path,
    )


def load_checkpoint(path, model, optim=None, sched=None, strict=True, device=None):
    # loads checkpoint into a given model and optimizer
    checkpoint = torch.load(path, map_location=device)

    del_keys = [
        # "eps_bar",
        # "scaled_average_strain",
        "constlaw.C_ref",
        "constlaw.S_ref",
        "greens_op.constlaw.C_ref",
        "greens_op.constlaw.S_ref",
        # "greens_op.G_freq",
        # "greens_op.constlaw.stiffness_mats",
        # "greens_op.constlaw.compliance_mats",
    ]
    for k in del_keys:
        if k in checkpoint["model_state_dict"].keys():
            del checkpoint["model_state_dict"][k]

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # which model is used for evaluation?
    eval_model = model

    if optim is not None:
        optim.load_state_dict(checkpoint["optimizer_state_dict"], strict=strict)
    if sched is not None:
        sched.load_state_dict(checkpoint["scheduler_state_dict"], strict=strict)

    # load ema model as well
    if checkpoint.get("ema_state_dict"):
        ema_model = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.99)
        )
        # copy in pointer to const info
        ema_model.overrideConstlaw(model.constlaw)
        ema_model.load_state_dict(checkpoint["ema_state_dict"], strict=strict)
        eval_model = model

    epoch = checkpoint["epoch"]
    loss = checkpoint["last_train_loss"]

    print(
        f"Loading model for epoch {epoch}, original conf was {checkpoint['conf_file']}! Last loss was {loss:.3f}"
    )

    return eval_model


def plot_pred(epoch, micro, y_true, y_pred, field_name, image_dir):
    vmin_t, vmax_t = y_true.min(), y_true.max()
    vmin_p, vmax_p = y_pred.min(), y_pred.max()
    # use outer extremes
    vmin = min(vmin_t, vmin_p)
    vmax = max(vmax_t, vmax_p)

    # if min and max are too close, override so that we get nice consistent plotting
    if (vmin - vmax) / vmax < 1e-3:
        # half-width of colorbar around constant value
        delta = 0.1 * vmax
        vmin -= delta
        vmax += delta

    # print("vminmax", vmin, vmax)

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


def check_constlaw(constlaw, C_field, strain, stress):
    # Check that our constlaw gives same stress-strain relation as FEA

    # print("C ref", constlaw.C_ref)

    stress_comp = constlaw(C_field, strain)

    err = (stress_comp - stress).abs()

    ind_max = torch.argmax(err).detach().cpu()

    ind_max = unravel_index(ind_max, err.shape)

    # print(ind_max)

    b, c, x, y, z = ind_max

    # print("Worst ind is", ind_max)

    # Plot z=const slice
    sind = s_[:, :, z]

    # print(f"Each component err mean is: {err.mean(dim=(0, -1, -2, -3))}")

    ok = torch.allclose(stress_comp, stress, rtol=1e-8, atol=1e-2)

    # if things don't match, plot for debugging purposes
    if not ok:

        print(f"Err mean {err.mean()} std {err.std()} min {err.min()} max {err.max()}")

        print("stress true", stress[b, :, x, y, z])
        print("stress comp", stress_comp[b, :, x, y, z])

        print("strain vals", strain[b, :, x, y, z])
        print("strain means", strain.mean((0, -3, -2, -1)))

        plot_pred(
            -1,
            C_field[b, 0, 0][sind],
            stress[b, c][sind],
            stress_comp[b, c][sind],
            "stresscomp",
            "images/",
        )

        plot_pred(
            -1,
            C_field[b, 0, 0][sind],
            0 * err[b, c][sind],
            err[b, c][sind],
            "err_stresscompx",
            "images/",
        )
        plot_pred(
            -1,
            C_field[b, 0, 0][sind],
            0 * strain[b, c][sind],
            strain[b, c][sind],
            "straincomp",
            "images/",
        )

    # make sure things match
    assert ok
