import torch
import json
import os


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
    # store this in the conf dict for later
    conf_args["_conf_file"] = conf_file
    conf_base = os.path.basename(conf_file)
    conf_base, _ = os.path.splitext(conf_base)
    conf_args["image_dir"] = f"images/{conf_base}"
    conf_args["arch_str"] = conf_base
    return conf_args


def save_checkpoint(model, optim, sched, epoch, loss):
    print(f"Saving model for epoch {epoch}!")
    path = model.config.get_save_str(model, epoch)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "last_train_loss": loss,
            "conf_file": model.config._conf_file,
        },
        path,
    )


def load_checkpoint(path, model, optim, sched):
    # loads checkpoint into a given model and optimizer
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    sched.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Loading model for epoch {epoch}! Last loss was {loss:.3f}")


def mat_to_vec(mat):
    torch.testing.assert_close(mat[:, 0, 1], mat[:, 1, 0], equal_nan=True)

    # print(mat[:, 0, 1], mat[:, 1, 0])

    new_shape = mat.shape[0:1] + (6,) + mat.shape[-3:]
    vec = mat.new_zeros(new_shape)

    # extract from diagonals
    vec[:, 0] = mat[:, 0, 0]
    vec[:, 1] = mat[:, 1, 1]
    vec[:, 2] = mat[:, 2, 2]

    # off-diag
    vec[:, 3] = mat[:, 0, 1]
    vec[:, 4] = mat[:, 0, 2]
    vec[:, 5] = mat[:, 1, 2]

    torch.testing.assert_close(mat, vec_to_mat(vec))
    return vec


def delta(i, j):
    # kronecker delta
    return int(i == j)


def vec_to_mat(vec):
    # torch.testing.assert_close(vec, mat_to_vec(vec_to_mat(vec)))

    # assumes vec has size [b, 6, i, j, k]
    # will return mat with size [b, 3, 3, i, j, k]
    # Convert a vector of length 6 to an equivalent symmetric 3 x 3 matrix using abaqus ordering
    new_shape = vec.shape[0:1] + (3, 3) + vec.shape[-3:]
    mat = vec.new_zeros(new_shape)

    # diagonals
    mat[:, 0, 0] = vec[:, 0]
    mat[:, 1, 1] = vec[:, 1]
    mat[:, 2, 2] = vec[:, 2]

    # off-diag
    mat[:, 0, 1] = vec[:, 3]
    mat[:, 1, 0] = vec[:, 3]
    mat[:, 0, 2] = vec[:, 4]
    mat[:, 2, 0] = vec[:, 4]
    mat[:, 1, 2] = vec[:, 5]
    mat[:, 2, 1] = vec[:, 5]

    return mat
