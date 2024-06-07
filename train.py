from numpy import s_, unravel_index
import torch

import time
import wandb

from config import Config, DELIM, LossSet
from helpers import *

from constlaw import *

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

FIELD_IND = 0
PLOT_IND = 1744


def PRMS_loss(y_true, y_pred, scale=None):
    mseloss = torch.nn.MSELoss()
    if scale is not None:
        # rescale outputs before computing loss
        y_true = y_true.squeeze() / scale
        y_pred = y_pred.squeeze() / scale
    loss = mseloss(y_true.squeeze(), y_pred.squeeze())
    loss = torch.sqrt(loss)
    # get percent RMSE
    loss = loss * 100.0
    return loss


def relative_PRMS(y_true, y_pred, scale=None):
    # ignore scale param
    # take spatial average
    scale_rel = (y_pred**2).mean(dim=(-3, -2, -1), keepdim=True).sqrt()

    # upscale important of higher-norm instances
    return PRMS_loss(y_true / scale_rel, y_pred / scale_rel, scale=None)


# compute an energy-type loss for two strain-like fields
def compute_energy_loss(model, e_true, e_pred, micro, add_deriv=False, ret_deriv=False):
    C_field = torch.einsum("bhxyz, hrc -> brcxyz", micro, model.constlaw.stiffness_mats)
    resid = e_true - e_pred

    Nx = resid.shape[-1]
    # domain is zero to 1
    L = 1
    # grid spacing
    h = L / Nx

    # contract strain and stress dimensions
    resid_energy = torch.einsum("brxyz, brcxyz, bcxyz -> bxyz", resid, C_field, resid)

    loss = resid_energy

    # print(f"Resid loss is {resid_energy.mean():4f}")

    if add_deriv:
        # take finite differences
        resid_grad = central_diff_3d(resid, h=1)
        resid_grad = torch.stack(resid_grad, dim=-1)

        # also sum over last dimension (spatial deriv index) to get squared 2-norm of vector field
        resid_grad_energy = (
            torch.einsum(
                "brxyzd, brcxyz, bcxyzd -> bxyz", resid_grad, C_field, resid_grad
            )
            / 10.0
        )

        # print(f"Grad loss is {resid_grad_energy.mean():4f}")
        loss += resid_grad_energy

    if ret_deriv:

        return resid_energy, resid_grad_energy

    return loss


def H1_loss(y_true, y_pred, scale=None, deriv_scale=10):
    # resid = y_true - y_pred

    diff_resid = central_diff_3d(y_true - y_pred)
    # stack along a new batch dimension (for now, will quickly get summed out)
    diff_resid = torch.stack(diff_resid, dim=1)

    L2_loss = PRMS_loss(y_true, y_pred, scale=scale)
    diff_loss = PRMS_loss(diff_resid, 0 * diff_resid, scale=scale) * deriv_scale

    # print(f"L2 is {L2_loss}, diff is {diff_loss}")

    return L2_loss + diff_loss


def plot_worst(epoch, model, micro, strain_true):
    print(f"Saving fig for epoch {epoch}")

    micro = micro.to(model.config.device)
    strain_true = strain_true.to(model.config.device)

    with torch.inference_mode():
        output = model(micro)

    if model.config.return_resid and model.config.use_deq:
        (strain_pred, resid) = output
    else:
        strain_pred = output
        resid = 0 * strain_pred

    # recompute quantities
    stress_pred, stress_polar_pred, energy_pred = compute_quants(
        model, strain_pred, micro
    )
    stress_true, stress_polar_true, energy_true = compute_quants(
        model, strain_true, micro
    )

    # also compute VM stresses at that location
    VM_stress_pred = VMStress(stress_pred)
    VM_stress_true = VMStress(stress_true)

    stressdiv_true = stressdiv(stress_true, use_FFT_deriv=True)
    stressdiv_pred = stressdiv(stress_pred, use_FFT_deriv=True)

    print(stressdiv_true.shape, stressdiv_pred.shape)

    print("Stressdiv stats")
    print(
        f"true: min {stressdiv_true.min()}, max {stressdiv_true.max()}, mean {stressdiv_true.mean()}, std {stressdiv_true.std()}"
    )
    print(
        f"pred: min {stressdiv_pred.min()}, max {stressdiv_pred.max()}, mean {stressdiv_pred.mean()}, std {stressdiv_pred.std()}"
    )

    # get worst L1 error
    ind_max = (
        torch.argmax((strain_true - strain_pred)[:, FIELD_IND].abs().max())
        .detach()
        .cpu()
    )

    ind_max = unravel_index(ind_max, strain_true[:, FIELD_IND].detach().cpu().shape)

    # Plot z=const slice
    sind = s_[:, :, ind_max[-1]]

    plot_pred(
        epoch,
        micro[0, 0][sind],
        strain_true[0, FIELD_IND][sind] / model.strain_scaling,
        strain_pred[0, FIELD_IND][sind] / model.strain_scaling,
        "strain",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        0 * resid[0, FIELD_IND][sind] / model.strain_scaling,
        resid[0, FIELD_IND][sind] / model.strain_scaling,
        "resid",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        stress_true[0, FIELD_IND][sind] / model.stress_scaling,
        stress_pred[0, FIELD_IND][sind] / model.stress_scaling,
        "stress",
        model.config.image_dir,
    )
    plot_pred(
        epoch,
        micro[0, 0][sind],
        stress_polar_true[0, FIELD_IND][sind] / model.stress_scaling,
        stress_polar_pred[0, FIELD_IND][sind] / model.stress_scaling,
        "stress_polarization",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        VM_stress_true[0, FIELD_IND][sind] / model.stress_scaling,
        VM_stress_pred[0, FIELD_IND][sind] / model.stress_scaling,
        "VM_stress",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        stressdiv_true[0, 0][sind] / model.stress_scaling,
        stressdiv_pred[0, 0][sind] / model.stress_scaling,
        "stressdiv",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        energy_true[0, 0][sind] / model.energy_scaling,
        energy_pred[0, 0][sind] / model.energy_scaling,
        "energy",
        model.config.image_dir,
    )


def plot_pred(epoch, micro, y_true, y_pred, field_name, image_dir):
    vmin_t, vmax_t = y_true.min(), y_true.max()
    vmin_p, vmax_p = y_pred.min(), y_pred.max()
    vmin = vmin_t  # min(vmin_t, vmin_p)
    vmax = vmax_t  # max(vmax_t, vmax_p)

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

    # bbox_ax_top = ax[0, 1].get_position()
    # bbox_ax_bottom = ax[1, 1].get_position()

    # cbar_im1a_ax = fig.add_axes(
    #     [1.01, bbox_ax_top.y0, 0.02, bbox_ax_top.y1 - bbox_ax_top.y0]
    # )
    # cbar_im1a = plt.colorbar(im2, cax=cbar_im1a_ax)

    # cbar_im2a_ax = fig.add_axes(
    #     [1.01, bbox_ax_bottom.y0, 0.02, bbox_ax_bottom.y1 - bbox_ax_bottom.y0]
    # )
    # cbar_im1a = plt.colorbar(im4, cax=cbar_im2a_ax)

    fig.suptitle(f"{field_name}", y=0.95)
    fig.tight_layout()

    os.makedirs(image_dir, exist_ok=True)

    plt.savefig(f"{image_dir}/epoch_{epoch}_{field_name}.png", dpi=300)

    plt.close(fig)


def compute_quants(model, strain, micros):
    stress = model.constlaw(strain, micros)
    stress_polar = model.constlaw.stress_pol(strain, micros)
    energy = compute_strain_energy(strain, stress)

    return stress, stress_polar, energy


def compute_losses(model, micros, quants_pred, quants_true, resid):

    strain_pred, stress_pred, energy_pred = quants_pred
    strain_true, stress_true, energy_true = quants_true

    strain_loss = H1_loss(
        strain_true,
        strain_pred,
        scale=model.strain_scaling,
        deriv_scale=model.config.H1_deriv_scaling,
    )
    stress_loss = H1_loss(
        stress_true,
        stress_pred,
        scale=model.stress_scaling,
        deriv_scale=model.config.H1_deriv_scaling,
    )
    # energy_loss = H1_loss(
    #     energy_true,
    #     energy_pred,
    #     scale=model.energy_scaling,
    #     deriv_scale=model.config.H1_deriv_scaling,
    # )

    energy_loss = compute_energy_loss(
        model, strain_true, strain_pred, micros, add_deriv=True
    ).mean()

    resid_loss = torch.tensor(0.0)
    stressdiv_loss = torch.tensor(0.0)

    if model.config.return_resid and not model.pretraining:
        resid_loss = compute_energy_loss(model, resid, 0.0, micros, add_deriv=False)

    # if model.config.compute_stressdiv:
    #     stressdiv_loss = (
    #         100
    #         * (stressdiv(stress_pred, use_FFT_deriv=True) ** 2).mean().sqrt()
    #         / model.stress_scaling
    #     )

    losses = LossSet(
        model.config, strain_loss, stress_loss, energy_loss, resid_loss, stressdiv_loss
    )

    return losses.detach(), losses.compute_total()


def valid_pass(model, epoch, valid_loader):
    # zero out losses to star
    running_loss = LossSet(config=model.config)
    L1_strain_err = 0
    L1_VM_stress_err = 0

    micro_worst = None
    strain_true_worst = None
    strain_pred_worst = None
    diff_worst = 0
    ind_worst = None

    homog_err = 0
    mean_homog = 0
    for _, (micros, strain_true, _) in enumerate(valid_loader):
        micros = micros.to(model.config.device)
        # only predict first component
        strain_true = strain_true.to(model.config.device)

        with torch.inference_mode():
            output = model(micros)

        if model.config.return_resid and model.config.use_deq:
            (strain_pred, resid) = output
        else:
            strain_pred = output
            resid = 0 * strain_pred

        # TODO this should not be necessary
        strain_pred = strain_pred.detach()

        assert not torch.isnan(strain_pred).any()

        stress_pred, _, energy_pred = compute_quants(model, strain_pred, micros)
        stress_true, _, energy_true = compute_quants(model, strain_true, micros)

        # find worst e_xx strain pred
        ind_max = (
            torch.argmax((strain_true - strain_pred)[:, FIELD_IND].abs()).detach().cpu()
        )

        # get index for each dimension
        ind_max = unravel_index(ind_max, strain_true[:, FIELD_IND].detach().cpu().shape)

        losses_e, _ = compute_losses(
            model,
            micros,
            (strain_pred, stress_pred, energy_pred),
            (strain_true, stress_true, energy_true),
            resid,
        )

        C11_true = est_homog(strain_true, stress_true, (0, 0))
        C11_pred = est_homog(strain_pred, stress_pred, (0, 0))

        homog_err += (C11_true - C11_pred).abs().sum()

        mean_homog += C11_true.sum()

        # accumulate loss
        running_loss = running_loss + losses_e
        VM_stress_pred = VMStress(stress_pred)
        VM_stress_true = VMStress(stress_true)

        # Compute running L1 errors
        # average out over space
        LSE = (strain_pred - strain_true)[:, 0].abs().mean(dim=(-3, -2, -1)).detach()
        # rescale so that each instance contributes equally
        LSE *= (len(micros) / len(valid_loader.dataset)) / model.strain_scaling

        # now average out over batch size
        LSE = LSE.mean()
        LVE = (VM_stress_pred - VM_stress_true).abs().mean(dim=(-3, -2, -1)).detach()
        # rescale, and also divide by true mean absolute VM stress
        LVE *= (len(micros) / len(valid_loader.dataset)) / VM_stress_true.abs().mean(
            dim=(-3, -2, -1)
        )
        LVE = LVE.mean()

        L1_strain_err += LSE
        L1_VM_stress_err += LVE

    # divide out number of batches (simple normalization)
    running_loss /= len(valid_loader)

    m, e_true, _ = valid_loader.dataset[PLOT_IND : PLOT_IND + 1]

    # now valid loop is done
    plot_worst(
        epoch,
        model,
        m,
        e_true,
    )

    # print("last batch vm metrics")

    # print("VM_stress_true", VM_stress_true.min(), VM_stress_true.max(), VM_stress_true.mean(), VM_stress_true.std())
    # print("VM_stress_pred", VM_stress_pred.min(), VM_stress_pred.max(), VM_stress_pred.mean(), VM_stress_pred.std())
    homog_err_abs = homog_err / len(valid_loader.dataset)
    mean_homog = mean_homog / len(valid_loader.dataset)

    wandb.log(
        {
            "epoch": epoch,
            "total_valid_loss": running_loss.compute_total(),
            "validation_losses": running_loss.to_dict(),
            "homog_err_rel": homog_err_abs / mean_homog,
            "valid_exx_err": L1_strain_err,
            "valid_VM_err": L1_VM_stress_err,
        }
    )

    # print some metrics on the epoch
    print(f"Epoch {epoch}, validation: {running_loss}")
    print(f"Normalized e_xx absolute error is: {L1_strain_err * 100:.5} %")
    print(f"Normalized VM stress absolute error is: {L1_VM_stress_err * 100:.5} %")
    print(f"Abs homog err is {homog_err_abs}, rel is {homog_err_abs / mean_homog}")
    print(
        f"Pred range: min {strain_pred[:, 0].min():.5}, max {strain_pred[:, 0].max():.5}, mean {strain_pred[:, 0].mean():.5}, std {strain_pred[:, 0].std():.5}"
    )
    print(
        f"True range: min {strain_true[:, 0].min():.5}, max {strain_true[:, 0].max():.5}, mean {strain_true[:, 0].mean():.5}, std {strain_true[:, 0].std():.5}"
    )


def train_model(model, config, train_loader, valid_loader):
    model = model.to(config.device)  # move to GPU

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr_max, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.num_epochs, eta_min=1e-8
    )

    print(model.strain_scaling, model.stress_scaling, model.energy_scaling)

    for e in range(config.num_epochs):
        # only pretrain for given # epochs
        if e >= config.num_pretrain_epochs and model.pretraining:
            print(f"\nDisabling pretrain mode at epoch {e}\n")
            model.pretraining = False
            # also rebuild optimizer to reset internal states
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=optimizer.param_groups[0]["lr"],
                weight_decay=config.weight_decay,
            )

        print(DELIM)

        # Run a validation pass before training this epoch
        model.eval()

        # time validation pass
        start = time.time()

        valid_pass(model, e, valid_loader)

        diff = time.time() - start

        print(f"Validation pass took {diff}s")
        print(DELIM)

        model.train()
        running_loss = 0

        # now time training pass
        start = time.time()

        for batch_ind, (micros, strain_true, batch_stress_true) in enumerate(
            train_loader
        ):
            micros = micros.to(config.device)
            # only predict first component
            strain_true = strain_true.to(config.device)

            optimizer.zero_grad()

            output = model(micros)

            if config.use_deq and config.return_resid:
                (strain_pred, resid) = output
            else:
                strain_pred = output
                resid = 0 * strain_pred

            stress_pred, stress_polar_pred, energy_pred = compute_quants(
                model, strain_pred, micros
            )
            stress_true, stress_polar_true, energy_true = compute_quants(
                model, strain_true, micros
            )

            # now compute losses
            losses_e, total_loss = compute_losses(
                model,
                micros,
                (strain_pred, stress_pred, energy_pred),
                (strain_true, stress_true, energy_true),
                resid,
            )

            # backprop now
            total_loss.backward()
            # now do grad clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_mag)
            optimizer.step()

            # now accumulate losses for future
            running_loss += total_loss.detach() / len(train_loader)

            # printing once per epoch
            if batch_ind == 0:
                # print split on first batch to track progress
                print(f"Epoch {e}, batch {batch_ind}: {losses_e}")
                print(
                    f"Normalized e_xx absolute error is: {(strain_pred - strain_true)[:, 0].abs().mean() / model.strain_scaling * 100:.5} %"
                )
                print(
                    f"Pred range: min {strain_pred[:, 0].min():.5}, max {strain_pred[:, 0].max():.5}, mean {strain_pred[:, 0].mean():.5}, std {strain_pred[:, 0].std():.5}"
                )
                print(
                    f"True range: min {strain_true[:, 0].min():.5}, max {strain_true[:, 0].max():.5}, mean {strain_true[:, 0].mean():.5}, std {strain_true[:, 0].std():.5}"
                )

        # end epoch
        scheduler.step()

        diff = time.time() - start

        wandb.log(
            {
                "epoch": e,
                "total_train_loss": running_loss,
                "train_iter_time": diff,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        # clean out cuda cache
        if model.config.use_fancy_iter and model.config.use_deq:
            torch.cuda.empty_cache()

        print(f"Training pass took {diff}s")

        print(f"Epoch {e}: instance-average loss was {running_loss:5f}")

        # save_checkpoint(model, optimizer, scheduler, e, running_loss)
        print(DELIM)
        print("\n")
