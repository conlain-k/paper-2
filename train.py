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


def sync():
    # force cuda sync if cuda is available, otherwise skip
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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


def H1_loss(y_true, y_pred, scale=None, deriv_scale=10):
    # resid = y_true - y_pred

    diff_resid = central_diff_3d(y_true - y_pred)
    # stack along a new batch dimension (for now, will quickly get summed out)
    diff_resid = torch.stack(diff_resid, dim=1)

    L2_loss = PRMS_loss(y_true, y_pred, scale=scale)
    diff_loss = PRMS_loss(diff_resid, 0 * diff_resid, scale=scale) / deriv_scale

    # print(f"L2 is {L2_loss}, diff is {diff_loss}")

    return L2_loss + diff_loss


def plot_worst(epoch, model, micro, strain_true, strain_pred, ind_worst, resid):
    print(f"Saving fig for epoch {epoch}, min ind is {ind_worst}")
    assert ind_worst is not None

    # recompute quantities
    stress_pred, stress_polar_pred, energy_pred = compute_quants(
        model, strain_pred, micro
    )
    stress_true, stress_polar_true, energy_true = compute_quants(
        model, strain_true, micro
    )

    err_energy = compute_strain_energy(
        strain_true - strain_pred, stress_true - stress_pred
    )

    # also compute VM stresses at that location
    VM_stress_pred = VMStress(stress_pred)
    VM_stress_true = VMStress(stress_true)

    stressdiv_true = stressdiv(stress_true, use_FFT_deriv=False)
    stressdiv_pred = stressdiv(stress_pred, use_FFT_deriv=False)

    print(stressdiv_true.shape, stressdiv_pred.shape)

    print("Stressdiv stats")
    print(
        f"true: min {stressdiv_true.min()}, max {stressdiv_true.max()}, mean {stressdiv_true.mean()}, std {stressdiv_true.std()}"
    )
    print(
        f"pred: min {stressdiv_pred.min()}, max {stressdiv_pred.max()}, mean {stressdiv_pred.mean()}, std {stressdiv_pred.std()}"
    )
    # Plot z=const slice
    sind = s_[:, :, ind_worst[-1]]

    plot_pred(
        epoch,
        micro[0, 0][sind],
        strain_true[0, FIELD_IND][sind] / model.constlaw.strain_scaling,
        strain_pred[0, FIELD_IND][sind] / model.constlaw.strain_scaling,
        "strain",
        model.config.image_dir,
    )
    plot_pred(
        epoch,
        micro[0, 0][sind],
        stress_true[0, FIELD_IND][sind] / model.constlaw.stress_scaling,
        stress_pred[0, FIELD_IND][sind] / model.constlaw.stress_scaling,
        "stress",
        model.config.image_dir,
    )
    plot_pred(
        epoch,
        micro[0, 0][sind],
        stress_polar_true[0, FIELD_IND][sind] / model.constlaw.stress_scaling,
        stress_polar_pred[0, FIELD_IND][sind] / model.constlaw.stress_scaling,
        "stress_polarization",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        VM_stress_true[0, FIELD_IND][sind] / model.constlaw.stress_scaling,
        VM_stress_pred[0, FIELD_IND][sind] / model.constlaw.stress_scaling,
        "VM_stress",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        stressdiv_true[0, 0][sind] / model.constlaw.stress_scaling,
        stressdiv_pred[0, 0][sind] / model.constlaw.stress_scaling,
        "stressdiv",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        energy_true[0, 0][sind] / model.constlaw.energy_scaling,
        energy_pred[0, 0][sind] / model.constlaw.energy_scaling,
        "energy",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        0 * err_energy[0, 0][sind],
        err_energy[0, 0][sind] / model.constlaw.energy_scaling,
        "error_energy",
        model.config.image_dir,
    )
    plot_pred(
        epoch,
        micro[0, 0][sind],
        0 * resid[0, FIELD_IND][sind],
        resid[0, FIELD_IND][sind] / model.constlaw.strain_scaling,
        "resid",
        model.config.image_dir,
    )


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
    C_field = model.constlaw.compute_C_field(micros)
    stress = model.constlaw(strain, C_field)
    stress_polar = model.constlaw.stress_pol(strain, C_field)
    energy = compute_strain_energy(strain, stress)

    return stress, stress_polar, energy


def compute_losses(model, quants_pred, quants_true, resid):

    strain_pred, stress_pred, energy_pred = quants_pred
    strain_true, stress_true, energy_true = quants_true

    strain_loss = PRMS_loss(
        strain_true,
        strain_pred,
        scale=model.constlaw.strain_scaling,
        # deriv_scale=model.config.H1_deriv_scaling,
    )
    stress_loss = PRMS_loss(
        stress_true,
        stress_pred,
        scale=model.constlaw.stress_scaling,
        # deriv_scale=model.config.H1_deriv_scaling,
    )
    energy_loss = PRMS_loss(
        energy_true,
        energy_pred,
        scale=model.constlaw.energy_scaling,
        # deriv_scale=model.config.H1_deriv_scaling,
    )

    err_energy = compute_strain_energy(
        strain_true - strain_pred, stress_true - stress_pred
    )

    err_energy_loss = (
        100 * (err_energy**2).mean().sqrt() / model.constlaw.energy_scaling
    )

    resid_loss = 0
    compat_loss = 0

    if model.config.return_resid:
        resid_loss = 100 * (resid**2).mean().sqrt() / model.constlaw.strain_scaling

    if model.config.compute_stressdiv:

        err_compat, _ = model.greens_op.compute_residuals(strain_pred, stress_pred)
        compat_loss = 100 * batched_vec_avg_norm(err_compat).mean()

    losses = LossSet(
        model.config,
        strain_loss,
        stress_loss,
        energy_loss,
        err_energy_loss,
        resid_loss,
        compat_loss,
    )

    return losses.detach(), losses.compute_total()


def eval_pass(model, epoch, eval_loader, data_mode, ema_model=None):

    # if we're given an ema model, use that for forward passes
    eval_model = ema_model or model

    eval_model.eval()

    # zero out losses to star
    running_loss = LossSet(config=model.config)
    L1_strain_err = 0
    L1_VM_stress_err = 0

    micro_worst = None
    strain_true_worst = None
    strain_pred_worst = None
    resid_worst = None
    ind_worst = None

    homog_err = 0
    mean_homog = 0

    running_time_cost = 0
    for ind, (micros, strain_true, _) in enumerate(eval_loader):

        sync()
        if data_mode == DataMode.TEST:
            print(f"Testing batch {ind} of {len(eval_loader)}")
        t0 = time.time()
        micros = micros.to(model.config.device)
        # only predict first component
        strain_true = strain_true.to(model.config.device)

        with torch.inference_mode():
            output = eval_model(micros)

        sync()
        t1 = time.time()

        if data_mode == DataMode.TEST:
            print(f"This batch took {t1 - t0} seconds")

        # build up how long it's taken to run all samples
        running_time_cost += t1 - t0

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

        diff = (strain_true - strain_pred)[:, FIELD_IND][ind_max].abs()

        # just show last batch
        if True:  # diff > diff_worst:
            # location of worst in batch
            b_ind = ind_max[0]
            micro_worst = micros[b_ind : b_ind + 1].detach()
            strain_true_worst = strain_true[b_ind : b_ind + 1].detach()
            strain_pred_worst = strain_pred[b_ind : b_ind + 1].detach()
            resid_worst = resid[b_ind : b_ind + 1].detach()
            ind_worst = ind_max[1:]

        losses_e, _ = compute_losses(
            model,
            (strain_pred, stress_pred, energy_pred),
            (strain_true, stress_true, energy_true),
            resid,
        )

        C11_true = est_homog(strain_true, stress_true, (0, 0))
        C11_pred = est_homog(strain_pred, stress_pred, (0, 0))

        homog_err += (C11_true - C11_pred).abs().sum()

        mean_homog += C11_true.sum()

        # print("C11_true", C11_true)
        # print("C11_pred", C11_pred)

        # print(losses_e)

        # accumulate loss
        running_loss = running_loss + losses_e
        VM_stress_pred = VMStress(stress_pred)
        VM_stress_true = VMStress(stress_true)

        # Compute running L1 errors
        # average out over space
        LSE = (strain_pred - strain_true)[:, 0].abs().mean(dim=(-3, -2, -1)).detach()
        # rescale so that each instance contributes equally
        LSE *= (len(micros) / len(eval_loader.dataset)) / model.constlaw.strain_scaling

        # now average out over batch size
        LSE = LSE.mean()
        LVE = (VM_stress_pred - VM_stress_true).abs().mean(dim=(-3, -2, -1)).detach()
        # rescale, and also divide by true mean absolute VM stress
        LVE *= (len(micros) / len(eval_loader.dataset)) / VM_stress_true.abs().mean(
            dim=(-3, -2, -1)
        )
        LVE = LVE.mean()

        L1_strain_err += LSE
        L1_VM_stress_err += LVE

    # divide out number of batches (simple normalization)
    running_loss /= len(eval_loader)

    # now valid loop is done
    plot_worst(
        epoch,
        model,
        micro_worst,
        strain_true_worst,
        strain_pred_worst,
        ind_worst,
        resid_worst,
    )

    # print("last batch vm metrics")

    # print("VM_stress_true", VM_stress_true.min(), VM_stress_true.max(), VM_stress_true.mean(), VM_stress_true.std())
    # print("VM_stress_pred", VM_stress_pred.min(), VM_stress_pred.max(), VM_stress_pred.mean(), VM_stress_pred.std())
    homog_err_abs = homog_err / len(eval_loader.dataset)
    mean_homog = mean_homog / len(eval_loader.dataset)

    if data_mode == DataMode.VALID:
        wandb.log(
            {
                "epoch": epoch,
                f"total_{data_mode}_loss": running_loss.compute_total(),
                f"{data_mode}_losses": running_loss.to_dict(),
                "homog_err_rel": homog_err_abs / mean_homog,
                f"{data_mode}_exx_err": L1_strain_err,
                f"{data_mode}_VM_err": L1_VM_stress_err,
            }
        )

    # print some metrics on the epoch
    print(f"Epoch {epoch}, {data_mode} loss: {running_loss}")
    print(f"Normalized e_xx absolute error is: {L1_strain_err * 100:.5} %")
    print(f"Normalized VM stress absolute error is: {L1_VM_stress_err * 100:.5} %")
    print(f"Abs homog err is {homog_err_abs}, rel is {homog_err_abs / mean_homog}")
    print(
        f"Pred range: min {strain_pred[:, 0].min():.5}, max {strain_pred[:, 0].max():.5}, mean {strain_pred[:, 0].mean():.5}, std {strain_pred[:, 0].std():.5}"
    )
    print(
        f"True range: min {strain_true[:, 0].min():.5}, max {strain_true[:, 0].max():.5}, mean {strain_true[:, 0].mean():.5}, std {strain_true[:, 0].std():.5}"
    )

    valid_time_per_micro = 1000 * running_time_cost / len(eval_loader.dataset)

    print(
        f"Inference only cost {running_time_cost} s total, {valid_time_per_micro:.2f} ms per instance"
    )

    return running_loss.compute_total(), valid_time_per_micro


def train_model(model, config, train_loader, valid_loader):
    model = model.to(config.device)  # move to GPU

    if config.use_EMA:
        ema_model = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.99)
        )
    else:
        ema_model = None  # makes eval code simpler

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr_max, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.num_epochs, eta_min=1e-6
    )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     config.lr_max,
    #     epochs=config.num_epochs,
    #     steps_per_epoch=1,
    #     pct_start=0.2,  # first 20% is increase, then anneal
    # )

    print(
        "Scalings",
        model.constlaw.strain_scaling,
        model.constlaw.stiffness_scaling,
        model.constlaw.stress_scaling,
        model.constlaw.energy_scaling,
    )

    for e in range(config.num_epochs):
        print(DELIM)

        # Run a validation pass before training this epoch
        # time validation pass
        start = time.time()

        _, valid_time_per_micro = eval_pass(
            model, e, valid_loader, DataMode.VALID, ema_model=ema_model
        )

        diff = time.time() - start

        print(f"Validation pass took {diff}s")
        print(DELIM)

        model.train()
        running_loss = 0
        best_loss = 1e6

        # now time training pass
        start = time.time()

        for batch_ind, (micros, strain_true, batch_stress_true) in enumerate(
            train_loader
        ):
            micros = micros.to(config.device)
            # only predict first component
            strain_true = strain_true.to(config.device)

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
                (strain_pred, stress_pred, energy_pred),
                (strain_true, stress_true, energy_true),
                resid,
            )

            # backprop now
            total_loss.backward()
            # now do grad clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_mag)
            optimizer.step()

            # update averaged model after the first epoch (so that we discard the initial noisy model)
            if config.use_EMA and e > 0:
                ema_model.update_parameters(model)

            # now accumulate losses for future
            running_loss += total_loss.detach() / len(train_loader)

            # printing once per epoch
            if batch_ind == 0:
                # print split on first batch to track progress
                print(f"Epoch {e}, batch {batch_ind}: {losses_e}")
                print(
                    f"Normalized e_xx absolute error is: {(strain_pred - strain_true)[:, 0].abs().mean() / model.constlaw.strain_scaling * 100:.5} %"
                )
                print(
                    f"Strain Pred range: min {strain_pred[:, 0].min():5}, max {strain_pred[:, 0].max():5}, mean {strain_pred[:, 0].mean():5}, std {strain_pred[:, 0].std():5}"
                )
                print(
                    f"Strain True range: min {strain_true[:, 0].min():5}, max {strain_true[:, 0].max():5}, mean {strain_true[:, 0].mean():5}, std {strain_true[:, 0].std():5}"
                )
                print(
                    f"Stress Pred range: min {stress_pred[:, 0].min():5}, max {stress_pred[:, 0].max():5}, mean {stress_pred[:, 0].mean():5}, std {stress_pred[:, 0].std():5}"
                )
                print(
                    f"Stress True range: min {stress_true[:, 0].min():5}, max {stress_true[:, 0].max():5}, mean {stress_true[:, 0].mean():5}, std {stress_true[:, 0].std():5}"
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
                "valid_time_per_micro": valid_time_per_micro,
            }
        )

        # clean out cuda cache
        if model.config.use_fancy_iter and model.config.use_deq:
            torch.cuda.empty_cache()

        print(f"Training pass took {diff}s")

        print(f"Epoch {e}: instance-average loss was {running_loss:5f}")

        if running_loss < best_loss:
            best_loss = running_loss.detach()
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                e,
                running_loss,
                best=True,
                ema_model=ema_model,
            )

        print(DELIM)
        print("\n")
