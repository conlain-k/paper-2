from numpy import s_, unravel_index
import torch

import time
import wandb

from config import Config, DELIM, LossSet
from helpers import *

from constlaw import *

FIELD_IND = 0

PLOT_IND = 1744


def MAE_Loss(y_true, y_pred, scale=None):
    # add in scale factors before computing loss
    maeloss = torch.nn.L1Loss()
    if scale is not None:
        # rescale outputs before computing loss
        y_true = y_true.squeeze() / scale
        y_pred = y_pred.squeeze() / scale
    loss = maeloss(y_true.squeeze(), y_pred.squeeze())
    # loss = loss.sqrt() * 100
    return loss


def MSE_Loss(y_true, y_pred, scale=None):
    # add in scale factors before computing loss
    mseloss = torch.nn.MSELoss()
    if scale is not None:
        # rescale outputs before computing loss
        y_true = y_true.squeeze() / scale
        y_pred = y_pred.squeeze() / scale
    loss = mseloss(y_true.squeeze(), y_pred.squeeze())
    # loss = loss.sqrt() * 100
    return loss


def relative_mse(y_true, y_pred, scale=None):
    # average over channels and space, but not batch idx

    eps = 1e-1

    val = (y_true - y_pred) ** 2 / (y_true**2 + eps**2)

    return val.mean()

    ybar = y_true.mean(dim=(-3, -2, -1, 1), keepdim=True)
    scale_rel = ((y_true - ybar) ** 2).mean(dim=(-3, -2, -1, 1), keepdim=True).sqrt()
    return MSE_Loss(y_true / scale_rel, y_pred / scale_rel, scale=None)


def deriv_loss(y_true, y_pred):
    # compute loss of spatial derivatives (part of H1 loss)

    # # get spatial discr
    # Nx = y_true.shape[-1]
    # # domain is zero to 1
    # L = 1
    # # grid spacing
    # h = L / Nx

    # take finite differences of each field
    # override discr size to scale things nicely
    diff_true = central_diff_3d(y_true, h=1)
    diff_pred = central_diff_3d(y_pred, h=1)

    # concatenate along channel dim
    diff_true = torch.cat(diff_true, dim=1)
    diff_pred = torch.cat(diff_pred, dim=1)

    # print(diff_true.shape)

    # mean squared Frob-norm of error in stress gradients
    diff_loss = ((diff_true - diff_pred) ** 2).sum(dim=(1)).mean()

    return diff_loss


def plot_worst(epoch, model, micro, strain_true):

    micro = micro.to(model.config.device)
    strain_true = strain_true.to(model.config.device)

    # evaluate model
    strain_pred = model(micro)

    if model.config.return_resid and model.config.use_deq:
        # also grab resid if it's an option
        strain_pred, resid = strain_pred
    else:
        resid = 0 * strain_pred

    # get worst L1 error
    ind_max = (
        torch.argmax((strain_true - strain_pred)[:, FIELD_IND].abs().max())
        .detach()
        .cpu()
    )

    ind_max = unravel_index(ind_max, strain_true[:, FIELD_IND].detach().cpu().shape)

    C_field = model.constlaw.compute_C_field(micro)

    # recompute quantities
    stress_pred, stress_polar_pred, energy_pred = compute_quants(
        model, strain_pred, C_field
    )
    stress_true, stress_polar_true, energy_true = compute_quants(
        model, strain_true, C_field
    )

    # also compute VM stresses at that location
    VM_stress_pred = VMStress(stress_pred)
    VM_stress_true = VMStress(stress_true)

    # also compute VM stresses at that location
    stressdiv_pred = stressdiv(stress_pred, use_FFT_deriv=False)
    stressdiv_true = stressdiv(stress_true, use_FFT_deriv=False)

    err_energy = compute_strain_energy(
        strain_true - strain_pred, stress_true - stress_pred
    )

    print(
        f"\tStressdiv true mean {stressdiv_true.abs().mean()} std {stressdiv_true.abs().std()}"
    )
    print(
        f"\tStressdiv pred mean {stressdiv_pred.abs().mean()} std {stressdiv_pred.abs().std()}"
    )

    compat_err_true, equib_err_true = model.greens_op.compute_residuals(
        strain_true, stress_true
    )
    compat_err_pred, equib_err_pred = model.greens_op.compute_residuals(
        strain_pred, stress_pred
    )

    C11_true = est_homog(strain_true, stress_true, (0, 0)).squeeze()
    C11_pred = est_homog(strain_pred, stress_pred, (0, 0)).squeeze()
    print(
        f"Saving fig for epoch {epoch}, plotting micro {PLOT_IND} near {ind_max}, homog err is {(C11_true - C11_pred).abs():5f}"
    )

    print("Compatibility error stats")
    print(
        f"true: min {compat_err_true.min()}, max {compat_err_true.max()}, mean {compat_err_true.mean()}, std {compat_err_true.std()}"
    )
    print(
        f"pred: min {compat_err_pred.min()}, max {compat_err_pred.max()}, mean {compat_err_pred.mean()}, std {compat_err_pred.std()}"
    )
    # Plot z=const slice
    sind = s_[:, :, ind_max[-1]]

    plot_pred(
        epoch,
        micro[0, 0][sind],
        strain_true[0, FIELD_IND][sind],  # / model.constlaw.strain_scaling,
        strain_pred[0, FIELD_IND][sind],  # / model.constlaw.strain_scaling,
        "strain",
        model.config.image_dir,
    )
    plot_pred(
        epoch,
        micro[0, 0][sind],
        stress_true[0, FIELD_IND][sind],  # / model.constlaw.stress_scaling,
        stress_pred[0, FIELD_IND][sind],  # / model.constlaw.stress_scaling,
        "stress",
        model.config.image_dir,
    )
    plot_pred(
        epoch,
        micro[0, 0][sind],
        stress_polar_true[0, FIELD_IND][sind],  # / model.constlaw.stress_scaling,
        stress_polar_pred[0, FIELD_IND][sind],  # / model.constlaw.stress_scaling,
        "stress_polarization",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        VM_stress_true[0, FIELD_IND][sind],  # / model.constlaw.stress_scaling,
        VM_stress_pred[0, FIELD_IND][sind],  # / model.constlaw.stress_scaling,
        "VM_stress",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        compat_err_true[0, 0][sind],
        compat_err_pred[0, 0][sind],
        "compat_err",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        stressdiv_true[0, 0][sind],
        stressdiv_pred[0, 0][sind],
        "stressdiv",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        equib_err_true[0, 0][sind],
        equib_err_pred[0, 0][sind],
        "equib_err",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        energy_true[0, 0][sind],  # / model.constlaw.energy_scaling,
        energy_pred[0, 0][sind],  # / model.constlaw.energy_scaling,
        "energy",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        0 * err_energy[0, 0][sind],  # / model.constlaw.energy_scaling,
        err_energy[0, 0][sind],  # / model.constlaw.energy_scaling,
        "err_energy",
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        strain_pred[0, FIELD_IND][sind],  # / model.constlaw.strain_scaling,
        (resid + strain_pred)[0, FIELD_IND][sind],  # / model.constlaw.strain_scaling,
        "resid",
        model.config.image_dir,
    )


def compute_quants(model, strain, C_field):
    stress = model.constlaw(strain, C_field)
    stress_polar = model.constlaw.stress_pol(strain, C_field)
    energy = compute_strain_energy(strain, stress)

    return stress, stress_polar, energy


# def loss(y_true, y_pred, inner_prod=None, add_sobolev=False, scaling=1):
#     err = y_true - y_pred
#     # if given a weighted inner product, use that (use 2norm)
#     if inner_prod:
#         loss = inner_prod(y_true, y_pred) / scaling
#     else:
#         # contract over spatial dims first to get a scalar
#         loss = (err**2).sum(dim=1) / scaling

#     # now average over space and batch
#     mean_loss = loss.mean()

#     if add_sobolev:


def compute_losses(model, strain_pred, strain_true, C_field, resid):

    stress_pred, _, energy_pred = compute_quants(model, strain_pred, C_field)
    stress_true, _, energy_true = compute_quants(model, strain_true, C_field)

    # if model.config.use_C0_weighted_loss:
    #     # use C0 inner product
    strain_loss = (
        model.constlaw.C0_norm((strain_true - strain_pred)).mean()
        / model.constlaw.energy_scaling
    )

    stress_loss = (
        model.constlaw.S0_norm((stress_true - stress_pred)).mean()
        / model.constlaw.energy_scaling
    )

    # else:
    #     # use L2 inner product
    #     strain_loss = (
    #         ((strain_true - strain_pred) / model.constlaw.strain_scaling) ** 2
    #     ).mean()
    #     stress_loss = (
    #         ((stress_true - stress_pred) / model.constlaw.stress_scaling) ** 2
    #     ).mean()

    # if model.config.use_sobolev_loss:
    #     deriv_loss_strain = (
    #         deriv_loss(
    #             strain_true / model.constlaw.strain_scaling,
    #             strain_pred / model.constlaw.strain_scaling,
    #         )
    #         * 0.2
    #     )
    #     deriv_loss_stress = (
    #         deriv_loss(
    #             stress_true / model.constlaw.stress_scaling,
    #             stress_pred / model.constlaw.stress_scaling,
    #         )
    #         * 5
    #     )

    # print(f"strain {strain_loss:5f} deriv {deriv_loss_strain:5f}")

    # print(
    #     f"strain {strain_loss:5f} deriv {deriv_loss_strain:5f} stress {stress_loss:5f} deriv {deriv_loss_stress:5f}"
    # )

    # strain_loss += deriv_loss_strain  # * model.config.H1_deriv_scaling
    # stress_loss += deriv_loss_stress  # * model.config.H1_deriv_scaling

    stressdiv_pred = stressdiv(stress_pred, use_FFT_deriv=False)
    stressdiv_true = stressdiv(stress_true, use_FFT_deriv=False)

    energy_loss = (
        (stressdiv_pred - stressdiv_true) ** 2
    ).mean() / model.constlaw.stress_scaling

    # no term here
    # strain_loss = 0 * stress_loss

    strain_loss = (
        compute_strain_energy(
            strain_true - strain_pred, stress_true - stress_pred
        ).mean()
        / model.constlaw.energy_scaling
    )

    print(f"strain {strain_loss:5f} stress {stress_loss:5f} stressdiv {energy_loss:5f}")
    # energy_loss = MSE_Loss(
    #     energy_true,
    #     energy_pred,
    #     scale=model.constlaw.energy_scaling,
    #     # deriv_scale=model.config.H1_deriv_scaling,
    # )

    resid_loss = torch.as_tensor(0.0)
    compat_loss = torch.as_tensor(0.0)

    if model.config.return_resid:
        # compute energy in residual and add that term
        resid_loss = (
            model.constlaw.C0_norm(resid).mean() / model.constlaw.energy_scaling
        )

    if model.config.compute_compat_err:

        err_compat, _ = model.greens_op.compute_residuals(strain_pred, stress_pred)
        # compute RMSE, averaged across channels/batch, converted to percent
        compat_loss = 100 * (err_compat**2).mean(dim=(-3, -2, -1)).sqrt().mean()

    losses = LossSet(
        model.config,
        strain_loss,
        stress_loss,
        energy_loss,
        resid_loss,
        compat_loss,
    )

    return losses.detach(), losses.compute_total()


def eval_pass(model, epoch, eval_loader, data_mode, ema_model=None):

    # print("CREF", model.constlaw.C_ref)
    # print("SREF", model.constlaw.S_ref)

    # if we're given an ema model, use that for forward passes
    eval_model = ema_model or model

    eval_model.eval()

    # zero out losses to star
    running_loss = LossSet(config=model.config)
    L1_strain_err = 0
    L1_VM_stress_err = 0

    homog_err = 0
    mean_homog = 0

    running_time_cost = 0
    for ind, (micros, strain_true, stress_true) in enumerate(eval_loader):

        # force synchronization for timing
        sync()
        if data_mode == DataMode.TEST:
            print(f"Testing batch {ind} of {len(eval_loader)}")

        t0 = time.time()

        # include data migration in timing
        micros = micros.to(model.config.device)
        strain_true = strain_true.to(model.config.device)
        stress_true = stress_true.to(model.config.device)

        # now evaluate model
        with torch.inference_mode():
            output = eval_model(micros)

        # how long did that pass take?
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

        C_field = model.constlaw.compute_C_field(micros)

        losses_e, _ = compute_losses(
            model,
            strain_pred,
            strain_true,
            C_field,
            resid,
        )

        C11_true = est_homog(strain_true, stress_true, (0, 0))

        stress_pred = model.constlaw(strain_pred, C_field)
        C11_pred = est_homog(strain_pred, stress_pred, (0, 0))

        # worst_homog_ind = torch.argmax((C11_true - C11_pred).abs())

        # print(
        #     f"Batch {ind} worst is {worst_homog_ind} err {(C11_true - C11_pred).abs()[worst_homog_ind]:5f}"
        # )

        homog_err += (C11_true - C11_pred).abs().sum()

        mean_homog += C11_true.sum()

        # accumulate loss
        running_loss = running_loss + losses_e
        VM_stress_pred = VMStress(stress_pred)
        VM_stress_true = VMStress(stress_true)

        # Compute running L1 errors
        # average out over space
        LSE = (
            mean_L1_error(strain_pred[:, 0], strain_true[:, 0])
            / model.constlaw.strain_scaling
        )

        # now average out over each instance in train set
        LSE = LSE.sum() / len(eval_loader.dataset)
        LVE = (
            mean_L1_error(VM_stress_pred, VM_stress_true) / VM_stress_true.abs().mean()
        )
        # print(LVE.shape, LVE)
        # now average out over each instance in train set
        LVE = LVE.sum() / len(eval_loader.dataset)

        L1_strain_err += LSE
        L1_VM_stress_err += LVE

    # divide out number of batches (simple normalization)
    running_loss /= len(eval_loader)

    m, e_true, _ = eval_loader.dataset[PLOT_IND : PLOT_IND + 1]

    # now valid loop is done
    plot_worst(
        epoch,
        model,
        m,
        e_true,
    )

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
    print(
        f"Abs homog err is {homog_err_abs:5f}, rel is {homog_err_abs / mean_homog:5f}"
    )
    print(
        f"Pred range: min {strain_pred[:, 0].min():.5}, max {strain_pred[:, 0].max():.5}, mean {strain_pred[:, 0].mean():.5}, std {strain_pred[:, 0].std():.5}"
    )
    print(
        f"True range: min {strain_true[:, 0].min():.5}, max {strain_true[:, 0].max():.5}, mean {strain_true[:, 0].mean():.5}, std {strain_true[:, 0].std():.5}"
    )

    valid_time_per_micro = 1000 * running_time_cost / len(eval_loader.dataset)

    print(
        f"Inference cost {running_time_cost:.3f} s total, {valid_time_per_micro:.2f} ms per instance"
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

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_max)
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

        print(f"Validation pass took {diff:.3f}s")
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

            C_field = model.constlaw.compute_C_field(micros)

            stress_pred = model.constlaw(strain_pred, C_field)
            stress_true = model.constlaw(strain_true, C_field)

            # now compute losses
            losses_e, total_loss = compute_losses(
                model,
                strain_pred,
                strain_true,
                C_field,
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

            # print(f"batch {batch_ind}: {total_loss.detach().item():5f}")
            # printing once per epoch
            if batch_ind == 0:
                # print split on first batch to track progress
                print(f"Epoch {e}, batch {batch_ind}: {losses_e}")
                print(
                    f"Normalized e_xx absolute error is: {(strain_pred - strain_true)[:, 0].abs().mean() / model.constlaw.strain_scaling * 100:.5} %"
                )
                print(
                    f"Strain Pred range: min {strain_pred[:, 0].min():5f}, max {strain_pred[:, 0].max():5f}, mean {strain_pred[:, 0].mean():5f}, std {strain_pred[:, 0].std():5f}"
                )
                print(
                    f"Strain True range: min {strain_true[:, 0].min():5f}, max {strain_true[:, 0].max():5f}, mean {strain_true[:, 0].mean():5f}, std {strain_true[:, 0].std():5f}"
                )
                print(
                    f"Stress Pred range: min {stress_pred[:, 0].min():5f}, max {stress_pred[:, 0].max():5f}, mean {stress_pred[:, 0].mean():5f}, std {stress_pred[:, 0].std():5f}"
                )
                print(
                    f"Stress True range: min {stress_true[:, 0].min():5f}, max {stress_true[:, 0].max():5f}, mean {stress_true[:, 0].mean():5f}, std {stress_true[:, 0].std():5f}"
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
