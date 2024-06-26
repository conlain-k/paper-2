from numpy import s_, unravel_index
import torch

import time
import wandb

from config import Config, DELIM, LossSet
from helpers import *

from constlaw import *

FIELD_IND = 0
# PLOT_IND = 1744

PLOT_IND_BAD = 1744
PLOT_IND_GOOD = 0

CHECK_CONSTLAW = False


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


def compute_error_metrics(model, C_field, strain_true, strain_pred):
    # make sure all quantities are gradient-free and no backprop happens here
    with torch.inference_mode():
        strain_true = strain_true.detach()
        stress_true = model.constlaw(C_field, strain_true).detach()
        strain_pred = strain_pred.detach()
        stress_pred = model.constlaw(C_field, strain_pred).detach()

        # compute homogenized elastic stiffness values
        C11_true = est_homog(strain_true, stress_true, (0, 0))
        C11_pred = est_homog(strain_pred, stress_pred, (0, 0))

        # normalize instance-wise
        homog_err = (C11_true - C11_pred).abs() / C11_true.abs()

        def avg_L2_norm(x):
            # volume-averaged L2 norm of a vector quantity
            return (x**2).sum(dim=1).mean(dim=(-3, -2, -1)).sqrt()

        strain_L2_err = (
            100 * avg_L2_norm(strain_true - strain_pred) / model.constlaw.strain_scaling
        )
        stress_L2_err = (
            100 * avg_L2_norm(stress_true - stress_pred) / model.constlaw.stress_scaling
        )

        # Compute running L1 errors
        # average out over space
        L1_exx_err = (
            mean_L1_error(strain_pred[:, 0], strain_true[:, 0])
            / model.constlaw.strain_scaling
        )

        VM_stress_pred = VMStress(stress_pred)
        VM_stress_true = VMStress(stress_true)

        # normalize instance-wise as well
        L1_VM_err = mean_L1_error(VM_stress_pred, VM_stress_true) / mean_L1_error(
            VM_stress_true, 0 * VM_stress_true
        )

    return (
        strain_L2_err.reshape(-1),
        stress_L2_err.reshape(-1),
        L1_exx_err.reshape(-1),
        L1_VM_err.reshape(-1),
        homog_err.reshape(-1),
    )


# compute an energy-type loss for two strain-like fields
def compute_energy_loss(
    strain_error, C_field, add_deriv=False, ret_deriv=False, energy_scale=1
):

    # contract strain and stress dimensions
    strain_error_energy = (
        torch.einsum(
            "brxyz, brcxyz, bcxyz -> bxyz", strain_error, C_field, strain_error
        )
        / energy_scale
    )

    loss = strain_error_energy

    # if add_deriv:
    #     # take finite differences
    #     strain_error_grad = central_diff_3d(strain_error_energy, h=1)
    #     strain_error_grad = torch.stack(strain_error_grad, dim=-1)

    #     # contract over components at all locations
    #     strain_error_grad_energy = strain_error_grad**2
    #     # also sum out over gradient dimension (SPD means each entry should be positive??)
    #     loss = loss + strain_error_grad_energy.sum(dim=-1)

    if add_deriv:
        # take finite differences
        strain_error_grad = central_diff_3d(strain_error_energy, h=1)
        strain_error_grad = torch.stack(strain_error_grad, dim=-1)

        # contract over components at all locations
        strain_error_grad_energy = strain_error_grad**2
        # also sum out over gradient dimension (SPD means each entry should be positive??)
        loss = loss + strain_error_grad_energy.sum(dim=-1)

    if ret_deriv:
        return strain_error_energy, strain_error_grad_energy
    else:
        return loss


def deriv_loss(y_true, y_pred=None, reduce=True):
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

    # stack at last dim
    diff_true = torch.stack(diff_true, dim=-1)

    if y_pred is not None:
        diff_pred = central_diff_3d(y_pred, h=1)
        diff_pred = torch.stack(diff_pred, dim=-1)

        resid = diff_true - diff_pred
    else:
        resid = diff_true

    # mean squared Frob-norm of error in gradients
    diff_loss = (resid**2).sum(dim=(-1))
    if reduce:
        diff_loss = diff_loss.mean()

    return diff_loss


def plot_example(epoch, model, loader, ind, add_str=None):
    micro, strain_true, _ = loader.dataset[ind : ind + 1]

    micro = micro.to(model.inf_device)
    strain_true = strain_true.to(model.inf_device)

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

    # L1 VM stress error
    VM_L1_err = (
        mean_L1_error(VM_stress_pred, VM_stress_true) / VM_stress_true.abs().mean()
    )

    # also compute VM stresses at that location
    stressdiv_pred = stressdiv(stress_pred, use_FFT_deriv=False)
    stressdiv_true = stressdiv(stress_true, use_FFT_deriv=False)

    energy_err, energy_err_grad = compute_energy_loss(
        strain_true - strain_pred,
        C_field,
        add_deriv=True,
        ret_deriv=True,
        energy_scale=model.constlaw.energy_scaling,
    )
    energy_err = energy_err.squeeze()
    energy_err_grad = energy_err_grad.squeeze()

    print(
        f"\tStressdiv true mean {stressdiv_true.abs().mean()} std {stressdiv_true.abs().std()}"
    )
    print(
        f"\tStressdiv pred mean {stressdiv_pred.abs().mean()} std {stressdiv_pred.abs().std()}"
    )

    # C11_tr_homog(straue = est_homog(strain_true, stress_true, (0, 0)).squeeze()
    # C11_pred = estin_pred, stress_pred, (0, 0)).squeeze()

    energy = compute_strain_energy(strain_true, stress_true).mean()
    print(
        f"Saving fig for epoch {epoch}, plotting micro {ind} near {ind_max}, VM L1 err is {VM_L1_err.item():4f}, energy is {energy:4f}"
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

    # stress_loss = (
    #     model.constlaw.S0_norm((stress_true - stress_pred))
    #     / model.constlaw.energy_scaling
    # )

    # strain_loss = (
    #     model.constlaw.C0_norm((strain_true - strain_pred))
    #     / model.constlaw.energy_scaling
    # )

    # stress_loss_deriv = deriv_loss(stress_loss, reduce=False)

    # strain_loss_deriv = deriv_loss(strain_loss, reduce=False)

    plot_pred(
        epoch,
        micro[0, 0][sind],
        strain_true[0, FIELD_IND][sind],
        strain_pred[0, FIELD_IND][sind],
        "strain",
        model.config.image_dir + add_str,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        stress_true[0, FIELD_IND][sind],
        stress_pred[0, FIELD_IND][sind],
        "stress",
        model.config.image_dir + add_str,
    )
    plot_pred(
        epoch,
        micro[0, 0][sind],
        VM_stress_true[0, FIELD_IND][sind],
        VM_stress_pred[0, FIELD_IND][sind],
        "VM_stress",
        model.config.image_dir + add_str,
    )

    if model.config.use_deq:

        compat_err_true, equib_err_true = model.greens_op.compute_residuals(
            strain_true, stress_true
        )
        compat_err_pred, equib_err_pred = model.greens_op.compute_residuals(
            strain_pred, stress_pred
        )

        plot_pred(
            epoch,
            micro[0, 0][sind],
            compat_err_true[0, 0][sind],
            compat_err_pred[0, 0][sind],
            "compat_err",
            model.config.image_dir + add_str,
        )

        plot_pred(
            epoch,
            micro[0, 0][sind],
            equib_err_true[0, 0][sind],
            equib_err_pred[0, 0][sind],
            "equib_err",
            model.config.image_dir + add_str,
        )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        stressdiv_true[0, 0][sind],
        stressdiv_pred[0, 0][sind],
        "stressdiv",
        model.config.image_dir + add_str,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        energy_true[0, 0][sind] / model.constlaw.energy_scaling,
        energy_pred[0, 0][sind] / model.constlaw.energy_scaling,
        "energy",
        model.config.image_dir + add_str,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        0 * energy_err[sind] / model.constlaw.energy_scaling,
        energy_err[sind] / model.constlaw.energy_scaling,
        "energy_err",
        model.config.image_dir + add_str,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        strain_pred[0, FIELD_IND][sind] / model.constlaw.strain_scaling,
        (resid + strain_pred)[0, FIELD_IND][sind] / model.constlaw.strain_scaling,
        "resid",
        model.config.image_dir + add_str,
    )

    # ediff = compute_strain_energy(
    #     strain_true - strain_pred, stress_true - stress_pred
    # ).squeeze()

    # print(
    #     "epsbar", strain_true.mean((-3, -2, -1, 0)), strain_pred.mean((-3, -2, -1, 0))
    # )
    # print(
    #     "sigbar", stress_true.mean((-3, -2, -1, 0)), stress_pred.mean((-3, -2, -1, 0))
    # )

    # plot_pred(
    #     epoch,
    #     micro[0, 0][sind],
    #     0 * ediff[sind],  # / model.constlaw.energy_scaling,
    #     ediff[sind],  # / model.constlaw.energy_scaling,
    #     "ediff",
    #     model.config.image_dir+add_str,
    # )

    # err = (ediff - energy_err).abs()
    # print("errstats", err.min(), err.max(), err.mean(), err.std())


def compute_losses(model, strain_pred, strain_true, C_field, resid):

    # compute stresses
    stress_pred = model.constlaw(C_field, strain_pred)
    stress_true = model.constlaw(C_field, strain_true)

    # get stiffness scalings (to offset effect of higher-vf micros)
    # get dimension (assumes square)
    # how many voxels?
    S = torch.numel(C_field[0, 0, 0])
    # C_scaling = torch.einsum("brcxyz, brcxyz -> b", C_field, C_field).sqrt() / S
    # # broadcast across all components and locations
    # # print(C_scaling)
    # C_scaling = C_scaling.reshape(-1, 1, 1, 1, 1, 1)

    def mean_frob2_norm(field):
        # mean squared batched frobenius norm (over all interior dimensions)
        return torch.einsum("b...xyz, b...xyz -> b", field, field).mean(
            (-3, -2, -1)
        ) / (S**2)

    # strain_scaling = frob_norm_2(strain_true)
    # stress_scaling = frob_norm_2(stress_true)

    # scaled errors in strain and stress
    strain_loss = model.constlaw.C0_norm(strain_true - strain_pred).mean() / (
        model.constlaw.energy_scaling
    )
    stress_loss = model.constlaw.S0_norm(stress_true - stress_pred).mean() / (
        model.constlaw.energy_scaling
    )

    energy_loss = (
        compute_strain_energy(strain_true - strain_pred, stress_true - stress_pred)
        / model.constlaw.energy_scaling
    )

    energy_loss = energy_loss.mean()

    # print(f"strain {strain_loss:4f} stress {stress_loss:4f} energy {energy_loss:4f}")

    resid_loss = torch.as_tensor(0.0)
    compat_loss = torch.as_tensor(0.0)

    if model.config.return_resid:
        # compute energy in residual and add that term
        resid_loss = (
            model.constlaw.C0_norm(resid).mean() / model.constlaw.energy_scaling
        )

    if model.config.use_deq:

        err_compat, _ = model.greens_op.compute_residuals(
            strain_pred, model.constlaw(C_field, strain_pred)
        )
        # compute RMSE, averaged across channels/batch, converted to percent
        compat_loss = 100 * (err_compat**2).mean(dim=(-3, -2, -1)).sqrt().mean()

    if model.config.use_sqrt_loss:
        # take square roots before balancing
        strain_loss = strain_loss.sqrt()
        stress_loss = stress_loss.sqrt()
        energy_loss = energy_loss.sqrt()
        resid_loss = resid_loss.sqrt()
        compat_loss = compat_loss.sqrt()

    losses = LossSet(
        model.config,
        strain_loss,
        stress_loss,
        energy_loss,
        resid_loss,
        compat_loss,
    )

    # return losses.detach(), stress_loss
    return losses.detach(), losses.compute_total()


def eval_pass(model, epoch, eval_loader, data_mode, ema_model=None):
    # if we're given an ema model, use that for forward passes
    eval_model = ema_model or model

    eval_model.eval()

    # zero out losses to star
    running_loss = LossSet(config=model.config)

    # empty tensors initially
    strain_L2_errs = torch.Tensor().to(model.inf_device)
    stress_L2_errs = torch.Tensor().to(model.inf_device)
    L1_exx_errs = torch.Tensor().to(model.inf_device)
    L1_VM_errs = torch.Tensor().to(model.inf_device)
    homog_errs = torch.Tensor().to(model.inf_device)

    # plot worst micro in validation set
    plot_example(epoch, model, eval_loader, PLOT_IND_BAD, "/hard/")

    plot_example(epoch, model, eval_loader, PLOT_IND_GOOD, "/easy/")
    running_time_cost = 0
    for batch_ind, (micros, strain_true, stress_true) in enumerate(eval_loader):

        # force synchronization for timing
        sync()
        if data_mode == DataMode.TEST:
            print(f"Testing batch {batch_ind} of {len(eval_loader)}")

        t0 = time.time()

        # include data migration in timing
        micros = micros.to(model.inf_device)
        strain_true = strain_true.to(model.inf_device)
        stress_true = stress_true.to(model.inf_device)

        if batch_ind == 0 and CHECK_CONSTLAW:
            check_constlaw(model.constlaw, micros, strain_true, stress_true)

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

        assert not torch.isnan(strain_pred).any()

        C_field = model.constlaw.compute_C_field(micros)

        losses_e, _ = compute_losses(
            model,
            strain_pred,
            strain_true,
            C_field,
            resid,
        )

        # accumulate loss
        running_loss = running_loss + losses_e

        # get instance-wise error for this batch
        (
            strain_L2_err_batch,
            stress_L2_err_batch,
            L1_exx_err_batch,
            L1_VM_err_batch,
            homog_err_batch,
        ) = compute_error_metrics(model, C_field, strain_true, strain_pred)

        # print("strain_L2_err_batch", strain_L2_err_batch.shape)
        # print("stress_L2_err_batch", stress_L2_err_batch.shape)
        # print("L1_exx_err_batch", L1_exx_err_batch.shape)
        # print("L1_VM_err_batch", L1_VM_err_batch.shape)
        # print("homog_err_batch", homog_err_batch.shape)

        # build up list of errors
        strain_L2_errs = torch.cat([strain_L2_errs, strain_L2_err_batch])
        stress_L2_errs = torch.cat([stress_L2_errs, stress_L2_err_batch])
        L1_exx_errs = torch.cat([L1_exx_errs, L1_exx_err_batch])
        L1_VM_errs = torch.cat([L1_VM_errs, L1_VM_err_batch])
        homog_errs = torch.cat([homog_errs, homog_err_batch])

    # divide out number of batches (simple normalization)
    running_loss /= len(eval_loader)

    # divide out # samples
    L1_exx_err_mean = L1_exx_errs.mean()
    L1_exx_err_std = L1_exx_errs.std()

    L1_VM_err_mean = L1_VM_errs.mean()
    L1_VM_err_std = L1_VM_errs.std()

    strain_L2_err_mean = strain_L2_errs.mean()
    strain_L2_err_std = strain_L2_errs.std()

    stress_L2_err_mean = stress_L2_errs.mean()
    stress_L2_err_std = stress_L2_errs.std()

    homog_err_mean = homog_errs.mean()
    homog_err_std = homog_errs.std()

    if data_mode == DataMode.VALID:
        wandb.log(
            {
                "epoch": epoch,
                f"total_valid_loss": running_loss.compute_total(),
                f"valid_losses": running_loss.to_dict(),
                # these three are for legacy
                "homog_err_rel": homog_err_mean,
                f"valid_exx_err": L1_exx_err_mean,
                f"valid_VM_err": L1_VM_err_mean,
                # actual metrics for publication
                "L1_exx_err_mean": L1_exx_err_mean,
                "L1_exx_err_std": L1_exx_err_std,
                "L1_VM_err_mean": L1_VM_err_mean,
                "L1_VM_err_std": L1_VM_err_std,
                "strain_L2_err_mean": strain_L2_err_mean,
                "strain_L2_err_std": strain_L2_err_std,
                "stress_L2_err_mean": stress_L2_err_mean,
                "stress_L2_err_std": stress_L2_err_std,
                "homog_err_mean": homog_err_mean,
                "homog_err_std": homog_err_std,
            }
        )

    # print some metrics on the epoch
    print(f"Epoch {epoch}, {data_mode} loss: {running_loss}")
    print(f"Normalized e_xx absolute error is: {L1_exx_err_mean * 100:.5} %")
    print(f"Normalized VM stress absolute error is: {L1_VM_err_mean:.5}")
    print(f"Abs homog err is mean {homog_err_mean:4f} std {homog_err_std}")
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
    if config.use_EMA:
        ema_model = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.99)
        )
    else:
        ema_model = None  # makes eval code simpler

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_max)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.num_epochs, eta_min=1e-8
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
            micros = micros.to(model.inf_device)
            # only predict first component
            strain_true = strain_true.to(model.inf_device)

            if batch_ind == 0 and CHECK_CONSTLAW:
                batch_stress_true = batch_stress_true.to(model.inf_device)

                check_constlaw(model.constlaw, micros, strain_true, batch_stress_true)

            # zero out gradients
            optimizer.zero_grad()

            # apply model forward pass
            output = model(micros)

            # extract output
            if config.use_deq and config.return_resid:
                (strain_pred, resid) = output
            else:
                strain_pred = output
                resid = 0 * strain_pred

            C_field = model.constlaw.compute_C_field(micros)

            stress_pred = model.constlaw(C_field, strain_pred)
            stress_true = model.constlaw(C_field, strain_true)

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
            running_loss = running_loss + total_loss.detach() / len(train_loader)

            # print(f"batch {batch_ind}: {total_loss.detach().item():4f}")
            # printing once per epoch
            if batch_ind == 0:
                # print split on first batch to track progress
                print(f"Epoch {e}, batch {batch_ind}: {losses_e}")
                print(
                    f"Strain Pred range: min {strain_pred[:, 0].min():4f}, max {strain_pred[:, 0].max():4f}, mean {strain_pred[:, 0].mean():4f}, std {strain_pred[:, 0].std():4f}"
                )
                print(
                    f"Strain True range: min {strain_true[:, 0].min():4f}, max {strain_true[:, 0].max():4f}, mean {strain_true[:, 0].mean():4f}, std {strain_true[:, 0].std():4f}"
                )
                print(
                    f"Stress Pred range: min {stress_pred[:, 0].min():4f}, max {stress_pred[:, 0].max():4f}, mean {stress_pred[:, 0].mean():4f}, std {stress_pred[:, 0].std():4f}"
                )
                print(
                    f"Stress True range: min {stress_true[:, 0].min():4f}, max {stress_true[:, 0].max():4f}, mean {stress_true[:, 0].mean():4f}, std {stress_true[:, 0].std():4f}"
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
        # if  model.config.use_deq:
        #     torch.cuda.empty_cache()

        print(f"Training pass took {diff}s")

        print(f"Epoch {e}: instance-average loss was {running_loss:4f}")

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
