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


# compute an energy-type loss for two strain-like fields
def compute_energy_loss(strain_error, C_field, add_deriv=False, ret_deriv=False):
    # Nx = strain_error.shape[-1]
    # # domain is zero to 1
    # L = 1
    # # grid spacing
    # h = L / Nx

    stress_error = torch.einsum("brcxyz, bcxyz -> brxyz", C_field, strain_error)

    err_alt = torch.einsum("brxyz, brxyz -> bxyz", strain_error, stress_error)

    # contract strain and stress dimensions
    strain_error_energy = torch.einsum(
        "brxyz, brcxyz, bcxyz -> bxyz", strain_error, C_field, strain_error
    )

    assert torch.allclose(err_alt, strain_error_energy)

    loss = strain_error_energy

    # print(f"strain_error loss is {strain_error_energy.mean():4f}")

    if add_deriv:
        # take finite differences
        strain_error_grad = central_diff_3d(strain_error, h=1)
        strain_error_grad = torch.stack(strain_error_grad, dim=-1)

        # also sum over last dimension (spatial deriv index) to get squared 2-norm of vector field
        strain_error_grad_energy = (
            torch.einsum(
                "brxyzd, brcxyz, bcxyzd -> bxyz",
                strain_error_grad,
                C_field,
                strain_error_grad,
            )
            * 0.0
        )

        # print(f"Grad loss is {strain_error_grad_energy.mean():4f}")
        loss += strain_error_grad_energy

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

    # L1 VM stress error
    LVE = mean_L1_error(VM_stress_pred, VM_stress_true) / VM_stress_true.abs().mean()

    # also compute VM stresses at that location
    stressdiv_pred = stressdiv(stress_pred, use_FFT_deriv=False)
    stressdiv_true = stressdiv(stress_true, use_FFT_deriv=False)

    energy_err, energy_err_grad = compute_energy_loss(
        strain_true - strain_pred,
        C_field,
        add_deriv=True,
        ret_deriv=True,
        # energy_scale=model.constlaw.energy_scaling,
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
        f"Saving fig for epoch {epoch}, plotting micro {ind} near {ind_max}, VM L1 err is {LVE.item():4f}, energy is {energy:4f}"
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

    stress_loss = (
        model.constlaw.S0_norm((stress_true - stress_pred))
        / model.constlaw.energy_scaling
    )

    strain_loss = (
        model.constlaw.C0_norm((strain_true - strain_pred))
        / model.constlaw.energy_scaling
    )

    stress_loss_deriv = deriv_loss(stress_loss, reduce=False).squeeze()

    strain_loss_deriv = deriv_loss(strain_loss, reduce=False).squeeze()

    plot_pred(
        epoch,
        micro[0, 0][sind],
        strain_true[0, FIELD_IND][sind] / model.constlaw.strain_scaling,
        strain_pred[0, FIELD_IND][sind] / model.constlaw.strain_scaling,
        "strain",
        model.config.image_dir + add_str,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        0 * strain_loss[0][sind],  # / model.constlaw.strain_scaling,
        strain_loss[0][sind],  # / model.constlaw.strain_scaling,
        "strain_loss",
        model.config.image_dir + add_str,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        0 * strain_loss_deriv[sind],  # / model.constlaw.strain_scaling,
        strain_loss_deriv[sind],  # / model.constlaw.strain_scaling,
        "strain_loss_deriv",
        model.config.image_dir + add_str,
    )
    plot_pred(
        epoch,
        micro[0, 0][sind],
        0 * stress_loss_deriv[sind],  # / model.constlaw.strain_scaling,
        stress_loss_deriv[sind],  # / model.constlaw.strain_scaling,
        "stress_loss_deriv",
        model.config.image_dir + add_str,
    )
    plot_pred(
        epoch,
        micro[0, 0][sind],
        stress_true[0, FIELD_IND][sind] / model.constlaw.stress_scaling,
        stress_pred[0, FIELD_IND][sind] / model.constlaw.stress_scaling,
        "stress",
        model.config.image_dir + add_str,
    )
    plot_pred(
        epoch,
        micro[0, 0][sind],
        0 * stress_loss[0][sind],
        stress_loss[0][sind],
        "stress_loss",
        model.config.image_dir + add_str,
    )
    plot_pred(
        epoch,
        micro[0, 0][sind],
        VM_stress_true[0, FIELD_IND][sind] / model.constlaw.stress_scaling,
        VM_stress_pred[0, FIELD_IND][sind] / model.constlaw.stress_scaling,
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
        model.config.image_dir,
    )

    plot_pred(
        epoch,
        micro[0, 0][sind],
        strain_pred[0, FIELD_IND][sind] / model.constlaw.strain_scaling,
        (resid + strain_pred)[0, FIELD_IND][sind] / model.constlaw.strain_scaling,
        "resid",
        model.config.image_dir + add_str,
    )


def compute_losses(model, strain_pred, strain_true, C_field, resid):

    # compute stresses
    stress_pred = model.constlaw(C_field, strain_pred)
    stress_true = model.constlaw(C_field, strain_true)

    # compute C0-norm of stress error
    stress_err_norm = (
        model.constlaw.S0_norm((stress_true - stress_pred))
        / model.constlaw.energy_scaling
    )
    stress_loss_L2 = stress_err_norm.mean().sqrt()

    stress_deriv_loss = deriv_loss(stress_err_norm)
    stress_loss = stress_loss_L2  # + stress_deriv_loss

    strain_resid = (
        model.constlaw.C0_norm((strain_true - strain_pred))
        / model.constlaw.energy_scaling
    )

    strain_loss_L2 = strain_resid.mean().sqrt()
    strain_deriv_loss = deriv_loss(strain_resid) / 20.0

    strain_loss = strain_loss_L2  # + strain_deriv_loss

    energy_loss = (
        compute_energy_loss(strain_true - strain_pred, C_field, add_deriv=True).mean()
        / model.constlaw.energy_scaling
    )

    # energy_diff = (
    #     compute_strain_energy(
    #         strain_true - strain_pred, stress_true - stress_pred
    #     ).mean()
    #     / model.constlaw.energy_scaling
    # )

    # print(
    #     f"L2: strain {strain_loss_L2:4f} stress {stress_loss_L2:4f} energy {energy_loss:4f}"
    # )
    # print(f"H1: strain {strain_deriv_loss:4f} stress {stress_deriv_loss:4f}")

    resid_loss = torch.as_tensor(0.0)
    compat_loss = torch.as_tensor(0.0)

    # print("RMS strain", (strain_true**2).sum(dim=1).mean().sqrt())
    # print("mean strain", strain_true.mean(dim=(0, -1, -2, -3)))
    # print("std strain", strain_true.std(dim=(0, -1, -2, -3)))

    if model.config.return_resid:
        # compute energy in residual and add that term
        resid_loss = (
            model.constlaw.C0_norm(resid).mean() / model.constlaw.energy_scaling
        )

        # compute energy in residual
        # resid_loss = (
        #     compute_strain_energy(resid, model.constlaw(resid, C_field)).mean()
        #     / model.constlaw.energy_scaling
        # )

    if model.config.use_deq:

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

    # return losses.detach(), stress_loss
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
        micros = micros.to(model.config.device)
        strain_true = strain_true.to(model.config.device)
        stress_true = stress_true.to(model.config.device)

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

        C11_true = est_homog(strain_true, stress_true, (0, 0))

        stress_pred = model.constlaw(C_field, strain_pred)
        C11_pred = est_homog(strain_pred, stress_pred, (0, 0))

        # worst_homog_ind = torch.argmax((C11_true - C11_pred).abs())

        # print(
        #     f"Batch {ind} worst is {worst_homog_ind} err {(C11_true - C11_pred).abs()[worst_homog_ind]:4f}"
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

        LVE = mean_L1_error(VM_stress_pred, VM_stress_true) / VM_stress_true.abs().mean(
            dim=(-3, -2, -1)
        )

        # print(LVE.shape, LSE.shape)
        # print(mean_L1_error(VM_stress_pred, VM_stress_true))
        # print(VM_stress_true.abs().mean())
        # print(VM_stress_true.abs().mean(dim=(-3, -2, -1)))

        # print(VM_stress_pred.shape)
        # print(LVE.shape, len(eval_loader.dataset))

        # print(LVE)

        # now average out over each instance in train set
        L1_strain_err = L1_strain_err + LSE.sum()
        L1_VM_stress_err = L1_VM_stress_err + LVE.sum()

    # divide out # samples
    L1_strain_err /= len(eval_loader.dataset)
    L1_VM_stress_err /= len(eval_loader.dataset)

    # divide out number of batches (simple normalization)
    running_loss /= len(eval_loader)

    homog_err_abs = homog_err / len(eval_loader.dataset)
    mean_homog = mean_homog / len(eval_loader.dataset)

    if data_mode == DataMode.VALID:
        wandb.log(
            {
                "epoch": epoch,
                f"total_valid_loss": running_loss.compute_total(),
                f"valid_losses": running_loss.to_dict(),
                "homog_err_rel": homog_err_abs / mean_homog,
                f"valid_exx_err": L1_strain_err,
                f"valid_VM_err": L1_VM_stress_err,
            }
        )

    # print some metrics on the epoch
    print(f"Epoch {epoch}, {data_mode} loss: {running_loss}")
    print(f"Normalized e_xx absolute error is: {L1_strain_err * 100:.5} %")
    print(f"Normalized VM stress absolute error is: {L1_VM_stress_err:.5}")
    print(
        f"Abs homog err is {homog_err_abs:4f}, rel is {homog_err_abs / mean_homog:4f}"
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
        # # only pretrain for given # epochs
        # if e >= config.num_pretrain_epochs and model.pretraining:
        #     print(f"\nDisabling pretrain mode at epoch {e}\n")
        #     model.pretraining = False
        #     # also rebuild optimizer to reset internal states / momentum
        #     optimizer = torch.optim.Adam(
        #         model.parameters(),
        #         lr=optimizer.param_groups[0]["lr"],
        #     )
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #         optimizer, config.num_epochs, eta_min=1e-8
        #     )

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

            if batch_ind == 0 and CHECK_CONSTLAW:
                batch_stress_true = batch_stress_true.to(config.device)

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
            running_loss += total_loss.detach() / len(train_loader)

            # print(f"batch {batch_ind}: {total_loss.detach().item():4f}")
            # printing once per epoch
            if batch_ind == 0:
                # print split on first batch to track progress
                print(f"Epoch {e}, batch {batch_ind}: {losses_e}")
                print(
                    f"Normalized e_xx absolute error is: {(strain_pred - strain_true)[:, 0].abs().mean() / model.constlaw.strain_scaling * 100:.5} %"
                )
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
