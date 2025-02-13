from numpy import s_, unravel_index
import torch

import time
import wandb

from config import DELIM
from helpers import *

from constlaw import *

FIELD_IND = 0

# index of worst strain microstructure
PLOT_IND_WORST_STRAIN = 1872
# index of worst stress microstructure
PLOT_IND_WORST_STRESS = 0
# generic easy micro
PLOT_IND_GENERIC = 155

CHECK_CONSTLAW = True

EPS = 1e-6


def compute_error_metrics(model, C_field, strain_true, strain_pred):
    # make sure all quantities are gradient-free and no backprop happens here
    with torch.inference_mode():
        strain_true = strain_true.detach()
        strain_pred = strain_pred.detach()
        # also compute stresses
        stress_true = strain_to_stress(C_field, strain_true).detach()
        stress_pred = strain_to_stress(C_field, strain_pred).detach()

        # compute homogenized elastic stiffness values
        C11_true = est_homog(strain_true, stress_true, (0, 0))
        C11_pred = est_homog(strain_pred, stress_pred, (0, 0))

        # normalize instance-wise
        homog_err = (C11_true - C11_pred).abs() / C11_true.abs()

        def avg_L2_norm(x):
            # volume-averaged L2 norm of a vector quantity
            return (x**2).sum(dim=1).mean(dim=(-3, -2, -1)).sqrt()

        strain_L2_err = 100 * avg_L2_norm(model.scale_strain(strain_true - strain_pred))
        stress_L2_err = 100 * avg_L2_norm(model.scale_stress(stress_true - stress_pred))

        # Compute running L1 errors
        # average out over space
        L1_exx_err = model.scale_strain(
            mean_L1_error(strain_pred[:, 0], strain_true[:, 0])
        )

        VM_stress_pred = VMStress(stress_pred)
        VM_stress_true = VMStress(stress_true)

        # normalize instance-wise as well
        L1_VM_err = mean_L1_error(VM_stress_pred, VM_stress_true) / mean_L1_error(
            VM_stress_true, 0 * VM_stress_true
        )

    return (
        strain_L2_err.reshape(-1).detach(),
        stress_L2_err.reshape(-1).detach(),
        L1_exx_err.reshape(-1).detach(),
        L1_VM_err.reshape(-1).detach(),
        homog_err.reshape(-1).detach(),
    )


def plot_example(epoch, model, loader, ind, add_str=None):
    C_field, bc_vals, strain_true, _ = loader.dataset[ind : ind + 1]

    C_field = C_field.to(model.inf_device)
    strain_true = strain_true.to(model.inf_device)
    bc_vals = bc_vals.to(model.inf_device)

    # evaluate model
    strain_pred = model(C_field, bc_vals)

    last_iterate = None
    if model.config.add_resid_loss:
        # also grab resid if it's an option
        strain_pred, last_iterate = strain_pred

    # recompute quantities
    stress_pred, stress_polar_pred, energy_pred = compute_quants(
        model, strain_pred, C_field
    )
    stress_true, stress_polar_true, energy_true = compute_quants(
        model, strain_true, C_field
    )

    # mean_L1_strain_errs = (
    #     model.scale_strain(strain_pred - strain_true)
    #     .abs()
    #     .mean(dim=(-3, -2, -1))
    # )
    # mean_L1_stress_errs = (
    #     model.scale_stress(stress_pred - stress_true)
    #     .abs()
    #     .mean(dim=(-3, -2, -1))
    # )

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

    energy_err = compute_strain_energy(
        strain_true - strain_pred, stress_true - stress_pred
    )

    energy_err = energy_err.squeeze()

    energy = compute_strain_energy(strain_true, stress_true).mean()

    # get equivalent strains for comparison
    equiv_strain_true = equivalent(strain_true)
    equiv_strain_pred = equivalent(strain_pred)

    # get worst L1 error in equivalent strains
    ind_max = (
        torch.argmax((equiv_strain_true - equiv_strain_pred).abs().max()).detach().cpu()
    )

    ind_max = unravel_index(ind_max, equiv_strain_true[:].detach().cpu().shape)

    print(
        f"Saving fig for epoch {epoch}, plotting micro {ind} near {ind_max}, VM L1 err is {VM_L1_err.item():4f}, energy is {energy:4f}"
    )

    # Plot z=const slice
    sind = s_[:, :, ind_max[-1]]

    micro_slice = C_field[0, 0, 0][sind]

    plot_pred(
        epoch,
        micro_slice,
        equiv_strain_true[0][sind],
        equiv_strain_pred[0][sind],
        "equiv_strain",
        model.config.image_dir + add_str,
    )

    plot_pred(
        epoch,
        micro_slice,
        VM_stress_true[0][sind],
        VM_stress_pred[0][sind],
        "VM_stress",
        model.config.image_dir + add_str,
    )

    if True:

        compat_err_true, equib_err_true = model.greens_op.compute_residuals(
            strain_true, stress_true
        )
        compat_err_pred, equib_err_pred = model.greens_op.compute_residuals(
            strain_pred, stress_pred
        )

        print(
            f"\tTrue equib {equib_err_true.mean():4f}, {equib_err_true.std():4f}, {equib_err_true.min():4f}, {equib_err_true.max():4f}"
        )
        print(
            f"\tPred equib {equib_err_pred.mean():4f}, {equib_err_pred.std():4f}, {equib_err_pred.min():4f}, {equib_err_pred.max():4f}"
        )

        plot_pred(
            epoch,
            micro_slice,
            compat_err_true[0, 0][sind],
            compat_err_pred[0, 0][sind],
            "compat_err",
            model.config.image_dir + add_str,
        )

        plot_pred(
            epoch,
            micro_slice,
            equib_err_true[0, 0][sind],
            equib_err_pred[0, 0][sind],
            "equib_err",
            model.config.image_dir + add_str,
        )

    plot_pred(
        epoch,
        micro_slice,
        stressdiv_true[0, 0][sind],
        stressdiv_pred[0, 0][sind],
        "stressdiv",
        model.config.image_dir + add_str,
    )

    plot_pred(
        epoch,
        micro_slice,
        energy_true[0, 0][sind],
        energy_pred[0, 0][sind],
        "energy",
        model.config.image_dir + add_str,
    )

    plot_pred(
        epoch,
        micro_slice,
        0 * energy_err[sind],
        energy_err[sind],
        "energy_err",
        model.config.image_dir + add_str,
    )

    if model.config.add_resid_loss:
        input_encoded = model._encodeInput(C_field, bc_vals)

        h_in_last = last_iterate
        h_out_last = model.F(h_in_last, input_encoded)
        plot_pred(
            epoch,
            micro_slice,
            h_in_last[0, FIELD_IND][sind],
            h_out_last[0, FIELD_IND][sind],
            "last_iterate",
            model.config.image_dir + add_str,
        )

        if model.config.penalize_teacher_resid:
            h_in_teacher = model.scale_strain(strain_true)

            h_out_teacher = model.F(h_in_teacher, input_encoded)

            plot_pred(
                epoch,
                micro_slice,
                h_in_teacher[0, FIELD_IND][sind],
                h_out_teacher[0, FIELD_IND][sind],
                "teacher_resid",
                model.config.image_dir + add_str,
            )

            plot_pred(
                epoch,
                micro_slice,
                h_out_last[0, FIELD_IND][sind],
                h_out_teacher[0, FIELD_IND][sind],
                "image_teacher_resid",
                model.config.image_dir + add_str,
            )


def compute_losses(
    model, strain_pred, strain_true, C_field, bc_vals, last_iterate=None
):

    # compute stresses
    # stress_pred = strain_to_stress(C_field, strain_pred)
    # stress_true = strain_to_stress(C_field, strain_true)

    def frob_norm_2(field):
        """squared batched frobenius norm (sum over first dimension)"""
        return (field**2).sum(1).mean((-3, -2, -1))

    resid_loss = torch.as_tensor(0.0)

    if model.config.add_resid_loss:
        input_encoded = model._encodeInput(C_field, bc_vals)
        # do we use predicted or true strain to penalize resid?
        # NOTE: only use this when we iterate directly over strain fields
        h_in_last = last_iterate

        h_out_last = model.F(h_in_last.detach(), input_encoded)

        # compute energy in residual and add that term
        resid_loss = frob_norm_2(h_out_last - h_in_last)

        if model.config.penalize_teacher_resid:
            h_in_teacher = model.scale_strain(strain_true)
            h_out_teacher = model.F(h_in_teacher.detach(), input_encoded)
            resid_loss = resid_loss + frob_norm_2(h_in_teacher - h_out_teacher)

        if model.config.penalize_resid_misalignment:
            # resid_deq = h_out_last
            # error_true = model.scale_strain(strain_true) - h_in_last

            align_diff = h_out_last - model.scale_strain(strain_true)

            # inner product is between -1 and 1 (1 best, -1 worst)
            # 1 means perfectly aligned, -1 means perfectly misaligned
            align_loss = frob_norm_2(align_diff, align_diff)
            print("align", align_loss)
            resid_loss = resid_loss + align_loss

    # print(f"{strain_true.mean((0,-3,-2,-1))}")
    # print(f"{strain_pred.mean((0,-3,-2,-1))}")

    # print(f"{stress_true.mean((0,-3,-2,-1))}")
    # print(f"{stress_pred.mean((0,-3,-2,-1))}")

    # print(bc_vals[:10])

    strain_err = model.scale_strain(strain_true - strain_pred)
    stress_err = strain_to_stress(model.scale_stiffness(C_field), strain_err)
    strain_loss = frob_norm_2(strain_err)
    stress_loss = frob_norm_2(stress_err)

    energy_loss = compute_strain_energy(strain_err, stress_err)

    # volume-average each quantity
    energy_loss = energy_loss.mean((-3, -2, -1))

    # now average across instances
    strain_loss = strain_loss.mean()
    stress_loss = stress_loss.mean()
    energy_loss = energy_loss.mean()
    resid_loss = resid_loss.mean()

    # print(
    #     f"strain: {strain_loss:.4f}, stress: {stress_loss:.4f}, energy: {energy_loss:.4f},resid: {resid_loss:.4f}"
    # )

    # if model.config.balance_losses:
    #     # balance so that each term contributes equally (change direction but not total magnitude of gradients)
    #     sum = (strain_loss + stress_loss + energy_loss + 3 * EPS).detach()
    #     lam_strain = sum / (strain_loss.detach() + EPS)
    #     lam_stress = sum / (stress_loss.detach() + EPS)
    #     lam_energy = sum / (energy_loss.detach() + EPS)
    # else:
    lam_strain = model.config.lam_strain
    lam_stress = model.config.lam_stress
    lam_energy = model.config.lam_energy

    # collect all losses into one term
    total_loss = (
        lam_strain * strain_loss + lam_stress * stress_loss + lam_energy * energy_loss
    )
    if model.config.add_resid_loss:
        total_loss = total_loss + model.config.lam_resid * resid_loss

    # return losses.detach(), stress_loss
    return total_loss


def eval_pass(model, epoch, eval_loader, data_mode, ema_model=None):
    # if we're given an ema model, use that for forward passes
    eval_model = ema_model or model

    eval_model.eval()

    # zero out losses to star
    running_loss = 0.0

    # empty tensors initially
    strain_L2_errs = torch.Tensor().to(model.inf_device)
    stress_L2_errs = torch.Tensor().to(model.inf_device)
    L1_exx_errs = torch.Tensor().to(model.inf_device)
    L1_VM_errs = torch.Tensor().to(model.inf_device)
    homog_errs = torch.Tensor().to(model.inf_device)

    # plot worst micros in validation set
    plot_example(epoch, model, eval_loader, PLOT_IND_WORST_STRAIN, "/hard/")
    plot_example(epoch, model, eval_loader, PLOT_IND_GENERIC, "/easy/")
    running_time_cost = 0
    for batch_ind, (C_field, bc_vals, strain_true, stress_true) in enumerate(
        eval_loader
    ):
        if data_mode == DataMode.TEST:
            print(f"Testing batch {batch_ind} of {len(eval_loader)}")

        C_field = C_field.to(model.inf_device)

        # check that constlaw is accurate (helps pick off errors in conversion, etc.)
        if (batch_ind == 0) and (epoch == 0) and CHECK_CONSTLAW:
            check_constlaw(
                model.to(model.inf_device).constlaw,
                C_field,
                strain_true.to(model.inf_device),
                stress_true.to(model.inf_device),
            )

        # force synchronization for timing
        sync()
        t0 = time.time()

        # include data migration in timing
        # micros = micros.to(model.inf_device)
        bc_vals = bc_vals.to(model.inf_device)

        # now evaluate model
        strain_pred = eval_model(C_field, bc_vals)
        # print("eval mem\n", torch.cuda.memory_summary())

        # how long did that pass take?
        sync()
        t1 = time.time()

        if data_mode == DataMode.TEST:
            print(f"This batch took {t1 - t0} seconds")

        # build up how long it's taken to run all samples
        running_time_cost += t1 - t0

        last_iterate = None
        # split off last iterate if needed
        if model.config.add_resid_loss:
            (strain_pred, last_iterate) = strain_pred

        assert not torch.isnan(strain_pred).any()

        strain_true = strain_true.to(model.inf_device)
        stress_true = stress_true.to(model.inf_device)
        # with torch.inference_mode():
        total_loss = compute_losses(
            model, strain_pred, strain_true, C_field, bc_vals, last_iterate
        )

        # accumulate loss
        running_loss = running_loss + total_loss.detach().item()

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

    worst_strain_L2_ind = torch.argmax(strain_L2_errs)
    worst_stress_L2_ind = torch.argmax(stress_L2_errs)

    print(f"worst strain: {worst_strain_L2_ind}, worst stress: {worst_stress_L2_ind}")

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
                f"total_valid_loss": total_loss,
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
    print(f"Epoch {epoch}, valid loss: {running_loss}")
    print(f"Normalized e_xx absolute error is: {L1_exx_err_mean * 100:.5} %")
    print(f"Normalized VM stress absolute error is: {L1_VM_err_mean * 100:.5} %")
    print(
        f"Error metrics: strain {strain_L2_err_mean:.4f} +- {strain_L2_err_std:.4f}, stress {stress_L2_err_mean:.4f} +- {stress_L2_err_std:.4f}"
    )
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

    return valid_time_per_micro


def train_model(model, config, train_loader, valid_loader):
    if config.use_EMA:
        ema_model = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
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
    #     pct_start=0.1,  # first 10% is increase, then anneal
    # )

    print(
        "Scalings",
        model.strain_scaling,
        model.stiffness_scaling,
        model.stress_scaling,
        model.energy_scaling,
    )

    for e in range(config.num_epochs):

        print(DELIM)

        # Run a validation pass before training this epoch
        # time validation pass
        start = time.time()

        # no backprop during eval mode
        with torch.inference_mode():
            model.eval()
            valid_time_per_micro = eval_pass(
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

        # zero out gradients
        optimizer.zero_grad()

        for batch_ind, (C_field, bc_vals, strain_true, batch_stress_true) in enumerate(
            train_loader
        ):
            C_field = C_field.to(model.inf_device)

            # C_field = model.constlaw.compute_C_field(micros)

            if (batch_ind == 0) and CHECK_CONSTLAW and (e == 0):
                check_constlaw(
                    model.to(model.inf_device).constlaw,
                    C_field,
                    strain_true.to(model.inf_device),
                    batch_stress_true.to(model.inf_device),
                )

            bc_vals = bc_vals.to(model.inf_device)

            # apply model forward pass
            strain_pred = model(C_field, bc_vals)
            # print("train mem\n", torch.cuda.memory_summary())

            # extract output
            last_iterate = None
            if config.add_resid_loss:
                (strain_pred, last_iterate) = strain_pred

            # only predict first component
            strain_true = strain_true.to(model.inf_device)

            # now compute losses
            total_loss = compute_losses(
                model,
                strain_pred,
                strain_true,
                C_field,
                bc_vals,
                last_iterate,
            )

            # make sure gradients are zeroed before we compute new ones
            optimizer.zero_grad()
            # backprop now
            total_loss.backward()
            # now do grad clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_mag)
            optimizer.step()

            # update averaged model after the first epoch (so that we discard the initial noisy model)
            if config.use_EMA and e > 0:
                # set ema model to training
                ema_model.train()
                ema_model.update_parameters(model)

            # now accumulate losses for future
            running_loss = running_loss + total_loss.detach().item() / len(train_loader)

            # print(f"batch {batch_ind}: {total_loss.detach().item():4f}")
            # printing once per epoch
            if batch_ind == 0:
                strain_pred = strain_pred.detach()
                strain_true = strain_true.detach()
                with torch.inference_mode():
                    stress_pred = strain_to_stress(C_field, strain_pred)
                    stress_true = strain_to_stress(C_field, strain_true)
                # print split on first batch to track progress
                print(f"Epoch {e}, batch {batch_ind}; loss is {total_loss}")
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

        print(f"Training pass took {diff}s")

        print(f"Epoch {e}: instance-average loss was {running_loss:4f}")

        if running_loss < best_loss:
            best_loss = running_loss
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
