from helpers import *
from constlaw import *
from tensor_ops import *
from plot_cube import *
from euler_ang import *
from greens_op import *
from loaders import *
from config import Config
from solvers import make_localizer
from h5py import File

import numpy as np
import torch
import time

# for storing metrics
import pandas as pd

torch.backends.cudnn.enabled = False


E_VALS_DEF = [120.0, 100.0 * 120.0]
NU_VALS = [0.3, 0.3]
E_BAR = [0.001, 0, 0, 0, 0, 0]

ADD_STD = False

# MODE = DataMode.VALID
MODE = DataMode.TEST

PLOT_IND = 1785
SLICE_IND = 22


first = True


from main import load_data, dataset_info

import matplotlib.ticker as tck


EVAL_COLUMNS = [
    "$C^*$",
    "L1 VM Stress",
    "L2 Strain",
    "L2 Stress",
]


def compute_running_metrics(
    metrics, inds, strain_pred, strain_true, stress_true, C_field
):
    # print("writing", inds)

    def L2_err(resid):
        # resid : resid, averaged over space then square-rooted
        return (resid**2).sum(-4).mean((-3, -2, -1)).sqrt()

    def L1_err(resid, spatial_avg=False, norm_fac=1.0):
        err = (resid).abs()
        if spatial_avg:
            err = err.mean((-3, -2, -1))
        err = err / norm_fac
        return err

    # get predicted stress
    stress_pred = strain_to_stress(C_field, strain_pred)

    print("true", strain_true.min(), strain_true.max())
    print("pred", strain_pred.min(), strain_pred.max())
    diff = (strain_true - strain_pred).abs()
    print("diff", diff.min(), diff.max())

    metrics.loc[inds, "L2 Strain"] = L2_err(strain_true - strain_pred).cpu().numpy()
    metrics.loc[inds, "L2 Stress"] = L2_err(stress_true - stress_pred).cpu().numpy()

    # C1111 for true and field
    C_homog_true = est_homog(strain_true, stress_true, (0, 0))
    C_homog_pred = est_homog(strain_pred, stress_pred, (0, 0))

    metrics.loc[inds, "$C^*$"] = (
        L1_err(C_homog_true - C_homog_pred, norm_fac=C_homog_true).cpu().numpy()
    )

    # C1111 for true and field
    vm_stress_true = VMStress(stress_true)
    vm_stress_pred = VMStress(stress_pred)

    L1_vm_true = L1_err(vm_stress_true, spatial_avg=True)
    metrics.loc[inds, "L1 VM Stress"] = (
        L1_err(vm_stress_true - vm_stress_pred, spatial_avg=True, norm_fac=L1_vm_true)
        .cpu()
        .numpy()
    )


def apply_model(model, dataloader, model_name, output_path=None, overwrite=False):
    # apply model to a given (unshuffled) dataloader and return set of performance metrics
    model = model.eval().cuda()

    # how many instances do we have?
    num_instances = len(dataloader.dataset)

    metrics = pd.DataFrame(
        np.zeros((num_instances, len(EVAL_COLUMNS))), columns=EVAL_COLUMNS
    )

    start = 0
    running_time = 0

    # apply model in batched form
    for ind, batch in enumerate(dataloader):
        # get batch size for writing to cpu tensor
        bs = len(batch[0])
        # print(start, bs)
        C_field, bcs, strain_true, stress_true = batch

        # print("mean strain", strain_true.shape, strain_true.mean((0, -3, -2, -1)))
        # print("mean stress", stress_true.shape, stress_true.mean((0, -3, -2, -1)))

        C_field = C_field.cuda()
        bcs = bcs.cuda()

        if ind == 0 or ind == len(dataloader) - 1:
            torch.cuda.synchronize()
            # print(torch.cuda.memory_summary())
            with torch.inference_mode():
                # do some burn-in on first and last batch, with different sizes to get cudnn warmed up as well
                tmp = model(C_field[:16], bcs[:16])
                tmp = model(C_field[:64], bcs[:64])
                tmp = model(C_field, bcs)

                global first
                # burn in first run heavily
                if first:
                    print("Doing initial burn-in")
                    for _ in range(5):
                        tmp = model(C_field, bcs)

                    first = False
                torch.cuda.synchronize()
                del tmp
                time.sleep(1)

        # print(torch.cuda.memory_summary())

        # timing events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        with torch.inference_mode():
            # synchronize and get start time
            torch.cuda.synchronize()
            start_event.record()

            # run model
            batch_strains_pred = model(C_field, bcs)

            end_event.record()
            # synchronize and get elapsed time
            torch.cuda.synchronize()
        running_time += start_event.elapsed_time(end_event)

        print(f"batch {ind} time {start_event.elapsed_time(end_event):.3f} ms")

        strain_true = strain_true.cuda()
        stress_true = stress_true.cuda()

        compute_running_metrics(
            metrics,
            slice(start, start + bs - 1),
            model.scale_strain(batch_strains_pred).double().detach(),
            model.scale_strain(strain_true).double().detach(),
            model.scale_stress(stress_true).double().detach(),
            model.scale_stiffness(C_field).double().detach(),
        )

        start += bs

        C_field.cpu(), bcs.cpu(), strain_true.cpu(), stress_true.cpu()

        # get rid of references (help free up memory)
        C_field = bcs = strain_true = stress_true = None
        del batch

        # only do one loop
        # break

    # print(100 * metrics.mean())
    # print(100 * metrics.std())

    print(metrics)

    means = 100 * metrics.mean()
    stddevs = 100 * metrics.std()

    worst_strain = np.argmax(metrics["L2 Strain"])
    worst_stress = np.argmax(metrics["L2 Stress"])

    print(
        f"Worst strain error is {metrics['L2 Strain'][worst_strain]:.3f} at index {worst_strain}"
    )
    print(
        f"Worst stress error is {metrics['L2 Stress'][worst_stress]:.3f} at index {worst_stress}"
    )

    # print(means)
    # print(stddevs)
    summary_stats = pd.DataFrame(
        np.empty((1, len(EVAL_COLUMNS) + 1)).astype("str"),
        columns=EVAL_COLUMNS + ["ms / micro"],
        index=[model_name],
    )

    summary_stats.index.rename("model")

    for key in EVAL_COLUMNS:
        strs = f"{means[key]:.3f}"
        if ADD_STD:
            strs = strs + f"\pm{stddevs[key]:.2f}"
        summary_stats.loc[model_name, key] = strs

    ms_per_micro = running_time / num_instances
    summary_stats["ms / micro"] = f"{ms_per_micro:.3f}"

    return summary_stats


def get_model_pred(model, C_field, bc_vals):
    # get input at given index
    model = model.eval().cuda()
    with torch.inference_mode():
        strain_pred = model(C_field.cuda(), bc_vals.cuda())

    # return prediction for usage later
    return strain_pred.detach().cpu()


def plot_model_predictions(
    plot_model_names, model_preds, C_field_plot, strain_true_plot, stress_true_plot, ds
):
    nc = len(plot_model_names) + 1

    equiv_strain_true = equivalent(strain_true_plot, fac=2.0 / 3.0)
    equiv_stress_true = equivalent(stress_true_plot, fac=3.0 / 2.0)

    equiv_strain_true = equiv_strain_true[..., SLICE_IND].squeeze().T
    equiv_stress_true = equiv_stress_true[..., SLICE_IND].squeeze().T

    vmin_strain = equiv_strain_true.min()
    vmax_strain = equiv_strain_true.max()

    vmin_stress = equiv_stress_true.min()
    vmax_stress = equiv_stress_true.max()

    fig_stress = plt.figure(figsize=(10, 3))
    grid_stress = AxesGrid(
        fig_stress,
        111,
        nrows_ncols=(2, nc),
        axes_pad=0.4,
        # share_all=True,
        # label_mode="1",
        cbar_location="right",
        cbar_mode="edge",
        cbar_pad="5%",
        cbar_size="15%",
        # direction="column"
    )

    equiv_strain_true = None
    equiv_stress_true = None

    def get_plot_vals(ind):

        if ind == 0:
            strain = strain_true_plot
            stress = stress_true_plot
        else:
            strain = model_preds[plot_model_names[ind - 1]]
            stress = strain_to_stress(C_field_plot, strain)

        equiv_strain = equivalent(strain, fac=2.0 / 3.0)
        equiv_stress = equivalent(stress, fac=3.0 / 2.0)

        if ind > 0:
            resid = ((strain_true_plot - strain) ** 2).sum(1).sqrt()
            worst = np.argmax(resid)
            worst = np.unravel_index(worst, resid.shape)
            print(f"Worst strain ind is {worst}")
        equiv_strain = equiv_strain[..., SLICE_IND].squeeze().T
        equiv_stress = equiv_stress[..., SLICE_IND].squeeze().T

        if ind == 0:
            global equiv_strain_true, equiv_stress_true

            equiv_strain_true = equiv_strain
            equiv_stress_true = equiv_stress
            # no error for first
            strain_err = stress_err = 0 * equiv_strain
        else:
            strain_err = (equiv_strain - equiv_strain_true).abs()
            stress_err = (equiv_stress - equiv_stress_true).abs()

        ret = np.stack((equiv_strain, strain_err, equiv_stress, stress_err), 0)
        print(ret.shape)
        return ret

    plot_arrays = np.stack([get_plot_vals(ind) for ind in range(nc)], 0)

    # exclude
    vmin_strain_err = plot_arrays[1:, 1].min()
    vmax_strain_err = plot_arrays[1:, 1].max()
    vmin_stress_err = plot_arrays[1:, 3].min()
    vmax_stress_err = plot_arrays[1:, 3].max()

    fig_strain = plt.figure(figsize=(8, 4))
    grid_strain = AxesGrid(
        fig_strain,
        111,
        nrows_ncols=(2, nc),
        axes_pad=0.1,
        cbar_location="right",
        cbar_mode="edge",
        cbar_pad="5%",
        cbar_size="15%",
    )

    fig_stress = plt.figure(figsize=(8, 4))
    grid_stress = AxesGrid(
        fig_stress,
        111,
        nrows_ncols=(2, nc),
        axes_pad=0.1,
        cbar_location="right",
        cbar_mode="edge",
        cbar_pad="5%",
        cbar_size="15%",
    )

    for ind in range(nc):
        grid_strain[ind].set_xticks([])
        grid_strain[nc + ind].set_xticks([])
        grid_strain[ind].set_yticks([])
        grid_strain[nc + ind].set_yticks([])

        grid_stress[ind].set_xticks([])
        grid_stress[nc + ind].set_xticks([])
        grid_stress[ind].set_yticks([])
        grid_stress[nc + ind].set_yticks([])
        im1 = grid_strain[ind].imshow(
            plot_arrays[ind, 0],
            cmap="turbo",
            vmin=vmin_strain,
            vmax=vmax_strain,
            origin="lower",
        )
        im3 = grid_stress[ind].imshow(
            plot_arrays[ind, 2],
            cmap="turbo",
            vmin=vmin_stress,
            vmax=vmax_stress,
            origin="lower",
        )
        if ind == 0:
            grid_strain[ind].set_title("FEA")
            grid_stress[ind].set_title("FEA")

            # grid_strain[nc + ind].set_title("Microstructure")
            # grid_stress[nc + ind].set_title("Microstructure")

            # leave an empty square here
            grid_strain[nc + ind].set_axis_off()
            grid_stress[nc + ind].set_axis_off()

            # # plot Cxx
            # C_plot_slice = C_field_plot[0, 0, 0, ..., SLICE_IND].squeeze().T
            # grid_strain[nc + ind].imshow(
            #     C_plot_slice,
            #     cmap="gray",
            #     origin="lower",
            # )
            # grid_stress[nc + ind].imshow(
            #     C_plot_slice,
            #     cmap="gray",
            #     origin="lower",
            # )
        else:
            grid_strain[ind].set_title(plot_model_names[ind - 1])
            grid_stress[ind].set_title(plot_model_names[ind - 1])

            im2 = grid_strain[nc + ind].imshow(
                plot_arrays[ind, 1],
                cmap="turbo",
                vmin=vmin_strain_err,
                vmax=vmax_strain_err,
                origin="lower",
            )
            im4 = grid_stress[nc + ind].imshow(
                plot_arrays[ind, 3],
                cmap="turbo",
                vmin=vmin_stress_err,
                vmax=vmax_stress_err,
                origin="lower",
            )

    # set colorbars
    grid_strain.cbar_axes[0].colorbar(im1).set_label("Equiv. Strain")
    grid_strain.cbar_axes[1].colorbar(im2).set_label("L1 Error")
    grid_stress.cbar_axes[0].colorbar(im3).set_label("Equiv. Stress")
    grid_stress.cbar_axes[1].colorbar(im4).set_label("L1 Error")

    # set layouts
    fig_strain.tight_layout()
    fig_stress.tight_layout()
    # save figs
    fig_strain.savefig(f"paper_plots/{ds}_strain_plots.png", dpi=300)
    fig_stress.savefig(f"paper_plots/{ds}_stress_plots.png", dpi=300)


def sweep(model_dict, ds_train, ds_eval):
    # override scaling values for high-contrast
    if ds_eval == "hiCR32":
        E_VALS = [120.0, 200.0 * 120.0]
    else:
        E_VALS = E_VALS_DEF

    constlaw = StrainToStress_2phase(E_VALS, NU_VALS)
    batch_size = 200
    plot_ind = PLOT_IND
    # override for poly dataset
    if dataset_info[ds_eval].get("is_poly"):
        constlaw = StrainToStress_crystal(2000, 1000, 2000)
        batch_size = 50

        plot_ind = 0

    # big batches, no shuffling
    loader_args = {"batch_size": batch_size, "shuffle": False}

    loader = load_data(
        dataset_info[ds_eval], MODE, constlaw=constlaw, loader_args=loader_args
    )

    # make cube plots
    make_cube_plots(loader, ds_eval, plot_ind=plot_ind)

    # exit(1)

    all_model_perfs = None

    torch.cuda.synchronize()

    model_preds = {}

    C_field_plot, bc_vals_plot, strain_true_plot, stress_true_plot = loader.dataset[
        plot_ind : plot_ind + 1
    ]

    for name, mn, model_size, vx in model_dict:
        print(f"Evaluating model {mn} on dataset {ds_eval}")
        # build checkpoint to evaluate
        if mn == "fft":
            ckpt = "paper_checkpoints/fft.ckpt"
        else:
            ckpt = f"paper_checkpoints/model_{mn}_{model_size}_s{vx}_{ds_train}_best.ckpt"

        model = load_checkpoint(ckpt)
        # print(model)
        # print(model.config)

        # make sure voxel count is correct
        model.config.num_voxels = strain_true_plot.shape[-1]
        model.overrideConstlaw(constlaw)
        model.compute_scalings(E_BAR)

        print(model.strain_scaling)
        print(model.stress_scaling)
        model.config.add_resid_loss = False
        if len(model.config.deq_args) > 0:
            pass
            # # rebuild DEQ to make sure # iters is right
            model.config.deq_args["f_solver"] = "anderson"
            # model.reinitDEQ()

        print(model.config.deq_args)

        perf = None
        perf = apply_model(model, loader, name)
        print(perf)
        if all_model_perfs is None:
            all_model_perfs = perf
        else:
            all_model_perfs = pd.concat([all_model_perfs, perf])

        # now get model predictions on a given index
        model_preds[name] = get_model_pred(model, C_field_plot, bc_vals_plot)

        # move to cpu then delete
        model = model.cpu()
        del model
        model = None

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(1)

    ds_name_str = ds_train
    if ds_eval is not None:
        ds_name_str += f"_{ds_eval}"
    plot_model_predictions(
        plot_model_names,
        model_preds,
        C_field_plot,
        strain_true_plot,
        stress_true_plot,
        ds_name_str,
    )

    if all_model_perfs is not None:
        # write latex version of table
        with open(f"paper_plots/{ds_name_str}_all_models.tex", "w") as f:
            f.write(all_model_perfs.to_latex())

        all_model_perfs.to_csv(f"paper_plots/{ds_name_str}_all_models.csv")

    torch.cuda.synchronize()
    # clear out old dataloader
    del loader._iterator


def make_cube_plots(loader, ds=None, plot_ind=PLOT_IND, slice_ind=SLICE_IND):
    C_field_plot, bc_vals_plot, strain, stress = loader.dataset[plot_ind : plot_ind + 1]

    print(C_field_plot.shape, strain.shape)

    C_field_plot = torch.roll(C_field_plot, shifts=-slice_ind, dims=-1)
    strain = torch.roll(strain, shifts=-slice_ind, dims=-1)
    strain_true_plot = equivalent(strain)

    stress = torch.roll(stress, shifts=-slice_ind, dims=-1)
    stress_true_plot = equivalent(stress)

    plot_cube(
        C_field_plot[0, 0, 0].squeeze(),
        cmap="gray_r",
        savedir=f"paper_plots/{ds}_C1111.png",
        title="$C_{1111}$",
        coord=False,
    )
    plot_cube(
        strain_true_plot[0].squeeze(),
        cmap="turbo",
        savedir=f"paper_plots/{ds}_strain_true.png",
        title="Equivalent Strain",
        coord=False,
    )
    plot_cube(
        stress_true_plot[0].squeeze(),
        cmap="turbo",
        savedir=f"paper_plots/{ds}_stress_true.png",
        title="Equivalent Stress",
        coord=False,
    )
    plot_cube(
        compute_strain_energy(strain, stress).squeeze(),
        cmap="turbo",
        savedir=f"paper_plots/{ds}_energy_true.png",
        title="Strain Energy",
        coord=False,
    )


if __name__ == "__main__":
    plot_model_names = ["Big FNO", "IFNO", "FNO-DEQ", "TherINO"]
    # plot_model_names = ["IFNO", "FNO-DEQ", "Mod. F-D"]
    # plot_model_names = ["FNO", "Big FNO"]
    # plot_model_names = ["Big FNO", "TherINO", "FFT"]

    models_32 = [
        # ("FNO", "ff", "18.9M", 32),
        ("FFT", "fft", "0", None),
        ("Big FNO", "ff", "151M", 32),
        ("IFNO", "ifno", "18.9M", 32),
        ("FNO-DEQ", "fno_deq", "18.9M", 32),
        # ("Mod. F-D", "therino_notherm", "18.9M", 32),
        ("TherINO", "therino", "18.9M", 32),
        ("TherINO (Pre)", "therino_pre", "18.9M", 32),
        ("TherINO (Post)", "therino_post", "18.9M", 32),
        ("TherINO (Hybrid)", "therino_hybrid", "18.9M", 32),
    ]

    # models_32 = [
    #     ("TherINO", "therino", "18.9M", 32),
    #     ("TherINO (Pre)", "therino_pre", "18.9M", 32),
    #     ("TherINO (Post)", "therino_post", "18.9M", 32),
    #     ("TherINO (Hybrid)", "therino_hybrid", "18.9M", 32),
    # ]

    # plot_model_names = [
    #     "TherINO",
    #     "TherINO (Pre)",
    #     "TherINO (Post)",
    #     "TherINO (Hybrid)",
    # ]

    models_16 = [
        ("FNO", "ff", "2.37M", 16),
        ("Big FNO", "ff", "18.9M", 16),
        ("IFNO", "ifno", "2.37M", 16),
        ("FNO-DEQ", "fno_deq", "2.37M", 16),
        ("Mod. F-D", "therino_notherm", "2.37M", 16),
        ("TherINO", "therino", "2.37M", 16),
    ]

    # override datasets
    ds_train = "fixed32"
    if len(sys.argv) > 1:
        ds_train = sys.argv[1]

    # eval dataset if not same
    ds_eval = ds_train
    if len(sys.argv) > 2:
        ds_eval = sys.argv[2]

    if ds_train in models_16:
        models = models_16
    else:
        models = models_32

    sweep(models, ds_train, ds_eval)
