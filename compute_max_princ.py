from h5py import File
from helpers import *
from tensor_ops import *
from constlaw import StrainToStress_2phase

import einops
import torch

from main import load_data, dataset_info

num_keep = 64


def get_stress_tail(stress_field):
    # reshape into matrix
    # print("field", stress_field.shape)
    stress_mat = mandel_to_mat_3x3(stress_field)
    # push matrix indices to end, then flatten spatial dims
    stress_mat = einops.rearrange(stress_mat, "b i j x y z -> b (x y z) i j")
    # print("mat", stress_mat.shape)
    princ_stresses = torch.linalg.eigvalsh(stress_mat)
    # print("princ", princ_stresses.shape)
    max_princ_stresses = princ_stresses[..., 0]
    # print("max", max_princ_stresses.shape)

    num_vox = stress_mat.shape[1]
    num_keep = round(num_vox / 100)
    # sort principal stresses by magnitude and take top k
    values, _ = torch.topk(max_princ_stresses, num_keep, dim=-1)

    # print("top", values.shape)

    return values.detach().cpu().numpy()


def compute_tail_stresses(name, downsamp_fac=1):
    loader = load_data(
        dataset_info[name],
        DataMode.VALID,
        constlaw=StrainToStress_2phase([1, 1], [0, 0]),
        loader_args={"batch_size": 32, "shuffle": False, "num_workers": 8},
    )
    dataset = loader.dataset

    mf = dataset.mf

    num_samp = len(dataset)

    output_file = File(f"principal_stresses_data_{name}_{downsamp_fac}.h5", "w")
    output_file.create_dataset(
        "design_params", data=torch.zeros((num_samp, 4), dtype=float)
    )
    output_file["design_params"][:, 0] = dataset.getData(mf, "params_vf")
    output_file["design_params"][:, 1] = dataset.getData(mf, "params_gx")
    output_file["design_params"][:, 2] = dataset.getData(mf, "params_gy")
    output_file["design_params"][:, 3] = dataset.getData(mf, "params_gz")

    start = 0

    for _, _, _, stress_field in loader:
        # if start > 420:
        #     break
        # print(stress_field.shape)
        stop = start + stress_field.shape[0]
        print(start, stop)
        stress_field = stress_field.cuda()
        if downsamp_fac > 1:
            stress_field = average_field(stress_field, fac=downsamp_fac)

        stress_tail = get_stress_tail(stress_field)

        if "stress_tails" not in output_file.keys():
            output_file.create_dataset(
                "stress_tails",
                data=torch.zeros((num_samp, stress_tail.shape[-1]), dtype=float),
            )
        output_file["stress_tails"][start:stop] = stress_tail
        start = stop

    return output_file


stress_file_32 = compute_tail_stresses("fixed16_u2")
stress_file_32_down = compute_tail_stresses("fixed16_u2", downsamp_fac=2)
stress_file_16 = compute_tail_stresses("fixed16")


def plot_percentiles(stresses, vox, label=None):
    num = stresses.shape[-1]
    offset = 100.0 * (vox - num + np.arange(1, num + 1))[::-1] / vox
    plt.scatter(offset, stresses, label=label, linestyle="-", marker=".")


IND = -1

plt.figure()
plot_percentiles(stress_file_16["stress_tails"][IND], 16**3, label="16")
plot_percentiles(stress_file_32["stress_tails"][IND], 32**3, label="32")
plot_percentiles(stress_file_32_down["stress_tails"][IND], 16**3, label="32_downsamp")
plt.legend()
# plt.gca().invert_xaxis()
plt.xlabel("Percentile")
plt.ylabel("Max. Princ. Stress")
plt.tight_layout()
plt.savefig(f"max_princ_res_{IND}.png", dpi=300)
