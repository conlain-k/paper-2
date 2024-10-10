from h5py import File
from matplotlib import pyplot as plt
import numpy as np

SCRATCH = "/storage/home/hcoda1/3/ckelly84/scratch/"
f = File(SCRATCH + "micros/materials_data.h5", "a")

rf = File(SCRATCH + "outputs/paper2_smooth_train_u1_responses.h5")

fig, ax = plt.subplots(1, 2)
ax[0].imshow(f["micros"][0, 0, ..., 0])
ax[1].imshow(rf["strain"][0, 0, ..., 0])

plt.savefig("preproc_comp.png")

print(rf["strain"].shape)
print(rf["strain"].compression)
print(rf["strain"].chunks)

if "homog" not in f.keys():
    print("copying data")
    strain = rf["strain"][:]
    stress = rf["stress"][:]

    print("stressmean", stress[:, 0, ...].mean((-3, -2, -1)).shape)
    print("computing homog vals")
    homog = stress[:, 0, ...].mean((-3, -2, -1)) / strain[:, 0, ...].mean((-3, -2, -1))
    print("computing energy vals")
    energy = np.einsum("brijk, brijk -> b", strain, stress)

    print("copying to disk")
    f["homog"] = homog
    f["energy"] = energy

    print(homog.shape, homog[:10])
    print(energy.shape, energy[:10])

if "X" not in f.keys():
    print(f["params_vf"][:].shape)
    X = np.stack(
        (f["params_vf"][:], f["params_gx"][:], f["params_gy"][:], f["params_gz"][:]), 1
    )
    y = np.stack((f["homog"], f["energy"]), 1)
    print(X.shape, y.shape)

    f["X"] = X
    f["y"] = y
