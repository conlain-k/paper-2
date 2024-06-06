from dataclasses import dataclass, field
from enum import Enum
from helpers import *

import os

DELIM = "-" * 40


# Python 3.9 is weird about StrEnum
class DataMode(str, Enum):
    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"


SCRATCH_DIR = "/storage/home/hcoda1/3/ckelly84/scratch/"

MICRO_TRAIN = SCRATCH_DIR + "micros/paper2_smooth_train.h5"
MICRO_VALID = SCRATCH_DIR + "micros/paper2_smooth_valid.h5"
MICRO_TEST = SCRATCH_DIR + "micros/paper2_smooth_test.h5"
RESP_TRAIN = SCRATCH_DIR + "outputs/paper2_smooth_cr100.0_bc0_responses_train.h5"
RESP_VALID = SCRATCH_DIR + "outputs/paper2_smooth_cr100.0_bc0_responses_valid.h5"
RESP_TEST = SCRATCH_DIR + "outputs/paper2_smooth_cr100.0_bc0_responses_test.h5"

CHECKPOINT_DIR = "checkpoints"

datasets = {
    DataMode.TRAIN: {"micro_file": MICRO_TRAIN, "resp_file": RESP_TRAIN},
    DataMode.VALID: {"micro_file": MICRO_VALID, "resp_file": RESP_VALID},
    DataMode.TEST: {"micro_file": MICRO_TEST, "resp_file": RESP_TEST},
}


# coefficients for balancing loss functions
lam_strain = 1
lam_stress = 1
lam_energy = 0
# lam_stressdiv = 0.1
lam_stressdiv = 0

lam_sum = lam_strain + lam_stress + lam_energy + lam_stressdiv

lam_strain = lam_strain / lam_sum
lam_stress = lam_stress / lam_sum
lam_energy = lam_energy / lam_sum
lam_stressdiv = lam_stressdiv / lam_sum

# residual error is usually small anyways, and we want our DEQ gradients to be accurate
lam_resid = 100


@dataclass
class Config:
    # helpers for conf loading
    _description: str = "Default args"
    _parent: str = None
    _conf_file: str = None
    image_dir: str = "images/default"
    arch_str: str = ""

    # train info
    num_epochs: int = 200
    lr_init: float = 1e-3
    weight_decay: float = 0

    loader_args: dict = field(
        default_factory=lambda: {
            DataMode.TRAIN: {"batch_size": 16, "shuffle": True, "num_workers": 1},
            DataMode.VALID: {"batch_size": 128, "shuffle": False, "num_workers": 1},
            DataMode.TEST: {"batch_size": 32, "shuffle": False, "num_workers": 1},
        }
    )

    deq_args: dict = field(
        default_factory=lambda: {
            "f_solver": "anderson",
            "b_solver": "anderson",
            "f_max_iter": 16,
            "b_max_iter": 16,
            "f_tol": 1e-4,
            "b_tol": 1e-5,
            # use last 3 steps
            # "grad": 5,
            "use_ift": True,
        }
    )
    fno_args: dict = field(
        default_factory=lambda: {
            "modes": (12,),
            "normalize": True,
            "activ_type": "gelu",
            "init_weight_scale": 0.01,
            # IMPORTANT: lift into higher dim before projection (original paper does this)
            "use_weight_norm": True,
        }
    )
    # how many channels to use in final projection block?
    projection_channels: int = 128
    network_args: dict = field(
        default_factory=lambda: {"inner_channels": 48, "num_blocks": 2}
    )
    device: str = "cpu"

    num_aux_dim: int = 0

    # otherwise use inception net
    use_fno: bool = False

    use_micro: bool = False
    use_strain: bool = True
    use_bc_strain: bool = True
    use_stress: bool = True
    use_stress_polarization: bool = False
    use_energy: bool = True

    # whether to output strain or displacement
    output_displacement: bool = False
    compute_stressdiv: bool = True

    grad_clip_mag: float = 10
    use_skip_update: bool = False
    enforce_mean: bool = True

    use_deq: bool = True
    use_fancy_iter: bool = False
    return_resid: bool = True
    add_Green_iter: bool = True
    teacher_forcing: bool = False
    latent_dim: int = 32

    # domain length in one direction
    num_voxels: int = 31

    H1_deriv_scaling: float = 10

    def get_save_str(self, model, epoch):
        # get save string with info regarding this run
        params_str = human_format(count_parameters(model))
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        return f"{CHECKPOINT_DIR}/model_{self.arch_str}_{params_str}_s{self.N}_{epoch}.ckpt"


@dataclass
class LossSet:
    # holds a set of losses for a given epoch
    config: Config
    # total_loss : float = 0
    strain_loss: float = 0
    stress_loss: float = 0
    energy_loss: float = 0
    resid_loss: float = 0
    stressdiv_loss: float = 0

    def __add__(self, other):
        # total_loss = self.total_loss + other.total_loss
        strain_loss = self.strain_loss + other.strain_loss
        stress_loss = self.stress_loss + other.stress_loss
        energy_loss = self.energy_loss + other.energy_loss
        resid_loss = self.resid_loss + other.resid_loss
        stressdiv_loss = self.stressdiv_loss + other.stressdiv_loss

        return LossSet(
            self.config,
            strain_loss,
            stress_loss,
            energy_loss,
            resid_loss,
            stressdiv_loss,
        )

    def __truediv__(self, x):
        strain_loss = self.strain_loss / x
        stress_loss = self.stress_loss / x
        energy_loss = self.energy_loss / x
        resid_loss = self.resid_loss / x
        stressdiv_loss = self.stressdiv_loss / x

        return LossSet(
            self.config,
            strain_loss,
            stress_loss,
            energy_loss,
            resid_loss,
            stressdiv_loss,
        )

    def compute_total(self):
        # compute "total" loss metric as weighted average
        loss = (
            lam_strain * self.strain_loss
            + lam_stress * self.stress_loss
            + lam_energy * self.energy_loss
            # + lam_stressdiv * self.stressdiv_loss
        )
        if self.config.use_deq:
            loss += lam_resid * self.resid_loss

        return loss

    def detach(self):
        return LossSet(
            self.config,
            self.strain_loss.detach(),
            self.stress_loss.detach(),
            self.energy_loss.detach(),
            self.resid_loss.detach(),
            self.stressdiv_loss.detach(),
        )

    def to_dict(self):
        # get all losses as dictionary

        return {
            "strain_loss": self.strain_loss,
            "stress_loss": self.stress_loss,
            "energy_loss": self.energy_loss,
            "resid_loss": self.resid_loss,
            "stressdiv_loss": self.stressdiv_loss,
        }

    def __repr__(self):
        return f"strain loss is {self.strain_loss:.5}, stress loss is {self.stress_loss:.5}, energy loss is {self.energy_loss:.5}, resid loss is {self.resid_loss:.5}, stressdiv loss is {self.stressdiv_loss:.5}"
