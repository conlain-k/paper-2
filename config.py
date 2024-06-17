from dataclasses import dataclass, field
from helpers import *

import os

DELIM = "-" * 40

# coefficients for balancing loss functions
lam_strain = 1
lam_stress = 1
lam_energy = 0


# penalize compatibility error heavily
lam_compat = 0

lam_sum = lam_strain + lam_stress + lam_energy

lam_strain = lam_strain / lam_sum
lam_stress = lam_stress / lam_sum
lam_energy = lam_energy / lam_sum

# residual error is usually small anyways, and we want our DEQ gradients to be accurate
lam_resid = 100


@dataclass
class Config:
    # helpers for conf loading
    _description: str = "Default args"
    _parent: str = None
    _conf_file: str = None
    image_dir: str = "images/default/"
    arch_str: str = ""

    # input features (at least one of these must be set to true!)
    use_micro: bool = False
    use_C_flat: bool = False
    use_strain: bool = False
    use_bc_strain: bool = False
    use_stress: bool = False
    use_stress_polarization: bool = False
    use_energy: bool = False
    use_FFT_resid: bool = False

    num_epochs: int = 100
    lr_max: float = 1e-3
    weight_decay: float = 0

    loader_args: dict = field(
        default_factory=lambda: {
            DataMode.TRAIN: {"batch_size": 8, "shuffle": True, "num_workers": 1},
            DataMode.VALID: {"batch_size": 256, "shuffle": False, "num_workers": 1},
            DataMode.TEST: {"batch_size": 256, "shuffle": False, "num_workers": 1},
        }
    )

    device: str = "cpu"
    return_deq_trace: bool = False

    # Should we use a fixed maximum # iters, or randomize over training
    deq_randomize_max: bool = True
    deq_min_iter: int = 5

    # num_pretrain_epochs: int = 0

    deq_args: dict = field(
        default_factory=lambda: {
            "f_solver": None,
            "b_solver": None,
            "f_max_iter": None,
            "b_max_iter": None,
            "f_tol": None,
            "b_tol": None,
            "use_ift": None,
        }
    )
    fno_args: dict = field(
        default_factory=lambda: {
            "modes": None,
            "normalize": None,
            "activ_type": None,
            "init_weight_scale": None,
            # IMPORTANT: lift into higher dim before projection (original paper does this)
            "use_weight_norm": None,
            "final_projection_channels": None,
        }
    )

    grad_clip_mag: float = 1
    use_skip_update: bool = False
    scale_output: bool = True
    enforce_mean: bool = True
    add_bcs_to_iter: bool = True

    use_EMA: bool = False

    use_deq: bool = True
    return_resid: bool = True
    add_Green_iter: bool = True

    # domain length in one direction
    num_voxels: int = 31

    H1_deriv_scaling: float = 0.1

    # default to global values, but allow overwrite
    lam_strain: float = lam_strain
    lam_stress: float = lam_stress
    lam_energy: float = lam_energy
    lam_compat: float = lam_compat
    lam_resid: float = lam_resid

    def __post_init__(self):
        conf_base = os.path.basename(self._conf_file)
        conf_base, _ = os.path.splitext(conf_base)
        self.arch_str = conf_base
        if self.use_EMA:
            self.arch_str += "_EMA"

        self.image_dir = f"images/{self.arch_str}/"

    def get_save_str(self, model, epoch, best=False):
        # get save string with info regarding this run
        params_str = human_format(count_parameters(model))
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        savestr = f"{CHECKPOINT_DIR}/model_{self.arch_str}_{params_str}_s{self.num_voxels}_{epoch}.ckpt"

        if best:
            savestr = f"{CHECKPOINT_DIR}/model_{self.arch_str}_{params_str}_s{self.num_voxels}_best.ckpt"

        return savestr


@dataclass
class LossSet:
    # holds a set of losses for a given epoch
    config: Config
    # total_loss : float = 0
    strain_loss: float = 0
    stress_loss: float = 0
    energy_loss: float = 0
    resid_loss: float = 0
    compat_loss: float = 0

    def __add__(self, other):
        # total_loss = self.total_loss + other.total_loss
        strain_loss = self.strain_loss + other.strain_loss
        stress_loss = self.stress_loss + other.stress_loss
        energy_loss = self.energy_loss + other.energy_loss
        resid_loss = self.resid_loss + other.resid_loss
        compat_loss = self.compat_loss + other.compat_loss

        return LossSet(
            self.config,
            strain_loss,
            stress_loss,
            energy_loss,
            resid_loss,
            compat_loss,
        )

    def __truediv__(self, x):
        strain_loss = self.strain_loss / x
        stress_loss = self.stress_loss / x
        energy_loss = self.energy_loss / x
        resid_loss = self.resid_loss / x
        compat_loss = self.compat_loss / x

        return LossSet(
            self.config,
            strain_loss,
            stress_loss,
            energy_loss,
            resid_loss,
            compat_loss,
        )

    def compute_total(self):
        # compute "total" loss metric as weighted average
        loss = 0
        if lam_strain > 0:
            loss += lam_strain * self.strain_loss
        if lam_stress > 0:
            loss += lam_stress * self.stress_loss
        if lam_energy > 0:
            loss += lam_energy * self.energy_loss
        if lam_compat > 0:
            loss += lam_compat * self.compat_loss
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
            self.compat_loss.detach(),
        )

    def to_dict(self):
        # get all losses as dictionary

        return {
            "strain_loss": self.strain_loss,
            "stress_loss": self.stress_loss,
            "energy_loss": self.energy_loss,
            "resid_loss": self.resid_loss,
            "compat_loss": self.compat_loss,
        }

    def __repr__(self):
        return f"strain loss is {self.strain_loss:.5}, stress loss is {self.stress_loss:.5}, energy loss is {self.energy_loss:.5}, resid loss is {self.resid_loss:.5}, compat loss is {self.compat_loss:.5}"
