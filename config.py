from dataclasses import dataclass, field
from helpers import *

import os

DELIM = "-" * 40

# coefficients for balancing loss functions
lam_strain = 1
lam_stress = 1
lam_energy = 0
lam_err_energy = 1
# lam_stressdiv = 0.1
lam_stressdiv = 0

lam_sum = lam_strain + lam_stress + lam_energy + lam_stressdiv + lam_err_energy

lam_strain = lam_strain / lam_sum
lam_stress = lam_stress / lam_sum
lam_energy = lam_energy / lam_sum
# lam_err_energy = lam_err_energy / lam_sum
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
    lr_max: float = 1e-3
    weight_decay: float = 0

    loader_args: dict = field(
        default_factory=lambda: {
            DataMode.TRAIN: {"batch_size": 16, "shuffle": True, "num_workers": 2},
            DataMode.VALID: {"batch_size": 256, "shuffle": False, "num_workers": 2},
            DataMode.TEST: {"batch_size": 256, "shuffle": False, "num_workers": 2},
        }
    )

    # Should we use a fixed maximum # iters, or randomize over training
    deq_randomize_max: bool = True
    deq_min_iter: int = 5

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
            "final_projection_channels": 128,
        }
    )
    # how many channels to use in final projection block?
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
    # output_displacement: bool = False
    compute_stressdiv: bool = True

    grad_clip_mag: float = 100
    use_skip_update: bool = False
    enforce_mean: bool = True

    use_EMA: bool = False

    use_deq: bool = True
    use_fancy_iter: bool = False
    return_resid: bool = True
    add_Green_iter: bool = True
    # teacher_forcing: bool = False
    latent_dim: int = 32

    # domain length in one direction
    num_voxels: int = 31

    H1_deriv_scaling: float = 10

    # default to global values, but allow overwrite
    lam_strain: float = lam_strain
    lam_stress: float = lam_stress
    lam_energy: float = lam_energy
    lam_stressdiv: float = lam_stressdiv
    lam_resid: float = lam_resid

    def __post_init__(self):
        conf_base = os.path.basename(self._conf_file)
        conf_base, _ = os.path.splitext(conf_base)
        self.arch_str = conf_base
        if self.use_EMA:
            self.arch_str += "_EMA"

        self.image_dir = f"images/{self.arch_str}"

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
    err_energy_loss: float = 0
    resid_loss: float = 0
    stressdiv_loss: float = 0

    def __add__(self, other):
        # total_loss = self.total_loss + other.total_loss
        strain_loss = self.strain_loss + other.strain_loss
        stress_loss = self.stress_loss + other.stress_loss
        energy_loss = self.energy_loss + other.energy_loss
        err_energy_loss = self.err_energy_loss + other.err_energy_loss
        resid_loss = self.resid_loss + other.resid_loss
        stressdiv_loss = self.stressdiv_loss + other.stressdiv_loss

        return LossSet(
            self.config,
            strain_loss,
            stress_loss,
            energy_loss,
            err_energy_loss,
            resid_loss,
            stressdiv_loss,
        )

    def __truediv__(self, x):
        strain_loss = self.strain_loss / x
        stress_loss = self.stress_loss / x
        energy_loss = self.energy_loss / x
        err_energy_loss = self.err_energy_loss / x
        resid_loss = self.resid_loss / x
        stressdiv_loss = self.stressdiv_loss / x

        return LossSet(
            self.config,
            strain_loss,
            stress_loss,
            energy_loss,
            err_energy_loss,
            resid_loss,
            stressdiv_loss,
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
        if lam_stressdiv > 0:
            loss += lam_stressdiv * self.stressdiv_loss

        if lam_err_energy > 0:
            loss += lam_err_energy * self.err_energy_loss

        if self.config.use_deq:
            loss += lam_resid * self.resid_loss

        return loss

    def detach(self):
        return LossSet(
            self.config,
            self.strain_loss.detach(),
            self.stress_loss.detach(),
            self.energy_loss.detach(),
            self.err_energy_loss.detach(),
            self.resid_loss.detach(),
            self.stressdiv_loss.detach(),
        )

    def to_dict(self):
        # get all losses as dictionary

        return {
            "strain_loss": self.strain_loss,
            "stress_loss": self.stress_loss,
            "energy_loss": self.energy_loss,
            "err_energy_loss": self.err_energy_loss,
            "resid_loss": self.resid_loss,
            "stressdiv_loss": self.stressdiv_loss,
        }

    def __repr__(self):
        return f"strain loss is {self.strain_loss:.5}, stress loss is {self.stress_loss:.5}, energy loss is {self.energy_loss:.5}, err energy loss is {self.err_energy_loss:.5}, resid loss is {self.resid_loss:.5}, stressdiv loss is {self.stressdiv_loss:.5}"
