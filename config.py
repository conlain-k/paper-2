from dataclasses import dataclass, field
from helpers import *

import os

DELIM = "-" * 40

# coefficients for balancing loss functions
lam_strain = 1
lam_stress = 1
lam_energy = 0

# lam_sum = lam_strain + lam_stress + lam_energy

# lam_strain = lam_strain / lam_sum
# lam_stress = lam_stress / lam_sum
# lam_energy = lam_energy / lam_sum

# residual error is usually small anyways, and we want our DEQ gradients to be accurate
lam_resid = 1


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

    num_epochs: int = 200
    lr_max: float = 1e-3

    loader_args: dict = field(
        default_factory=lambda: {
            DataMode.TRAIN: {"batch_size": 8, "shuffle": True, "num_workers": 1},
            DataMode.VALID: {"batch_size": 256, "shuffle": False, "num_workers": 1},
            DataMode.TEST: {"batch_size": 256, "shuffle": False, "num_workers": 1},
        }
    )

    # whether to override lambdas and balance loss terms manually
    balance_losses: bool = False

    # device: str = "cpu"
    return_deq_trace: bool = False

    # Should we use a fixed maximum # iters, or randomize over training
    deq_randomize_max: bool = True
    deq_min_iter: int = 4

    # passthrough args to DEQ
    deq_args: dict = field(default_factory=lambda: {})
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
    # add encoding of FFT update into normal iteration
    add_fft_encoding: bool = False
    # do one FFT step first
    use_fft_pre_iter: bool = False
    # do one step FFT after (can be combined with above)
    use_fft_post_iter: bool = False

    # domain length in one direction
    num_voxels: int = None

    H1_deriv_scaling: float = 0.1

    # default to global values, but allow overwrite
    lam_strain: float = lam_strain
    lam_stress: float = lam_stress
    lam_energy: float = lam_energy
    lam_resid: float = lam_resid

    # use regular L2 norm (rather than squared)
    # adds cost / complexity, but makes balancing terms easier
    use_sqrt_loss: bool = False

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

    def using_thermo_features(self):
        # check if we are using any thermodynamic encodings
        return (
            self.use_strain
            or self.use_stress
            or self.use_stress_polarization
            or self.use_energy
            or self.add_fft_encoding
            or self.use_fft_pre_iter
            or self.use_fft_post_iter
        )
