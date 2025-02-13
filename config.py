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
    _conf_file: str = ""
    image_dir: str = "images/default/"
    arch_str: str = ""
    model_type: str = None
    # name of traiing dataset
    train_dataset_name: str = None

    num_epochs: int = 200
    lr_max: float = 1e-3

    # whether to override lambdas and balance loss terms manually
    balance_losses: bool = False
    grad_clip_mag: float = 1
    use_EMA: bool = False

    # input features (at least one of these must be set to true!)
    use_micro: bool = False
    use_C_flat: bool = False
    use_bc_strain: bool = True

    # default to global values, but allow overwrite
    lam_strain: float = lam_strain
    lam_stress: float = lam_stress
    lam_energy: float = lam_energy
    lam_resid: float = lam_resid

    # used to simpify results filtering
    finalize: bool = True

    # use regular L2 norm (rather than squared)
    # adds cost / complexity, but makes balancing terms easier
    # use_sqrt_loss: bool = False

    # args when thermo features are used
    # "use_strain": None
    # "use_stress": None
    # "use_stress_polarization": None
    # "use_energy": None
    thermo_feat_args: dict = field(default_factory=lambda: {})

    # args for dataloader
    loader_args: dict = field(
        default_factory=lambda: {
            DataMode.TRAIN: {"batch_size": 32, "shuffle": True, "num_workers": 4},
            DataMode.VALID: {"batch_size": 128, "shuffle": False, "num_workers": 4},
            DataMode.TEST: {"batch_size": 128, "shuffle": False, "num_workers": 4},
        }
    )

    scale_output: bool = True
    enforce_mean: bool = True
    add_bcs_to_iter: bool = True

    # penalize DEQ residual of solution?
    add_resid_loss: bool = False

    # penalize DEQ residual of true solution?
    penalize_teacher_resid: bool = False
    # penalize misalignment between DEQ residual and error?
    penalize_resid_misalignment: bool = False

    # Should we use a fixed maximum # iters, or randomize over training
    # only used for DEQ-type models
    deq_randomize_max: bool = True
    deq_min_iter: int = 2

    # should we use zero for TherINO initial iterate?
    therino_init_zero: bool = False
    # number of IFNO iterations (only used for IFNO)
    num_ifno_iters: int = None

    # passthrough args to DEQ
    deq_args: dict = field(default_factory=lambda: {})
    # args for fno pieces
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

    latent_channels: int = None

    # args for hybrid mode only
    hybrid_args: dict = field(
        default_factory=lambda: {
            # add encoding of FFT update into normal iteration
            "add_fft_encoding": None,
            # do one FFT step first
            "use_fft_pre_iter": None,
            # do one step FFT after (can be combined with above)
            "use_fft_post_iter": None,
        }
    )

    # domain length in one direction
    num_voxels: int = None
    # how much to upsample internally for greens op
    greens_upsample: int = 1

    # H1_deriv_scaling: float = 0.1

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

        savestr = f"{CHECKPOINT_DIR}/model_{self.arch_str}_{params_str}_s{self.num_voxels}_{self.train_dataset_name}_{epoch}.ckpt"

        if best:
            savestr = f"{CHECKPOINT_DIR}/model_{self.arch_str}_{params_str}_s{self.num_voxels}_{self.train_dataset_name}_best.ckpt"

        return savestr

    def using_thermo_features(self):
        # check if we are using any thermodynamic encodings
        return bool(self.thermo_feat_args)
