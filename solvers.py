from constlaw import *
from greens_op import GreensOp
from fno import *
from layers import SimpleLayerNet, TinyVNet, ProjectionBlock
import torch
from torchdeq import get_deq, mem_gc
from torch.utils.checkpoint import checkpoint
from helpers import print_activ_map


def make_localizer(config):
    if config.use_deq:
        return Localizer_DEQ(config)
    else:
        return Localizer_FeedForward(config)


class LocalizerBase(torch.nn.Module):
    # base class for localization models
    def __init__(self, config):
        super().__init__()

        self.register_buffer("eps_bar", torch.zeros(6))
        self.constlaw = None

        # default to output strain (6 channels)
        self.output_channels = 6

        # cache config
        self.config = config

        # if in pretraining mode, don't perform certain normalizations
        # this allows us to enforce constraints (e.g. average mean) without messing up early training
        self.pretraining = True

    def count_input_channels(self):
        channels = 0
        # micro-type features
        if self.config.use_micro:
            channels += 2  # TODO change for n-phase / polycrystal
        if self.config.use_C_flat:
            channels += 21
        # strain-type features
        if self.config.use_strain:
            channels += 6
        if self.config.use_bc_strain:
            channels += 6

        # stress-type features
        if self.config.use_stress:
            channels += 6
        if self.config.use_stress_polarization:
            channels += 6

        # other thermo features
        if self.config.use_energy:
            channels += 1
        if self.config.use_FFT_resid:
            channels += 6

        return channels

    def setConstParams(self, E_vals, nu_vals, eps_bar):
        # set up constitutive model
        self.constlaw = StrainToStress_2phase(E_vals, nu_vals)
        self.eps_bar = torch.as_tensor(eps_bar)

        if self.config.use_deq:
            self.greens_op = GreensOp(self.constlaw, self.config.num_voxels)

        # take frob norm of a given quantity

        self.constlaw.compute_scalings(self.eps_bar)
        self.register_buffer(
            "scaled_average_strain",
            self.eps_bar.reshape(1, 6, 1, 1, 1) / self.constlaw.strain_scaling,
        )

    def set_constlaw_crystal(self, C11, C12, C44):
        self.constlaw = StrainToStress_crystal(C11, C12, C44)
        # make sure scalings are set correctly
        self.constlaw.compute_scalings(self.eps_bar)

    def enforce_zero_mean(self, x):
        # remove the average value from a field (for each instance, channel)
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)

    # def green_iter(self, m, strain):
    #     eps = self.greens_op(strain, m)
    #     return eps

    def filter_result(self, x):
        # no filter in pretraining
        if self.pretraining:
            return x

        # push mean to equal zero (requires it to be corrected somewhere else)
        if self.config.enforce_mean:
            x = self.enforce_zero_mean(x)

        # add mean strain as correction
        if self.config.add_bcs_to_iter:
            x += self.scaled_average_strain

        return x

    def build_nonthermo_features(self, m, C_field=None):
        # build non-thermodynamic features (e.g. stiffness, micro, BC strains);
        # call this from child classes to ensure we build the full vector!
        feat = []
        if self.config.use_micro:
            # append all phases for convenience
            feat.append(m)

        if self.config.use_C_flat:
            if C_field is None:
                C_field = (
                    self.constlaw.compute_C_field(m) / self.constlaw.stiffness_scaling
                )

            # append all phases for convenience
            feat.append(flatten_stiffness(C_field))

        # add in BC strains
        if self.config.use_bc_strain:
            eps_avg = self.compute_init_scaled_strain(m, None)
            feat.append(eps_avg)

        return feat

    def compute_init_scaled_strain(self, micro, init_guess):
        # get initial strain field (scaled appropriately)
        if init_guess is not None:
            return init_guess
        eps_shape = list(micro.shape)
        # eps should be b * 6 * xyz (since 6 indep strain components)
        # TODO get # voxels in a more stable fashion
        Nx = eps_shape[-2]

        # broadcast avg strain to all locations
        init_strain = (
            micro.new_ones((eps_shape[0], 6, Nx, Nx, Nx)) * self.scaled_average_strain
        )

        return init_strain


class Localizer_FeedForward(LocalizerBase):
    # base class for localization models
    def __init__(self, config):
        super().__init__(config)
        self.net = FNO(
            in_channels=self.count_input_channels(),
            out_channels=6,
            **config.fno_args,
        )

    def forward(self, m):
        # compute input features (micro, c_flat, bcs, etc.)
        x = torch.concatenate(self.build_nonthermo_features(m), dim=1)

        # print(x.detach().cpu().mean(dim=(-3, -2, -1, 0)), x.std(dim=(-3, -2, -1, 0)))

        # apply NN to encoded input
        x = self.net(x)

        # push mean to equal zero (requires it to be corrected somewhere else)
        if self.config.enforce_mean:
            x = self.enforce_zero_mean(x)

        if self.config.scale_output:
            x *= self.constlaw.strain_scaling

        if self.config.add_bcs_to_iter:
            x += self.eps_bar.reshape(1, 6, 1, 1, 1)

        # print(x.detach().cpu().mean(dim=(-3, -2, -1, 0)), x.std(dim=(-3, -2, -1, 0)))

        return x


class Localizer_DEQ(LocalizerBase):
    def __init__(self, config):
        super().__init__(config)
        # deep-EQ solver
        self.deq = get_deq(config.deq_args)

        input_channels = self.count_input_channels()

        if config.add_Green_iter:
            # add FFT prediction and features as well
            input_channels *= 2

        self.forward_net = FNO(
            in_channels=input_channels,
            out_channels=6,
            **config.fno_args,
        )

    def encode_micro_strain(self, m, C_field, strain):

        feat = self.build_nonthermo_features(m, C_field)

        # do we use current strain as feature?
        if self.config.use_strain:
            feat.append(strain)

        if self.config.use_stress or self.config.use_energy:
            # precompute stress for convenience
            stress = self.constlaw(strain, C_field)

        # build up features
        if self.config.use_stress:
            feat.append(stress)

        if self.config.use_energy:
            strain_energy = compute_strain_energy(strain, stress)
            feat.append(strain_energy)

        if self.config.use_stress_polarization:
            stress_polar = self.constlaw.stress_pol(strain, C_field, scaled=True)
            # negate stress polarization to get positive-ish values
            stress_polar *= -1
            feat.append(stress_polar)

        # collect features into a vector
        nn_features = torch.concatenate(feat, dim=1)

        # now return dense feat vec
        return nn_features

    def single_iter_simple(self, strain_k, C_field, m):
        """
        Given a stiffness tensor C corresponding to micro m, update the current strain field using the neural net
        """
        z_k = self.encode_micro_strain(m, C_field, strain_k)
        # print(
        #     z_k.detach().cpu().mean(dim=(-3, -2, -1, 0)),
        #     z_k.detach().cpu().std(dim=(-3, -2, -1, 0)),
        # )

        if self.config.add_Green_iter:
            # get moulinec-suquet update
            # NOTE: block gradients flowing through this step (e.g. don't differentiate through MS step)
            # hopefully this stabilizes training
            eps_ms = self.greens_op.forward(strain_k, C_field)
            z_ms = self.encode_micro_strain(m, C_field, eps_ms)

            # stack on M-S features after current features
            z_k = torch.concatenate([z_k, z_ms], dim=1)

        # predict new strain perturbation
        strain_kp = self.forward_net(z_k)
        # print(
        #     strain_kp.detach().cpu().mean(dim=(-3, -2, -1, 0)),
        #     strain_kp.detach().cpu().std(dim=(-3, -2, -1, 0)),
        # )

        assert not torch.isnan(strain_kp).any()

        return strain_kp

    def forward(self, m):

        C_field = self.constlaw.compute_C_field(m) / self.constlaw.stiffness_scaling

        # just iterate over strain dim directly
        F = lambda h: self.single_iter_simple(h, C_field, m)
        h0 = self.compute_init_scaled_strain(m, None)

        # randomize # iters during train time
        if self.config.deq_randomize_max and self.training:
            min = self.config.deq_min_iter
            avg = self.config.deq_args["f_max_iter"]
            # average # iters given by config (makes it easier to evaluate later)
            it_max = 2 * avg - min

            max_iters = torch.randint(min, it_max, (1,))
            iter_arg = {"f_max_iter": max_iters.item()}
        else:
            iter_arg = None

        if self.pretraining:
            # just do 1 pass in pretrain mode (to quickly get main features)
            hstar = F(h0)
        else:
            # solve FP over h system
            sol, _ = self.deq(F, h0, solver_kwargs=iter_arg)

            hstar = sol[-1]

        out_scale = self.constlaw.strain_scaling if self.config.scale_output else 1

        # now project out output and rescale appropriately
        strain_pred = self.filter_result(hstar) * out_scale

        if self.config.return_deq_trace:
            # just return raw deq trace without postprocessing
            return [s * out_scale for s in sol]

        # return model and residual
        elif self.config.return_resid:
            resid = (F(hstar) - hstar) * out_scale
            return strain_pred, resid

        else:
            return strain_pred
