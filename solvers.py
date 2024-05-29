from constlaw import *
from greens_op import GreensOp
from fno import FNO
from layers import SimpleLayerNet, TinyVNet, ProjectionBlock
import torch
from torchdeq import get_deq, mem_gc
from torch.utils.checkpoint import checkpoint
from helpers import print_activ_map


def make_model(config, input_channels, output_channels):
    # neural net component takes in strain, stress, energy and outputs strain
    if config.use_fno:
        # use full FNO
        return FNO(
            in_channels=input_channels,
            out_channels=output_channels,
            mid_channels=config.latent_dim,
            **config.fno_args,
        )
    else:
        return SimpleLayerNet(
            in_channels=input_channels,
            out_channels=output_channels,
            **config.network_args,
        )


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

    def setConstParams(self, E_vals, nu_vals, eps_bar):
        # set up constitutive model
        self.constlaw = StrainToStress_2phase(E_vals, nu_vals)
        self.eps_bar = torch.as_tensor(eps_bar)

        # if self.config.add_Green_iter:
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

        if self.config.enforce_mean:
            x = self.enforce_zero_mean(x)

        # either way, add mean strain as correction
        x += self.scaled_average_strain

        return x

    # def forward(self, m, init_guess=None):
    #     raise NotImplementedError


class Localizer_FeedForward(LocalizerBase):
    # base class for localization models
    def __init__(self, config):
        super().__init__(config)
        if self.config.use_C_flat:
            input_channels = 21
        else:
            input_channels = 2

        self.net = make_model(config, input_channels=input_channels, output_channels=6)

    def forward(self, m):

        if self.config.use_C_flat:
            C_field = self.constlaw.compute_C_field(m) / self.constlaw.stiffness_scaling
            # use flattened stiffness as micro
            m = flatten_stiffness(C_field)

        x = self.net(m)
        x = self.filter_result(x) * self.constlaw.strain_scaling

        return x


class Localizer_DEQ(LocalizerBase):
    def __init__(self, config):
        super().__init__(config)
        # deep-EQ solver
        self.deq = get_deq(config.deq_args)

        # always input strain
        self.num_strain_feats = 6

        # add channels as appropriate
        if config.use_micro:
            self.num_strain_feats += 2
        if config.use_C_flat:
            self.num_strain_feats += 21
        if config.use_stress:
            self.num_strain_feats += 6
        if config.use_stress_polarization:
            self.num_strain_feats += 6
        if config.use_energy:
            self.num_strain_feats += 1

        channels_in = self.num_strain_feats
        if config.add_Green_iter:
            # add FFT prediction and features as well
            channels_in *= 2

        # now make network either way and let lifting block do the hard work
        self.forward_net = make_model(
            config, input_channels=channels_in, output_channels=6
        )

    def encode_micro_strain(self, strain, C_field, m):
        # encode microstructure and strain field
        # always use strain as input
        feat = [strain]

        # precompute stress for convenience
        stress = self.constlaw(strain, C_field)

        if self.config.use_micro:
            # append all phases for convenience
            feat.append(m)

        if self.config.use_C_flat:
            # append all phases for convenience
            feat.append(flatten_stiffness(C_field))

        # build up features
        if self.config.use_stress:
            feat.append(stress)

        if self.config.use_stress_polarization:
            stress_polar = self.constlaw.stress_pol(strain, C_field, scaled=True)
            # negate stress polarization to get positive-ish values
            # stress_polar *= -1
            feat.append(stress_polar)

        if self.config.use_energy:
            strain_energy = compute_strain_energy(strain, stress)
            feat.append(strain_energy)

        # if self.config.use_bc_strain:
        #     eps_avg = self.compute_init_scaled_strain(m, None)
        #     feat.append(eps_avg )

        # collect features into a vector
        nn_features = torch.concatenate(feat, dim=1)

        # now return dense feat vec
        return nn_features

    def compute_init_scaled_strain(self, micro, init_guess):
        # micro should be
        # get initial strain field
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

    def single_iter_simple(self, strain_k, C_field, m):
        """
        Given a stiffness tensor C corresponding to micro m, update the current strain field using the neural net
        """
        z_k = self.encode_micro_strain(strain_k, C_field, m)

        # print_activ_map(z_k.detach())

        if self.config.add_Green_iter:
            # get moulinec-suquet update
            # NOTE: block gradients flowing through this step (e.g. don't differentiate through MS step)
            # hopefully this stabilizes training
            eps_ms = self.greens_op.forward(strain_k, C_field)
            z_ms = self.encode_micro_strain(eps_ms, C_field, m)

            # stack on M-S features after current features
            z_k = torch.concatenate([z_k, z_ms], dim=1)

        # predict new strain perturbation
        strain_kp = self.forward_net(z_k)

        if self.config.use_skip_update:
            strain_kp += strain_k

        strain_kp = self.filter_result(strain_kp)

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

        # solve FP over h system
        sol, _ = self.deq(F, h0, solver_kwargs=iter_arg)
        hstar = sol[-1]

        if self.config.use_fancy_iter:
            # only project out strain if needed
            strain_pred = hstar[:, :6] * self.constlaw.strain_scaling
        else:
            strain_pred = hstar * self.constlaw.strain_scaling

        if self.config.return_deq_trace:
            # just return raw deq trace without postprocessing
            return [s * self.constlaw.strain_scaling for s in sol]

        # return model and residual
        elif self.config.return_resid:
            resid = (F(hstar) - hstar) * self.constlaw.strain_scaling
            return strain_pred, resid

        else:
            return strain_pred
