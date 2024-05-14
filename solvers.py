from constlaw import (
    StrainToStress_2phase,
    StrainToStress_crystal,
    compute_strain_from_displacment,
    compute_strain_energy,
    central_diff_3d,
)
from greens_op import GreensOp
from fno import FNO
from layers import SimpleLayerNet, TinyVNet, ProjectionBlock
import torch
from torchdeq import get_deq, mem_gc
from torch.utils.checkpoint import checkpoint


def make_model(config, input_channels, output_channels):
    # neural net component takes in strain, stress, energy and outputs strain
    if config.use_fno:
        # use full FNO
        return FNO(
            in_channels=input_channels,
            out_channels=output_channels,
            mid_channels=config.latent_dim,
            **config.fno_args
        )
    else:
        return SimpleLayerNet(
            in_channels=input_channels,
            out_channels=output_channels,
            **config.network_args
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

        # if config.use_bc_strain:
        #     input_channels += 6

        # default to output strain (6 channels)
        self.output_channels = 6

        # cache config
        self.config = config

    def setConstParams(self, E_vals, nu_vals, eps_bar):
        # set up constitutive model
        self.constlaw = StrainToStress_2phase(E_vals, nu_vals)
        self.eps_bar = torch.as_tensor(eps_bar)

        if self.config.add_Green_iter:
            self.greens_op = GreensOp(self.constlaw, self.config.num_voxels)

        # take frob norm of a given quantity
        frob = lambda x: (x**2).sum().sqrt()

        # print("eps", self.eps_bar.shape)
        # print("C", self.constlaw.C_ref.shape)
        # print((self.eps_bar @ self.constlaw.C_ref @ self.eps_bar).shape)
        self.strain_scaling = frob(self.eps_bar)

        # stress corresponding to a scaled strain
        self.stress_scaling = frob(self.constlaw.C_ref @ self.eps_bar)
        self.energy_scaling = frob(
            (self.eps_bar @ (self.constlaw.C_ref @ self.eps_bar))
        )

        # print(self.strain_scaling, self.stress_scaling, self.energy_scaling)
        # print(self.eps_bar)
        # exit(1)

    def set_constlaw_crystal(self, C11, C12, C44):
        self.constlaw = StrainToStress_crystal(C11, C12, C44)

    def enforce_zero_mean(self, x):
        # remove the average value from a field (for each instance, channel)
        xmean = x.mean(dim=(-3, -2, -1), keepdim=True)
        xp = x - xmean
        # print("mean", xmean[0].mean(dim=(-3, -2, -1)))
        # print("X", x[0].mean(dim=(-3, -2, -1)))
        # print("xp", xp[0].mean(dim=(-3, -2, -1)))
        return xp

    # def green_iter(self, m, strain):
    #     eps = self.greens_op(strain, m)
    #     return eps

    def filter_result(self, x):
        if self.config.enforce_mean:
            x = self.enforce_zero_mean(x)

        # either way, add mean strain as correction
        x += self.eps_bar.reshape(1, 6, 1, 1, 1)

        return x

    # def forward(self, m, init_guess=None):
    #     raise NotImplementedError


class Localizer_FeedForward(LocalizerBase):
    # base class for localization models
    def __init__(self, config):
        super().__init__(config)

        self.net = make_model(config, input_channels=2, output_channels=6)

    def forward(self, m):

        x = self.net(m)
        x = self.filter_result(x)

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
        feat = [strain / self.strain_scaling]

        # precompute stress for convenience
        stress = self.constlaw(strain, C_field)

        if self.config.use_micro:
            # append all phases for convenience
            feat.append(m)

        # build up features
        if self.config.use_stress:
            feat.append(stress / self.stress_scaling)

        if self.config.use_stress_polarization:
            stress_polar = self.constlaw.stress_pol(strain, C_field)
            feat.append(stress_polar / self.stress_scaling)

        if self.config.use_energy:
            strain_energy = compute_strain_energy(strain, stress)
            feat.append(strain_energy / self.energy_scaling)

        # if self.config.use_bc_strain:
        #     eps_avg = self.compute_init_strain(m, None)
        #     feat.append(eps_avg / self.strain_scaling)

        # collect features into a vector
        nn_features = torch.concatenate(feat, dim=1)

        # now return dense feat vec
        return nn_features

    def compute_init_strain(self, micro, init_guess):
        # micro should be
        # get initial strain field
        if init_guess is not None:
            return init_guess
        eps_shape = list(micro.shape)
        # eps should be b * 6 * xyz (since 6 indep strain components)
        Nx = self.config.num_voxels

        return micro.new_ones((eps_shape[0], 6, Nx, Nx, Nx)) * self.eps_bar.reshape(
            1, 6, 1, 1, 1
        )

    def single_iter_simple(self, strain_k, C_field, m):
        """
        Given a stiffness tensor C corresponding to micro m, update the current strain field using the neural net
        """

        z_k = self.encode_micro_strain(strain_k, C_field, m)

        if self.config.add_Green_iter:
            # get moulinec-suquet update
            # NOTE: block gradients flowing through this step (e.g. don't differentiate through MS step)
            # hopefully this stabilizes training
            eps_ms = self.greens_op.forward(strain_k.detach(), C_field)
            z_ms = self.encode_micro_strain(eps_ms, C_field)
            # stack on M-S features after current features
            z_k = torch.concatenate([z_k, z_ms], dim=1)

        # predict new strain perturbation
        strain_kp = self.forward_net(z_k) * self.strain_scaling

        if self.config.use_skip_update:
            strain_kp += strain_k

        strain_kp = self.filter_result(strain_kp)

        return strain_kp

    # def single_iter_fancy(self, m, eh_k):

    #     # print("iter")
    #     # print(m.shape)
    #     # print(h_k.shape)
    #     # print(torch.cuda.memory_summary())

    #     # Split off current strain and encode into features
    #     strain_k, h_k = eh_k[:, :6], eh_k[:, 6:]

    #     # print("Init energy", (strain_k**2).mean().sqrt(), (h_k**2).mean().sqrt())

    #     # don't differentiate through this projection as input to model (only during output)
    #     strain_k = strain_k.detach()

    #     # Now get micro features
    #     z_k = self.encode_micro_strain(strain_k, m)

    #     # Combine information

    #     FNO_input = self.lift_h(h_k) + self.lift_z(z_k)

    #     # now apply model
    #     FNO_output = self.forward_net(FNO_input)

    #     h_kp = self.proj_h(FNO_output)

    #     strain_kp = self.proj_eps(FNO_output) * self.strain_scaling

    #     strain_kp = self.filter_result(strain_kp)

    #     # print("Final energy", (strain_kp**2).mean().sqrt(), (h_kp**2).mean().sqrt())

    #     eh_kp = torch.concatenate([strain_kp, h_kp], dim=1)

    #     return eh_kp

    def forward(self, m):
        # build up shape of latent dim based on micro
        h_shape = list(m.shape)

        C_field = self.constlaw.compute_C_field(m)

        # print("HH", m.shape, C_field.shape)
        # print(C_field[0, 0, 0, :, :, 0])

        # print(C_field.min(), C_field.max())

        # just iterate over strain dim directly
        F = lambda h: self.single_iter_simple(h, C_field, m)
        h0 = self.compute_init_strain(m, None)  # * 0

        # print("init", h0.shape)
        # h_shape[1] = 6 # done automatically

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
        # print(info)
        hstar = sol[-1]

        if self.config.use_fancy_iter:
            # only project out strain if needed
            strain_pred = hstar[:, :6]
        else:
            strain_pred = hstar

        if self.config.return_resid:
            resid = F(hstar.detach()) - hstar
            return strain_pred, resid
        else:
            return strain_pred
