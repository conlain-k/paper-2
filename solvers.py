from constlaw import (
    StrainToStress_2phase,
    compute_strain_from_displacment,
    compute_strain_energy,
    central_diff_3d,
)
from greens_op import GreensOp
from fno import *
from layers import SimpleLayerNet, TinyVNet, ProjectionBlock
import torch
from torchdeq import get_deq, mem_gc
from torch.utils.checkpoint import checkpoint


def make_model(config, input_channels, output_channels):
    # neural net component takes in strain, stress, energy and outputs strain
    if config.use_fno:
        if config.use_deq and config.use_fancy_iter:
            return FNO_Middle(
                # for this one the input and output are the same size
                in_channels=input_channels,
                mid_channels=input_channels,
                out_channels=input_channels,
                **config.fno_args
            )
        else:
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

        if self.config.use_deq:
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

    def enforce_zero_mean(self, x):
        # remove the average value from a field (for each instance, channel)
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)

    def green_iter(self, m, strain):
        eps = self.greens_op(strain, m)
        return eps

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
        # if config.use_bc_strain:
        #     self.num_strain_feats += 6

        if config.use_fancy_iter:
            self.forward_net = make_model(
                config,
                input_channels=config.latent_dim,
                output_channels=None,
            )
            # lift z into h-sized space
            self.lift_z = ProjectionBlock(
                in_channels=self.num_strain_feats, out_channels=config.latent_dim
            )

            # Lift h_k into FNO input space
            self.lift_h = ProjectionBlock(
                in_channels=config.latent_dim, out_channels=config.latent_dim
            )

            # Project h_k+1 from FNO output sapce
            self.proj_h = ProjectionBlock(
                in_channels=config.latent_dim, out_channels=config.latent_dim
            )

            # project out strain field from updated latent repr
            self.proj_eps = ProjectionBlock(
                in_channels=config.latent_dim,
                out_channels=6,
                hidden_channels=config.projection_channels,
            )
        else:
            # map features directly to strain
            self.forward_net = make_model(
                config, input_channels=self.num_strain_feats, output_channels=6
            )

    def encode_micro_strain(self, strain, m):
        # encode microstructure and strain field
        # always use strain as input
        feat = [strain / self.strain_scaling]

        # precompute stress for convenience
        stress = self.constlaw(strain, m)

        if self.config.use_micro:
            # append both phases for convenience
            feat.append(m)

        # build up features
        if self.config.use_stress:
            feat.append(stress / self.stress_scaling)

        if self.config.use_stress_polarization:
            stress_polar = self.constlaw.stress_pol(strain, micro=m)
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
        # get initial strain field
        if init_guess is not None:
            return init_guess
        eps_shape = list(micro.shape)
        # eps should be b * 6 * xyz (since 6 indep strain components)
        eps_shape[1] = 6
        return micro.new_ones(eps_shape) * self.eps_bar.reshape(1, 6, 1, 1, 1)

    def single_iter_simple(self, m, strain_k):
        # iterate over strain directly
        # if self.config.add_Green_iter:
        #     strain_k_tmp = self.green_iter(m, strain_k)

        # return 0 * strain_k + init_strain

        # return init_strain

        z_k = self.encode_micro_strain(strain_k, m)

        # predict new strain perturbation
        strain_kp = self.forward_net(z_k) * self.strain_scaling

        if self.config.use_skip_update:
            strain_kp += strain_k

        # strain_kp = self.filter_result(strain_kp)

        return strain_kp

    def single_iter_fancy(self, m, eh_k):

        # print("iter")
        # print(m.shape)
        # print(h_k.shape)
        # print(torch.cuda.memory_summary())

        # Split off current strain and encode into features
        strain_k, h_k = eh_k[:, :6], eh_k[:, 6:]

        # print("Init energy", (strain_k**2).mean().sqrt(), (h_k**2).mean().sqrt())

        # don't differentiate through this projection as input to model (only during output)
        strain_k = strain_k.detach()

        # Now get micro features
        z_k = self.encode_micro_strain(strain_k, m)

        # Combine information

        FNO_input = self.lift_h(h_k) + self.lift_z(z_k)

        # now apply model
        FNO_output = self.forward_net(FNO_input)

        h_kp = self.proj_h(FNO_output)

        strain_kp = self.proj_eps(FNO_output) * self.strain_scaling

        strain_kp = self.filter_result(strain_kp)

        # print("Final energy", (strain_kp**2).mean().sqrt(), (h_kp**2).mean().sqrt())

        eh_kp = torch.concatenate([strain_kp, h_kp], dim=1)

        return eh_kp

    def forward(self, m):
        # build up shape of latent dim based on micro
        h_shape = list(m.shape)
        # eps should be b * 6 * xyz (since 6 indep strain components)
        if self.config.use_fancy_iter:
            h_shape[1] = self.config.latent_dim
            F = lambda h: self.single_iter_fancy(m, h)
            h0 = m.new_zeros(h_shape)

            eps_0 = self.compute_init_strain(m, None)

            # Concat along channels
            h0 = torch.concatenate([eps_0, h0], dim=1)
        else:
            # just iterate over strain dim directly
            F = lambda h: self.single_iter_simple(m, h)
            h0 = self.compute_init_strain(m, None)
            # h_shape[1] = 6 # done automatically

        # solve FP over h system
        sol, _ = self.deq(
            F,
            h0,
        )
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
