from constlaw import *
from greens_op import GreensOp
from fno import *
from layers import SimpleLayerNet, TinyVNet, ProjectionBlock
import torch
from torchdeq import get_deq, mem_gc
from torch.utils.checkpoint import checkpoint
from helpers import print_activ_map
from math import ceil


def make_localizer(config):
    try:
        localizer_type = eval(config.model_type)
        return localizer_type(config)
    except:
        raise (ValueError(f"Error constructing model type: {config.model_type}"))


class LocalizerBase(torch.nn.Module):
    # base class for localization models
    def __init__(self, config):
        super().__init__()
        # cache config
        self.config = config

        # get FNO # modes
        # fno_args = config.fno_args
        modes = config.fno_args["modes"]

        # if # modes is negative or too big for given data, only keep amount that data can provide
        if -1 in modes:
            full_num_modes = ceil(config.num_voxels / 2)

            # treat -1 as "use all modes", assuming the # voxels was set
            config.fno_args["modes"] = [
                full_num_modes if (m == -1 or m > full_num_modes) else m for m in modes
            ]

        # could swap this with a DeepONet, CNN, etc.
        self.fno = FNO(
            in_channels=self.count_input_channels(),
            out_channels=6,
            **config.fno_args,
        )

        self.register_buffer("eps_bar", torch.zeros(6), persistent=False)

        # empty for now
        self.constlaw = None
        self.greens_op = None

        # default to output strain (6 channels)
        self.output_channels = 6

    def count_input_channels(self):
        channels = 0
        # micro-type features
        if self.config.use_micro:
            channels += 2  # only for n-phase
        if self.config.use_C_flat:
            channels += 21

        if self.config.use_bc_strain:
            channels += 6

        # thermodynamic features (if any)
        if self.config.thermo_feat_args:
            therm_feat = self.config.thermo_feat_args
            therm_channels = 0
            if therm_feat.get("use_strain"):
                channels += 6

            # stress-type features
            if therm_feat.get("use_stress"):
                channels += 6
            if therm_feat.get("use_stress_polarization"):
                channels += 6

            # other thermo features
            if therm_feat.get("use_energy"):
                channels += 1

            if therm_feat.get("add_fft_encoding"):
                # add FFT prediction and features as well
                therm_channels *= 2
            channels += therm_channels

        return channels

    def setConstlaw(self, constlaw, eps_bar_ref):
        # attach this constlaw to model
        self.constlaw = constlaw
        # also compute a corresponding Green's operator for FFT-type methods
        self.greens_op = GreensOp(self.constlaw, self.config.num_voxels)
        # also precompute scalings using sample reference strain
        self.constlaw.compute_scalings(eps_bar_ref)

    def enforce_zero_mean(self, x):
        # remove the average value from a field (for each instance, channel)
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)

    def nonthermo_encode(self, C_field, eps_bar):
        # build non-thermodynamic features (e.g. stiffness, micro, BC strains);
        # call this from child classes to ensure we build the full vector!
        feat = []

        # not always used in thermo-informed model
        if self.config.use_C_flat:
            # append all phases for convenience
            feat.append(flatten_stiffness(C_field))

        # add in BC strains
        if self.config.use_bc_strain:
            # rip out spatial dimensions and batch size
            bs, (nx, ny, nz) = C_field.shape[0], C_field.shape[-3:]
            eps_avg = C_field.new_ones((bs, 6, nx, ny, nz)) * eps_bar
            feat.append(eps_avg)

        return feat

    def thermo_encode(self, C_field, eps_bar, strain):
        # build non-thermodynamic features first
        feat = self.nonthermo_encode(C_field, eps_bar)

        thermo_args = self.config.thermo_feat_args

        # do we use current strain as feature?
        if thermo_args.get("use_strain"):
            feat.append(strain)

        # lazy-compute stres field if needed
        stress = None

        # build up features
        if thermo_args.get("use_stress"):
            stress = self.constlaw(C_field, strain)
            feat.append(stress)

        if thermo_args.get("use_energy"):
            if stress is None:
                stress = self.constlaw(C_field, strain)
            strain_energy = compute_strain_energy(strain, stress)
            feat.append(strain_energy)

        if self.config.thermo_feat_args.get("use_stress_polarization"):
            # C_field = self.lazy_m_to_C(m, C_field)
            stress_polar = self.constlaw.stress_pol(strain, C_field, scaled=True)
            # negate stress polarization to get positive-ish values
            feat.append(stress_polar)

        # collect features into a vector
        nn_features = torch.concatenate(feat, dim=1)

        # now return dense feat vec
        return nn_features

    def _encode(self, C_field, eps_bar):
        # always normalize stiffnesses before putting into NN
        C_field = C_field / self.constlaw.stiffness_scaling
        eps_bar = eps_bar.reshape(-1, 6, 1, 1, 1) / self.constlaw.strain_scaling
        return torch.concatenate(self.nonthermo_encode(C_field, eps_bar), dim=1)

    def _postprocess(self, x, eps_bar, scale):
        if scale:
            x = x * self.constlaw.strain_scaling

        # do we enforce mean is correct directly?
        if self.config.enforce_mean:
            x = self.enforce_zero_mean(x) + eps_bar.reshape(-1, 6, 1, 1, 1)
        # alternatively, do we add in bc vals but not enforce zero mean?
        elif self.config.add_bcs_to_iter:
            x = x + eps_bar.reshape(-1, 6, 1, 1, 1)

        return x


class FeedForwardLocalizer(LocalizerBase):
    # base class for localization models
    def __init__(self, config):
        super().__init__(config)

    def forward(self, C_field, eps_bar):
        # rescale stiffness and encode
        x = self._encode(C_field, eps_bar)

        # use FNO directly
        x = self.fno(x)

        # apply NN to encoded input
        x = self._postprocess(x, eps_bar, scale=self.config.scale_output)

        return x


class IFNOLocalizer(LocalizerBase):
    def __init__(self, config):
        super().__init__(config)
        self.num_passes = config.num_ifno_iters
        # quasi-step-size used in IFNO paper
        self.dt = 1.0 / self.num_passes

    # IFNO predictions
    def forward(self, C_field, eps_bar):
        x = self._encode(C_field, eps_bar)

        # use FNO lift and proj regularly
        x = self.fno.lift(x)

        # apply middle N-times (weight-tied)
        for _ in range(self.num_passes):
            x = x + self.dt * self.fno.middle(x)

        x = self.fno.proj(x)

        # apply NN to encoded input
        x = self._postprocess(x, eps_bar, scale=self.config.scale_output)

        return x


class FNODEQLocalizer(LocalizerBase):
    def __init__(self, config):
        super().__init__(config)
        self.deq = get_deq(config.deq_args)

    def _sample_num_iters(self):
        # only used if we randomize # iters
        if self.config.deq_randomize_max and self.training:
            min = self.config.deq_min_iter
            avg = self.config.deq_args.get("f_max_iter")
            # average # iters given by config (makes it easier to evaluate later)
            it_max = 2 * avg - min

            max_iters = torch.randint(min, it_max, (1,))
            iter_arg = {"f_max_iter": max_iters.item()}
        else:
            iter_arg = None

        return iter_arg

    def _initStrain(self, x0):
        # start w/ zeros in DEQ mode
        return x0.new_zeros(x0.shape)

    def _encodeInput(self, C_field, eps_bar):
        x = self._encode(C_field, eps_bar)
        x = self.fno.lift(x)
        return x

    def F(self, h, x):
        return x + self.fno.middle(h)

    # a single feedforward
    def forward(self, C_field, eps_bar):
        # pre-encoding step before DEQ (what gets input-injected?)
        inputs_encoded = self._encodeInput(C_field, eps_bar)

        # fixed-point equation h = F(h, x)
        # bind in current x
        F = lambda h: self.F(h, inputs_encoded)

        # start w/ all zeros initial latent guess unless we decided otherwise
        h0 = self._initStrain(inputs_encoded)

        # solve FP over h system
        hstar, _ = self.deq(F, h0, solver_kwargs=self._sample_num_iters())

        # torchdeq is tricky and returns a tuple of trajectories, so we need to pull out last one
        h_last = hstar[-1]

        x_decoded = self._decodeOutput(h_last, eps_bar)

        if self.config.return_resid:
            resid = F(h_last) - h_last
            return x_decoded, resid

        return x_decoded

    def _compute_trajectory(self, C_field, eps_bar, h0=None, num_iters=32):
        """Compute trajectory of solution (projected into output space) across iterations"""
        # pre-encoding step before DEQ (what gets input-injected?)
        inputs_encoded = self._encodeInput(C_field, eps_bar)

        # fixed-point equation h = F(h, x)
        # bind in current x
        F = lambda h: self.F(h, inputs_encoded)

        # start w/ all zeros initial latent guess unless we decided otherwise
        h0 = self._initStrain(inputs_encoded)

        # indexing starts at 3 in torchdeq (1-based, plus ignores first two iterates)
        indexing = torch.arange(3, num_iters + 3).tolist()
        with torch.inference_mode():
            # call into deq solver directly
            ret = self.deq.f_solver(
                F, x0=h0, indexing=indexing, max_iter=num_iters + 3, tol=0
            )

        _, traj, info = ret

        print("traj has length", len(traj))
        print(info)

        traj = [self._decodeOutput(state, eps_bar) for state in traj]

        return traj


class ThermINOLocalizer(FNODEQLocalizer):
    # thermo-informed implicit neural operator deq
    def __init__(self, config):
        super().__init__(config)
        self.deq = get_deq(config.deq_args)

    def _initStrain(self, Ceps):
        C_field, eps_bar = Ceps
        bs, (nx, ny, nz) = C_field.shape[0], C_field.shape[-3:]
        # start w/ avg strain for initial guess
        return C_field.new_ones(bs, 6, nx, ny, nz) * eps_bar

    def _encodeInput(self, C_field, eps_bar):
        # do scaling here since we aren't calling up to parent class _encode()
        C_field = C_field / self.constlaw.stiffness_scaling
        eps_bar = eps_bar.reshape(-1, 6, 1, 1, 1) / self.constlaw.strain_scaling

        return C_field, eps_bar

    def _decodeOutput(self, x, eps_bar):
        # do scaling here since we aren't calling up to parent class _postprocess()
        return x * self.constlaw.strain_scaling

    def F(self, h, Ceps):
        (C_field, eps_bar) = Ceps
        # here x is a (C_field, eps_bar) tuple and h is a candidate strain field

        if self.config.hybrid_args.get("use_fft_pre_iter"):
            # overwrite current iterate with FFT update
            h = self.greens_op.forward(h, C_field)

        # first compute thermo encodings (z is an encoded latent representation)
        z = self.thermo_encode(C_field, eps_bar, h)

        # add information from green's op
        if self.config.hybrid_args.get("add_fft_encoding"):
            # don't do both at once
            assert not self.config.hybrid_args.get("use_fft_pre_iter")
            # get moulinec-suquet update
            eps_ms = self.greens_op.forward(h, C_field)

            # don't differentiate through FFT update to stabilize / reduce memory overhead
            eps_ms = eps_ms.detach()
            z_ms = self.thermo_encode(C_field, eps_bar, eps_ms)

            # stack on M-S features after current features
            z = torch.concatenate([z, z_ms], dim=1)

        # now apply regular FNO operations
        z = self.fno.lift(z)
        z = self.fno.middle(z)
        # FNO outputs a strain field (no guarantees on avg strain / bcs)
        h = self.fno.proj(z)

        # now fix averages to get BCs right
        h = self._postprocess(h, eps_bar, scale=False)

        # do green's iteration on filtered output (guaranteed to get avg right anyways)
        if self.config.hybrid_args.get("use_fft_post_iter"):
            # overwrite current iterate with FFT update
            h = self.greens_op.forward(h, C_field)

        assert not torch.isnan(h).any()

        return h


# class Localizer_Iterative(LocalizerBase):
#     def __init__(self, config):
#         super().__init__(config)
#         # deep-EQ solver
#         self.deq = get_deq(config.deq_args)

#         input_channels = self.count_input_channels()

#     def single_iter_simple(self, strain_k, C_field, m):
#         """
#         Given a stiffness tensor C_field corresponding to micro m, update the current strain field using the neural net
#         """

#         # do green's iteration, then use that as input to network
#         if self.config.use_fft_pre_iter:
#             # overwrite current iterate with FFT update
#             strain_k = self.greens_op.forward(strain_k, C_field)

#         z_k = self.encode_micro_strain(m, C_field, strain_k)

#         # add information from green's op
#         if self.config.add_fft_encoding:
#             # don't do both at once
#             assert not self.config.use_fft_pre_iter
#             # get moulinec-suquet update
#             eps_ms = self.greens_op.forward(strain_k, C_field)

#             # don't differentiate through FFT update to stabilize / reduce memory overhead
#             eps_ms = eps_ms.detach()
#             z_ms = self.encode_micro_strain(m, C_field, eps_ms)

#             # stack on M-S features after current features
#             z_k = torch.concatenate([z_k, z_ms], dim=1)

#         # predict new strain perturbation
#         strain_kp = self.forward_net(z_k)

#         strain_kp = self.filter_result(strain_kp)
#         # do green's iteration on output of network
#         if self.config.use_fft_post_iter:
#             # overwrite current iterate with FFT update
#             strain_kp = self.greens_op.forward(strain_kp, C_field)

#         assert not torch.isnan(strain_kp).any()

#         return strain_kp

#     def _compute_trajectory(self, m, num_iters=32):
#         """hook into deq implementation to return full solver trajectory"""
#         C_field = self.lazy_m_to_C(m)

#         # just iterate over strain dim directly
#         F = lambda h: self.single_iter_simple(h, C_field, m)
#         h0 = self.compute_init_scaled_strain(m)

#         # indexing starts at 3 in torchdeq (1-based, plus ignores first two iterates)
#         indexing = torch.arange(3, num_iters + 3).tolist()
#         with torch.inference_mode():
#             # call into deq solver directly
#             ret = self.deq.f_solver(
#                 F, x0=h0, indexing=indexing, max_iter=num_iters + 3, tol=0
#             )

#         z, traj, info = ret

#         print("traj has length", len(traj))
#         print(info)

#         out_scale = self.constlaw.strain_scaling if self.config.scale_output else 1

#         traj = [s * out_scale for s in traj]

#         return traj

#     def forward(self, m):

#         C_field = self.lazy_m_to_C(m)

#         # just iterate over strain dim directly
#         F = lambda h: self.single_iter_simple(h, C_field, m)
#         h0 = self.compute_init_scaled_strain(m)

#         # randomize # iters during train time
#         if self.config.deq_randomize_max and self.training:
#             min = self.config.deq_min_iter
#             avg = self.config.deq_args.get("f_max_iter")
#             # average # iters given by config (makes it easier to evaluate later)
#             it_max = 2 * avg - min

#             max_iters = torch.randint(min, it_max, (1,))
#             iter_arg = {"f_max_iter": max_iters.item()}
#         else:
#             iter_arg = None

#         # solve FP over h system
#         sol, _ = self.deq(F, h0, solver_kwargs=iter_arg)

#         hstar = sol[-1]

#         out_scale = self.constlaw.strain_scaling if self.config.scale_output else 1

#         # now project out output and rescale appropriately
#         strain_pred = hstar * out_scale

#         # return model and residual
#         if self.config.return_resid:
#             resid = (F(hstar) - hstar) * out_scale
#             return strain_pred, resid

#         else:
#             return strain_pred
