from constlaw import *
from greens_op import GreensOp
from fno import *
from layers import SimpleLayerNet, TinyVNet, ProjectionBlock
import torch
from torchdeq import get_deq, mem_gc
from torch.utils.checkpoint import checkpoint
from helpers import print_activ_map
from math import ceil


def make_localizer(config, constlaw):
    try:
        localizer_type = eval(config.model_type)
        return localizer_type(config, constlaw)
    except:
        raise (
            ValueError(
                f"Error constructing model type: {config.model_type} with constlaw {constlaw}"
            )
        )


class LocalizerBase(torch.nn.Module):
    def __init__(self, config, constlaw):
        super().__init__()
        # cache config
        self.config = config
        if config.model_type is None:
            # get child class name and store it
            config.model_type = str(type(self).__name__)

        # store constlaw
        self.overrideConstlaw(constlaw)

        # these scalings should be stored persistently for trained models and set before training via compute_scalings
        self.register_buffer("stiffness_scaling", torch.tensor(1.0))
        self.register_buffer("strain_scaling", torch.tensor(1.0))
        self.register_buffer("stress_scaling", torch.tensor(1.0))
        self.register_buffer("energy_scaling", torch.tensor(1.0))

    def overrideConstlaw(self, constlaw):
        # allows overriding a const law if we got it from a checkpoint
        # attach this constlaw to model
        self.constlaw = constlaw
        if constlaw is not None:
            # also compute a corresponding Green's operator for FFT-type methods
            vox = self.config.num_voxels
            if isinstance(self, LocalizerFFT):
                vox *= self.config.greens_upsample
            self.greens_op = GreensOp(self.constlaw, vox)

    def compute_scalings(self, bc_strains):
        # compute scalings for stiffness and strain, then downstream scalings
        # be careful overriding this on an already trained model
        frob = lambda x: (x**2).sum().sqrt()
        # print("Computing scalings!")
        self.stiffness_scaling = frob(self.constlaw.C_ref)
        self.strain_scaling = frob(torch.as_tensor(bc_strains))

        # compute other scalings downstream of this one
        self.stress_scaling = self.stiffness_scaling * self.strain_scaling
        self.energy_scaling = self.stress_scaling * self.strain_scaling

    def scale_strain(self, strain):
        return strain / self.strain_scaling

    def unscale_strain(self, strain):
        return strain * self.strain_scaling

    def scale_stress(self, stress):
        return stress / self.stress_scaling

    def unscale_stress(self, stress):
        return stress * self.stress_scaling

    def scale_stiffness(self, stiffness):
        return stiffness / self.stiffness_scaling

    def unscale_stiffness(self, stiffness):
        return stiffness * self.stiffness_scaling


class LocalizerFFT(LocalizerBase):
    # FFT-based localizer
    def __init__(self, config, constlaw, num_iters=16):
        super().__init__(config, constlaw)

        self.num_iters = num_iters

    def _run_iters(
        self, C_field, bc_strains, eps_0=None, num_iters=None, return_traj=False
    ):
        # get initial conditions
        if eps_0 is None:
            bs, (nx, ny, nz) = C_field.shape[0], C_field.shape[-3:]
            eps_0 = bc_strains.expand(bs, 6, nx, ny, nz)
        eps = eps_0

        if self.config.greens_upsample > 1:
            eps = upsample_field(eps, self.config.greens_upsample)
            C_field = upsample_field(C_field, self.config.greens_upsample)

        if num_iters is None:
            num_iters = self.num_iters
        # do we return trajectory?
        if return_traj:
            traj = [None] * num_iters

        # print(eps.shape, C_field.shape, self.greens_op.G_freq.shape)

        # run a number of FFT iterations
        for i in range(num_iters):
            eps = self.greens_op.forward(eps, C_field, use_polar=True)

            if return_traj:
                if self.config.greens_upsample > 1:
                    ec = average_field(eps.detach(), self.config.greens_upsample)
                else:
                    ec = eps.detach()
                traj[i] = ec

        if self.config.greens_upsample > 1:
            eps = average_field(eps.detach(), self.config.greens_upsample)

        if return_traj:
            return eps, traj

        return eps

    # compute trajectory in output space across IFNO iterations
    def _compute_trajectory(
        self,
        C_field,
        bc_strains,
        h0=None,
        num_iters=32,
        return_latent=False,
        return_latent_resids=False,
    ):
        traj = self._run_iters(C_field, bc_strains, h0, num_iters, return_traj=True)[1]
        if return_latent:
            if return_latent_resids:
                return traj, traj, None
            return traj, traj
        return traj

    def forward(self, C_field, bc_strains):

        return self._run_iters(
            C_field, bc_strains, None, self.num_iters, return_traj=False
        )


class LocalizerFNOBase(LocalizerBase):
    # base class for FNO localization models
    def __init__(self, config, constlaw):
        super().__init__(config, constlaw)

        # get FNO # modes
        # fno_args = config.fno_args
        modes = self.config.fno_args["modes"]

        # if # modes is negative or too big for given data, only keep amount that data can provide
        if -1 in modes:
            full_num_modes = ceil(self.config.num_voxels / 2)

            # treat -1 as "use all modes", assuming the # voxels was set
            self.config.fno_args["modes"] = [
                full_num_modes if (m == -1 or m > full_num_modes) else m for m in modes
            ]

        # could swap this with a DeepONet, CNN, etc.
        self.fno = FNO(
            in_channels=self.count_input_channels(),
            out_channels=6,
            **self.config.fno_args,
        )

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
                therm_channels += 6

            # stress-type features
            if therm_feat.get("use_stress"):
                therm_channels += 6
            if therm_feat.get("use_stress_polarization"):
                therm_channels += 6

            # other thermo features
            if therm_feat.get("use_energy"):
                therm_channels += 1

            if self.config.hybrid_args.get("add_fft_encoding"):
                # add FFT prediction and features as well
                therm_channels *= 2
            channels += therm_channels

        return channels

    def enforce_zero_mean(self, x):
        # remove the average value from a field (for each instance, channel)
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)

    def nonthermo_encode(self, C_field, bc_strains):
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
            eps_avg = C_field.new_ones((bs, 6, nx, ny, nz)) * bc_strains
            feat.append(eps_avg)

        return feat

    def thermo_encode(self, C_field, bc_strains, strain, add_nontherm=True):
        # build non-thermodynamic features first
        if add_nontherm:
            feat = self.nonthermo_encode(C_field, bc_strains)
        else:
            feat = []

        thermo_args = self.config.thermo_feat_args

        # do we use current strain as feature?
        if thermo_args.get("use_strain"):
            feat.append(strain)

        # lazy-compute stres field if needed
        stress = None

        # build up features
        if thermo_args.get("use_stress"):
            stress = strain_to_stress(C_field, strain)
            feat.append(stress)

        if thermo_args.get("use_energy"):
            if stress is None:
                stress = strain_to_stress(C_field, strain)
            # print(strain.shape, stress.shape)
            strain_energy = compute_strain_energy(strain, stress)
            feat.append(strain_energy)

        if self.config.thermo_feat_args.get("use_stress_polarization"):
            # C_field = self.lazy_m_to_C(m, C_field)
            stress_polar = self.constlaw.stress_pol(
                strain, C_field, ref_scaling=self.stiffness_scaling
            )
            # negate stress polarization to get positive-ish values
            feat.append(stress_polar)

        # collect features into a vector
        nn_features = torch.concatenate(feat, dim=1)

        # now return dense feat vec
        return nn_features

    def _encode(self, C_field, bc_strains):
        # always normalize stiffnesses before putting into NN
        C_field = self.scale_stiffness(C_field)
        bc_strains = self.scale_strain(bc_strains)
        return torch.concatenate(self.nonthermo_encode(C_field, bc_strains), dim=1)

    def _postprocess(self, x, bc_strains, scale):
        if scale:
            x = self.unscale_strain(x)

        # do we enforce mean is correct directly?
        if self.config.enforce_mean:
            x = self.enforce_zero_mean(x) + bc_strains.reshape(-1, 6, 1, 1, 1)
        # alternatively, do we add in bc vals but not enforce zero mean?
        elif self.config.add_bcs_to_iter:
            x = x + bc_strains.reshape(-1, 6, 1, 1, 1)

        return x


class FeedForwardLocalizer(LocalizerFNOBase):
    # base class for localization models
    def __init__(self, config, constlaw):
        super().__init__(config, constlaw)

    def forward(self, C_field, bc_strains):
        # rescale stiffness and encode
        x = self._encode(C_field, bc_strains)

        # use FNO lift and proj regularly
        x = self.fno.lift(x)

        # apply middle with input injection
        x = self.fno.middle(x, input_inj=x)

        x = self.fno.proj(x)

        # apply NN to encoded input
        x = self._postprocess(x, bc_strains, scale=self.config.scale_output)

        return x


class IFNOLocalizer(LocalizerFNOBase):
    def __init__(self, config, constlaw):
        super().__init__(config, constlaw)
        self.num_passes = config.num_ifno_iters
        # quasi-step-size used in IFNO paper
        self.dt = 1.0 / self.num_passes

    # IFNO predictions
    def forward(self, C_field, bc_strains):
        x = self._encode(C_field, bc_strains)

        # use FNO lift and proj regularly
        x = self.fno.lift(x)

        # apply middle N-times (weight-tied)
        for _ in range(self.num_passes):
            x = x + self.dt * self.fno.middle(x)

        x = self.fno.proj(x)

        # apply NN to encoded input
        x = self._postprocess(x, bc_strains, scale=self.config.scale_output)

        return x

    # compute trajectory in output space across IFNO iterations
    def _compute_trajectory(
        self,
        C_field,
        bc_strains,
        h0=None,
        num_iters=32,
        return_latent=False,
        return_latent_resids=False,
    ):
        """Compute trajectory of solution (projected into output space) across iterations"""
        with torch.inference_mode():
            # pre-encoding step before IFNO
            x = self.fno.lift(self._encode(C_field, bc_strains))

            traj = []
            traj_proj = []
            # run thru IFNO loop and save intermediate iterates, then project into output space
            for _ in range(num_iters):
                # apply IFNO update
                x = x + self.dt * self.fno.middle(x)

                # un-projected trajectory
                traj.append(x)
                # postprocess output for later return
                traj_proj.append(
                    self._postprocess(
                        self.fno.proj(x), bc_strains, scale=self.config.scale_output
                    )
                )

            if return_latent:
                if return_latent_resids:
                    return traj_proj, traj, None
                return traj_proj, traj

            return traj_proj


class FNODEQLocalizer(LocalizerFNOBase):
    def __init__(self, config, constlaw):
        super().__init__(config, constlaw)

        self.reinitDEQ()

    def reinitDEQ(self):
        self.deq = get_deq(self.config.deq_args)

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

    def _initState(self, x0):
        # start w/ zeros in DEQ mode
        return x0.new_zeros(x0.shape)

    def _encodeInput(self, C_field, bc_strains):
        x = self._encode(C_field, bc_strains)
        x = self.fno.lift(x)
        return x

    def _decodeOutput(self, x, bc_strains):
        # apply fno projection
        x = self.fno.proj(x)

        # apply NN to encoded input
        return self._postprocess(x, bc_strains, scale=self.config.scale_output)

    def F(self, h, x):
        # print("iter mem\n", torch.cuda.memory_summary())

        return self.fno.middle(h, input_inj=x)

    # a single feedforward
    def forward(self, C_field, bc_strains):
        # pre-encoding step before DEQ (what gets input-injected?)
        inputs_encoded = self._encodeInput(C_field, bc_strains)

        # fixed-point equation h = F(h, x)
        # bind in current x
        F = lambda h: self.F(h, inputs_encoded)

        # start w/ all zeros initial latent guess unless we decided otherwise
        h0 = self._initState(inputs_encoded).detach()

        # solve FP over h system
        hstar, _ = self.deq(F, h0, z_kwargs=self._sample_num_iters())

        # torchdeq is tricky and returns a tuple of trajectories, so we need to pull out last one
        h_last = hstar[-1]

        x_decoded = self._decodeOutput(h_last, bc_strains)

        if self.config.add_resid_loss:
            return x_decoded, h_last

        return x_decoded

    def _compute_trajectory(
        self,
        C_field,
        bc_strains,
        h0=None,
        num_iters=32,
        return_latent=False,
        return_latent_resids=False,
    ):
        """Compute trajectory of solution (projected into output space) across iterations"""
        with torch.inference_mode():
            # pre-encoding step before DEQ (what gets input-injected?)
            inputs_encoded = self._encodeInput(C_field, bc_strains)

            # fixed-point equation h = F(h, x)
            # bind in current x
            F = lambda h: self.F(h, inputs_encoded)

            # start w/ all zeros initial latent guess unless we decided otherwise
            h0 = self._initState(inputs_encoded)

            # indexing starts at 3 in torchdeq (1-based, plus ignores first two iterates)
            indexing = torch.arange(3, num_iters + 3).tolist()
            # call into deq solver directly
            ret = self.deq.f_solver(
                F, x0=h0, indexing=indexing, max_iter=num_iters + 3, tol=0
            )

            _, traj, _ = ret

            # print("traj has length", len(traj))
            # print(info)

            traj_proj = [self._decodeOutput(state, bc_strains) for state in traj]

            if return_latent:
                # also get deq residuals in latent space
                if return_latent_resids:
                    traj_resids = [F(hi) - hi for hi in traj]
                    return traj_proj, traj, traj_resids
                return traj_proj, traj

            return traj_proj


class TherINOLocalizer(FNODEQLocalizer):
    # thermo-informed implicit neural operator deq
    def __init__(self, config, constlaw):
        super().__init__(config, constlaw)

        # parent initializes deq

    def reinitDEQ(self):
        self.deq = get_deq(self.config.deq_args)

    def _initState(self, Ceps):
        C_field, bc_strains = Ceps
        bs, (nx, ny, nz) = C_field.shape[0], C_field.shape[-3:]
        if self.config.therino_init_zero:
            # start w/ zero strain for initial guess
            return C_field.new_zeros(bs, 6, nx, ny, nz)
        else:
            # start w/ avg strain for initial guess
            return bc_strains.expand(bs, 6, nx, ny, nz)

    def _encodeInput(self, C_field, bc_strains):
        # do scaling here since we aren't calling up to parent class _encode()
        C_field = self.scale_stiffness(C_field)
        bc_strains = self.scale_strain(bc_strains)

        return C_field, bc_strains

    def _decodeOutput(self, x, bc_strains):
        # do scaling here since we aren't calling up to parent class _postprocess()
        return self.unscale_strain(x)

    def F(self, h, Ceps):
        # here x is a (C_field, bc_strains) tuple and h is a candidate strain field
        (C_field, bc_strains) = Ceps

        if self.config.hybrid_args.get("use_fft_pre_iter"):
            # overwrite current iterate with FFT update
            h = self.greens_op.forward(h, C_field)

        # first compute thermo encodings (z is an encoded latent representation)
        z = self.thermo_encode(C_field, bc_strains, h)

        # add information from green's op
        if self.config.hybrid_args.get("add_fft_encoding"):
            # don't do both at once
            assert not self.config.hybrid_args.get("use_fft_pre_iter")
            # get moulinec-suquet update
            eps_ms = self.greens_op.forward(h, C_field)

            # don't differentiate through FFT update to stabilize / reduce memory overhead
            eps_ms = eps_ms.detach()
            z_ms = self.thermo_encode(C_field, bc_strains, eps_ms, add_nontherm=False)

            # print("zshape", z.shape, z_ms.shape)
            # stack on M-S features after current features
            z = torch.concatenate([z, z_ms], dim=1)

        # now apply regular FNO operations
        z = self.fno.lift(z)
        z0 = z
        z = self.fno.middle(z, input_inj=z0)
        # FNO outputs a strain field (no guarantees on avg strain / bcs)
        h = self.fno.proj(z)

        # now fix averages to get BCs right
        h = self._postprocess(h, bc_strains, scale=False)

        # print(h.mean((0, -3, -2, -1)))

        # do green's iteration on filtered output (guaranteed to get avg right anyways)
        if self.config.hybrid_args.get("use_fft_post_iter"):
            # overwrite current iterate with FFT update
            h = self.greens_op.forward(h, C_field)

        assert not torch.isnan(h).any()

        return h
