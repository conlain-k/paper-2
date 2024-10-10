import torch
import torch.nn.functional as F

import fourier_conv
from torch.nn.utils.parametrizations import weight_norm

from layers import ProjectionBlock, get_activ


class FNO(torch.nn.Module):
    """Base class for FNO-type models. Could change lift/middle/proj blocks to anything (e.g. DeepONet, etc.)"""

    def __init__(
        self,
        in_channels,
        out_channels,
        latent_channels=24,
        final_projection_channels=128,  # defaults to bigger of latent_channels, out_channels
        activ_type="gelu",
        use_weight_norm=False,
        normalize_inputs=False,
        use_mlp_lifting=False,
        modes=(10, 10),
        **kwargs,
    ):
        super().__init__()
        # build lifting op
        if use_mlp_lifting:
            lift = ProjectionBlock(
                in_channels,
                latent_channels,
                final_projection_channels,
                activ_type,
                use_weight_norm,
            )
        else:
            lift = torch.nn.Conv3d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=latent_channels,
            )

        if normalize_inputs:
            input_norm = torch.nn.GroupNorm(num_groups=1, num_channels=in_channels)

            # normalize before lifting
            self.lift = torch.nn.Sequential(input_norm, lift)
        else:
            self.lift = lift

        # build projection op
        self.proj = ProjectionBlock(
            latent_channels,
            out_channels,
            hidden_channels=final_projection_channels,
            activ_type=activ_type,
            use_weight_norm=False,
        )

        blocks = []
        for mm in modes:
            blocks.append(
                FNO_Block(
                    activ_type=activ_type,
                    use_weight_norm=use_weight_norm,
                    latent_channels=latent_channels,
                    num_modes=mm,
                    **kwargs,
                )
            )

        self.middle = FNO_Middle(
            modes,
            activ_type=activ_type,
            use_weight_norm=use_weight_norm,
            latent_channels=latent_channels,
            **kwargs,
        )

    def forward(self, x):
        # a single feedforward
        # lift into latent space
        x = self.lift(x)

        # apply filter
        x = self.middle(x)

        # now project onto strain field
        x = self.proj(x)

        return x


class FNO_Middle(torch.nn.Module):
    def __init__(self, modes, *args, **kwargs):
        super().__init__()
        blocks = []
        for mm in modes:
            blocks.append(
                FNO_Block(
                    num_modes=mm,
                    **kwargs,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x, input_inj=None):
        # pass input injection onwards to all blocks
        for block in self.blocks:
            x = block(x, input_inj)
        return x


class FNO_Block(torch.nn.Module):
    def __init__(
        self,
        activ_type,
        use_weight_norm,
        latent_channels,
        num_modes,
        normalize,
        init_weight_scale,
        resid_conn,
        **kwargs,
    ):
        super().__init__()

        self.resid_conn = resid_conn
        self.normalize = normalize

        # spectral convolution
        self.conv = fourier_conv.SpectralConv3d(
            latent_channels,
            latent_channels,
            num_modes,
            num_modes,
            num_modes,
            scale_fac=init_weight_scale,
        )

        # local channel-wise filter
        self.filt = torch.nn.Conv3d(
            latent_channels, latent_channels, kernel_size=1, bias=True
        )

        if use_weight_norm:
            self.filt = weight_norm(self.filt)

        self.activ = get_activ(activ_type, latent_channels)

        if normalize:
            self.norm = torch.nn.GroupNorm(1, latent_channels)

    # just the middle bit of an FNO
    def forward(self, x, input_inj=None):
        # residual outside normalization
        if self.resid_conn:
            x0 = x

        # normalize inputs for stability
        x = self.norm(x) if self.normalize else x

        # apply fourier and spatial filters
        x = self.conv(x) + self.filt(x)

        if input_inj is not None:
            # allow for direct input injection before activation (follows FNO-deq code, not paper)
            # c.f. https://github.com/risteskilab/deq-neural-operators/blob/8bf24bd244e17b54903b3f7f5952399ab56ccfda/models/fourier_2d_deq.py#L67
            x = x + input_inj

        x = self.activ(x)

        if self.resid_conn:
            # residual connection
            x = x + x0

        return x
