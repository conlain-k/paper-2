import torch
import torch.nn.functional as F

import fourier_conv
from torch.nn.utils.parametrizations import weight_norm

from layers import ProjectionBlock, get_activ
from helpers import print_activ_map


class FNO(torch.nn.Module):
    """Fourier Neural Operator for localization"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=24,
        final_projection_channels=128,  # defaults to bigger of mid_channels, out_channels
        activ_type="gelu",
        use_weight_norm=False,
        modes=(10, 10),
        use_MLP=False,
        use_injection=False,
        **kwargs,
    ):
        super().__init__()
        # lifting and projection blocks
        self.lift = torch.nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=True)

        self.proj = ProjectionBlock(
            mid_channels,
            out_channels,
            hidden_channels=final_projection_channels,
            activ_type=activ_type,
            use_weight_norm=False,
            final_bias=False,
        )

        blocks = []
        for mm in modes:
            blocks.append(
                FNO_Block(
                    activ_type=activ_type,
                    use_weight_norm=use_weight_norm,
                    mid_channels=mid_channels,
                    num_modes=mm,
                    use_MLP=use_MLP,
                    **kwargs,
                )
            )

        self.blocks = torch.nn.ModuleList(blocks)

        self.use_injection = use_injection

        # if normalize:
        # group size = 1 (a.k.a. easy layernorm)
        # self.norm = torch.nn.GroupNorm(1, mid_channels)
        # self.norm = torch.nn.GroupNorm(1, mid_channels, affine=False)
        # normalize all inputs channel-wise to fix contrast issues, etc.
        # self.input_norm = torch.nn.InstanceNorm3d(
        #     in_channels, track_running_stats=False, affine=False
        # )
        self.input_norm = torch.nn.GroupNorm(1, in_channels)

    def forward(self, x):

        x = self.input_norm(x)
        x = self.lift(x)

        if self.use_injection:
            inj = x
        else:
            # don't add anything
            inj = 0.0

        for block in self.blocks:
            # apply FNO blocks sequentially
            x = block(x, inj)

        # also normalize output of FNO chain
        # x = self.input_norm(x)
        # now do two projection steps, with an activation in the middle
        x = self.proj(x) * 2.0

        return x


class FNO_Block(torch.nn.Module):
    def __init__(
        self,
        activ_type,
        use_weight_norm,
        mid_channels,
        num_modes,
        normalize,
        init_weight_scale,
        use_fourier_bias,
        resid_conn,
        use_MLP,
        **kwargs,
    ):
        super().__init__()

        self.resid_conn = resid_conn
        self.normalize = normalize

        # spectral convolution
        self.conv = fourier_conv.SpectralConv3d(
            mid_channels,
            mid_channels,
            num_modes,
            num_modes,
            num_modes,
            scale_fac=init_weight_scale,
            use_bias=use_fourier_bias,
        )

        # local channel-wise filter
        self.filt = torch.nn.Conv3d(
            mid_channels, mid_channels, kernel_size=1, bias=True
        )

        if use_weight_norm:
            self.filt = weight_norm(self.filt)

        self.activ = get_activ(activ_type, mid_channels)

        if normalize:
            # group size = 1 (a.k.a. easy layernorm)
            # self.norm = torch.nn.GroupNorm(1, mid_channels)
            self.norm = torch.nn.GroupNorm(1, mid_channels, affine=False)
            # self.norm = torch.nn.InstanceNorm3d(mid_channels)

        # add another local FC layer before activation
        self.use_MLP = use_MLP
        if self.use_MLP:
            self.mlp_layer = ProjectionBlock(
                mid_channels,
                mid_channels,
                activ_type=activ_type,
                use_weight_norm=use_weight_norm,
                final_bias=False,
                normalize=True,
            )

    # just the middle bit of an FNO
    def forward(self, x, injection=0):
        # residual outside normalization
        if self.resid_conn:
            x0 = x

        x = self.norm(x) if self.normalize else x
        x = self.activ(self.conv(x) + self.filt(x))

        if self.resid_conn:
            # residual connection
            x = x + x0

        return x
