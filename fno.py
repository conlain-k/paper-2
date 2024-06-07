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
                    **kwargs,
                )
            )

        self.blocks = torch.nn.ModuleList(blocks)

        self.input_norm = torch.nn.GroupNorm(1, in_channels)

    def forward(self, x):

        x = self.input_norm(x)
        x = self.lift(x)

        for block in self.blocks:
            # apply FNO blocks sequentially
            x = block(x)

        # also normalize output of FNO chain
        # x = self.input_norm(x)
        # now do two projection steps, with an activation in the middle
        x = self.proj(x)

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
        resid_conn,
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
        )

        # local channel-wise filter
        self.filt = torch.nn.Conv3d(
            mid_channels, mid_channels, kernel_size=1, bias=True
        )

        if use_weight_norm:
            self.filt = weight_norm(self.filt)

        self.activ = get_activ(activ_type, mid_channels)

        if normalize:
            self.norm = torch.nn.GroupNorm(1, mid_channels, affine=True)

    # just the middle bit of an FNO
    def forward(self, x):
        # residual outside normalization
        if self.resid_conn:
            x0 = x

        x = self.norm(x) if self.normalize else x
        x = self.activ(self.conv(x) + self.filt(x))

        if self.resid_conn:
            # residual connection
            x = x + x0

        return x
