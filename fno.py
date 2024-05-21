import torch
import torch.nn.functional as F

import fourier_conv
from torch.nn.utils.parametrizations import weight_norm

from layers import ProjectionBlock, get_activ


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
        use_MLP=True,
        **kwargs,
    ):
        super().__init__()
        # lifting and projection blocks
        self.lift = ProjectionBlock(
            in_channels,
            mid_channels,
            activ_type=activ_type,
            use_weight_norm=use_weight_norm,
            final_bias=True,
            final_activ=True,
            normalize=True,
        )
        self.proj = ProjectionBlock(
            mid_channels,
            out_channels,
            hidden_channels=final_projection_channels,
            activ_type=activ_type,
            use_weight_norm=use_weight_norm,
            final_bias=False,
            final_activ=False,
            normalize=True,
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

    def forward(self, x):
        # print(x.shape)
        x = self.lift(x)

        for block in self.blocks:
            # apply FNO blocks sequentially
            x = block(x)

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
        self.filt = torch.nn.Conv3d(mid_channels, mid_channels, kernel_size=1)

        if use_weight_norm:
            self.filt = weight_norm(self.filt)

        self.activ = get_activ(activ_type, mid_channels)

        if normalize:
            # group size = 1 (a.k.a. easy layernorm)
            self.norm_0 = torch.nn.GroupNorm(1, mid_channels)

        # add another local FC layer before activation
        self.use_MLP = use_MLP
        if self.use_MLP:
            self.mlp_layer = ProjectionBlock(
                mid_channels,
                mid_channels,
                activ_type=activ_type,
                use_weight_norm=use_weight_norm,
                final_bias=True,
                final_activ=True,
                normalize=True,
            )

    # just the middle bit of an FNO
    def forward(self, x):
        # residual outside normalization
        if self.resid_conn:
            x0 = x

        # normalize before input
        if self.normalize:
            x = self.norm_0(x)

        x1 = self.conv(x)
        if self.use_MLP:
            x1 = self.mlp_layer(x1)
        x2 = self.filt(x)

        # now apply activation
        x = self.activ(x1 + x2)

        if self.resid_conn:
            # residual connection
            x += x0

        return x
