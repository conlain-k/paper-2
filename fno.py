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
        projection_channels=None,  # defaults to bigger of mid_channels, out_channels
        activ_type="gelu",
        use_weight_norm=False,
        **kwargs
    ):
        super().__init__()
        # lifting and projection blocks
        self.lift = ProjectionBlock(
            in_channels,
            mid_channels,
            activ_type=activ_type,
            use_weight_norm=use_weight_norm,
            final_bias=True,
        )
        self.proj = ProjectionBlock(
            mid_channels,
            out_channels,
            hidden_channels=projection_channels,
            activ_type=activ_type,
            use_weight_norm=use_weight_norm,
            final_bias=False,
        )

        self.forward_latent = FNO_Middle(
            activ_type=activ_type,
            use_weight_norm=use_weight_norm,
            mid_channels=mid_channels,
            **kwargs
        )

    def forward(self, x):
        # print(x.shape)
        x = self.lift(x)

        x = self.forward_latent(x)

        # now do two projection steps, with an activation in the middle
        x = self.proj(x)

        return x


class FNO_Middle(torch.nn.Module):
    def __init__(
        self,
        activ_type,
        use_weight_norm,
        mid_channels,
        modes,
        normalize,
        init_weight_scale,
        use_fourier_bias,
        resid_conn,
        **kwargs
    ):
        super().__init__()

        self.resid_conn = resid_conn

        # set up FNO filters
        convs = []
        filts = []
        acts = []
        norms = []
        for mm in modes:
            # spectral convolution
            convs.append(
                fourier_conv.SpectralConv3d(
                    mid_channels,
                    mid_channels,
                    mm,
                    mm,
                    mm,
                    scale_fac=init_weight_scale,
                    use_bias=use_fourier_bias,
                )
            )

            # local channel-wise filter
            filt = torch.nn.Conv3d(mid_channels, mid_channels, kernel_size=1)
            if use_weight_norm:
                filt = weight_norm(filt)
            filts.append(filt)

            acts.append(get_activ(activ_type, mid_channels))

            if normalize:
                # group size = 1 (a.k.a. easy layernorm)
                norms.append(torch.nn.GroupNorm(1, mid_channels))

        # now store ops in the class
        self.convs = torch.nn.ModuleList(convs)
        self.filts = torch.nn.ModuleList(filts)
        self.acts = torch.nn.ModuleList(acts)
        if normalize:
            self.norms = torch.nn.ModuleList(norms)
        else:
            self.norms = None

    # just the middle bit of an FNO
    def forward(self, x):
        # for each conv/filter op, apply them
        for ind, (conv, filt, activ) in enumerate(
            zip(self.convs, self.filts, self.acts)
        ):
            if self.resid_conn:
                x0 = x
            # residual after normalization
            if self.norms:
                x = self.norms[ind](x)
            x1 = conv(x)
            x2 = filt(x)
            # now apply activation
            x = activ(x1 + x2)

            if self.resid_conn:
                # residual connection
                x += x0

        return x
