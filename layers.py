import torch

from torch.nn.utils.parametrizations import weight_norm


def get_activ(activ_type, num_chan):
    if activ_type == "gelu":
        activ = torch.nn.GELU()
    elif activ_type == "relu":
        activ = torch.nn.ReLU()
    elif activ_type == "softplus":
        activ = torch.nn.Softplus()
    else:
        activ = torch.nn.PReLU(num_parameters=num_chan)

    return activ


class SimpleLayerNet(torch.nn.Module):
    """A simple 3-layer update network. Adapted from the original paper."""

    def __init__(self, in_channels, out_channels, inner_channels=32, num_blocks=3):
        super().__init__()

        # how many channels to filter over
        self.chan = inner_channels

        # 1x1 convolution to lift to filter space
        self.lift = ProjectionBlock(in_channels, self.chan, activ_type="gelu")

        blocks = []
        for _ in range(num_blocks):
            blocks.append(InceptionBlock_1(self.chan, self.chan))

        self.blocks = torch.nn.ModuleList(blocks)

        # Now do a rotation before projecting
        self.proj = ProjectionBlock(
            in_channels=self.chan, out_channels=out_channels, activ_type="geluu"
        )

        # normalization inside
        self.norm = torch.nn.InstanceNorm3d(num_features=self.chan)

    def forward(self, x):
        # lift, filter, and project. Intentionally very simple

        x = self.lift(x)

        for i, block in enumerate(self.blocks):

            # print(f"block {i}", torch.cuda.memory_summary())

            # x = self.norm(x)
            x = block(self.norm(x))

        x = self.proj(x)

        return x


class ConvBlock(torch.nn.Module):
    """Run one conv layer, possibly with an activation"""

    def __init__(
        self,
        ks,
        in_channels,
        out_channels,
        stride=1,
        padding=None,
        use_activ=True,
        use_norm=False,
        bias=True,
        use_weight_norm=False,
        activ_type="gelu",
    ):
        super().__init__()

        # if we didn't override padding, just use half the kernel width (preserve size)
        if padding is None:
            padding = ks // 2

        # do one conv layer
        self.conv = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            padding_mode="circular",
            bias=bias,
        )

        if use_weight_norm and not use_norm:
            # print("Using weight norm!")
            self.conv = weight_norm(self.conv)

        self.use_activ = use_activ
        if self.use_activ:
            self.activ = get_activ(activ_type, out_channels)

        self.use_norm = use_norm
        if self.use_norm:
            self.layer_norm = torch.nn.InstanceNorm3d(num_features=out_channels)

    def forward(self, x):
        # apply operations
        x = self.conv(x)

        # add possible extra ops
        if self.use_activ:
            x = self.activ(x)

        if self.use_norm:
            x = self.layer_norm(x)

        return x


class ConvTransposeBlock(torch.nn.Module):
    """Run one conv layer, possibly with an activation"""

    def __init__(
        self,
        ks,
        in_channels,
        out_channels,
        stride=1,
        use_activ=True,
        bias=True,
        use_weight_norm=False,
    ):
        super().__init__()

        # do one conv layer
        self.conv = torch.nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ks,
            stride=stride,
            padding=0,  # do the padding ourselves
            bias=bias,
            output_padding=1,  # used since our
        )

        if use_weight_norm:
            # print("Using weight norm!")
            self.conv = weight_norm(self.conv)

        self.use_activ = use_activ
        if self.use_activ:
            self.activ = torch.nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        # apply operationsprint
        x = self.conv(x)

        if self.use_activ:
            x = self.activ(x)
        return x


class InceptionBlock_2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # VERY simple inception arch
        self.branch1 = ConvBlock(1, in_channels, in_channels // 2, use_norm=False)

        self.proj3 = ConvBlock(1, in_channels, in_channels // 2)
        self.branch3 = ConvBlock(3, in_channels // 2, in_channels // 2, use_norm=False)

        self.proj5 = ConvBlock(1, in_channels, in_channels // 2)
        self.branch5_a = ConvBlock(
            3, in_channels // 2, in_channels // 2, use_norm=False
        )
        self.branch5_b = ConvBlock(
            5, in_channels // 2, in_channels // 2, use_norm=False
        )

        self.combine_block = ConvBlock(
            1, 3 * (in_channels // 2), out_channels, use_norm=False
        )

        self.norm = torch.nn.InstanceNorm3d(num_features=in_channels)

    def forward(self, x):
        x0 = x
        # first normalize, then project
        x = self.norm(x)

        # go down each branch
        x1 = self.branch1(x)
        x3 = self.branch3(self.proj3(x))
        x5 = self.branch5_b(self.branch5_a(self.proj5(x)))

        # channel-wise concat all of them
        x = torch.cat([x1, x3, x5], 1)

        # now intermix and reduce dimension
        x = self.combine_block(x)

        # use block in residual form
        return x + x0


class InceptionBlock_1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = ConvBlock(1, in_channels, in_channels // 2)

        # VERY simple inception arch
        self.branch1x1 = ConvBlock(
            1, in_channels // 2, in_channels // 2, use_norm=False
        )
        self.branch3x3 = ConvBlock(
            3, in_channels // 2, in_channels // 2, use_norm=False
        )
        self.branch5x5 = ConvBlock(
            5, in_channels // 2, in_channels // 2, use_norm=False
        )

        self.combine_block = ConvBlock(
            1, 3 * (in_channels // 2), out_channels, use_norm=False
        )

        self.norm = torch.nn.InstanceNorm3d(num_features=in_channels)

    def forward(self, x):
        x0 = x
        # first normalize, then project
        x = self.norm(x)
        x = self.proj(x)

        # go down each branch
        x1 = self.branch1x1(x)
        x3 = self.branch3x3(x)
        x5 = self.branch5x5(x)

        # channel-wise concat all of them
        x = torch.cat([x1, x3, x5], 1)

        # now intermix and reduce dimension
        x = self.combine_block(x)

        # use block in residual form
        return x + x0


class TinyVNet(torch.nn.Module):
    """A minimal network based on V-Net. Adapted from the first paper."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        L1_c = 32  # channels at first level
        L2_c = 2 * L1_c  # double on way down

        self.p1 = ProjectionBlock(in_channels, L1_c)

        # initial forward blocks
        self.f1 = ConvBlock(3, L1_c, L1_c)
        self.f2 = ConvBlock(3, L1_c, L1_c)

        # downsample block
        self.d1 = ConvBlock(2, L1_c, L2_c, stride=2, padding=0)

        # process in downsample space
        self.df1 = ConvBlock(3, L2_c, L2_c)
        self.df2 = ConvBlock(3, L2_c, L2_c)

        # upsample block
        self.u1 = ConvTransposeBlock(2, L2_c, L2_c, stride=2)

        # mix different length scales
        # 1x1x1 conv to save weights
        self.f3 = ConvBlock(1, L1_c + L2_c, 2 * L1_c)

        # now process at original lengthscale
        self.f4 = ConvBlock(3, 2 * L1_c, 2 * L1_c)
        self.f5 = ConvBlock(3, 2 * L1_c, 2 * L1_c)

        # now filter out into 6 outputs

        self.p2 = ProjectionBlock(2 * L1_c, out_channels)

    def forward(self, x):
        # lift into space
        x = self.p1(x)

        x0 = x
        # first layer processing
        x = self.f1(x)
        x = self.f2(x)

        # resid connection
        x = x0 + x

        # down branch
        xd = self.d1(x)
        xd = self.df1(xd)
        xd = self.df2(xd)

        # bring back up and recombine
        xd = self.u1(xd)
        # channel-wise concat the two levels
        x = torch.cat((x, xd), dim=1)
        x = self.f3(x)

        x0 = x

        # now filter in original space out project out
        x = self.f4(x)
        x = self.f5(x)
        # add connection from before skip
        x = x0 + x

        x = self.p2(x)

        return x


class ProjectionBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        activ_type="gelu",
        use_weight_norm=False,
        final_bias=False,
        final_activ=False,
        normalize=False,
    ):
        super().__init__()
        # put larger dimension through the activation
        hidden_channels = hidden_channels or max(in_channels, out_channels)

        self.proj_1 = torch.nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.proj_2 = torch.nn.Conv3d(
            hidden_channels, out_channels, kernel_size=1, bias=final_bias
        )

        self.activ_1 = get_activ(activ_type, hidden_channels)
        self.final_activ = final_activ
        self.normalize = normalize

        # set up activation
        if final_activ:
            self.activ_2 = get_activ(activ_type, hidden_channels)

        # set up normalization
        if normalize:
            self.norm = torch.nn.GroupNorm(1, hidden_channels)

        if use_weight_norm:
            self.proj_1 = weight_norm(self.proj_1)
            self.proj_2 = weight_norm(self.proj_2)

    def forward(self, x):
        x = self.proj_1(x)
        x = self.activ_1(x)

        if self.normalize:
            # normalize before second projection layer
            x = self.norm(x)
        x = self.proj_2(x)
        # apply final activation (if relevant)
        if self.final_activ:
            x = self.activ_2(x)

        return x
