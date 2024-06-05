"""
3D FNO spectral conv, adapted from code by Zongyi Li
"""

# adapted from here: https://github.com/zongyi-li/fourier_neural_operator

import torch
import numpy as np

################################################################
# 3d fourier layer
################################################################


class SpectralConv3d(torch.nn.Module):
    """
    3D Fourier layer. It does FFT, linear transform, and Inverse FFT.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        modes1,
        modes2,
        modes3,
        scale_fac=1,
        use_bias=False,
    ):
        super(SpectralConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = scale_fac / (in_channels * out_channels)

        # weight start uniformly between 0 and 1
        self.weights = torch.nn.Parameter(
            self.scale
            * torch.rand(
                self.in_channels,
                self.out_channels,
                2 * self.modes1 - 1,
                2 * self.modes2 - 1,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

        print(f"Weights shape is {self.weights.shape}")

        if use_bias:
            # use uniform bias everywhere
            # self.bias = torch.nn.Parameter(torch.zeros((self.out_channels, 1, 1, 1)))
            self.bias = torch.nn.Parameter(torch.zeros((self.out_channels,)))
            print(f"Bias shape is {self.bias.shape}")
        else:
            self.bias = None

    def get_central_inds(self, size):
        # get a slice corresponding to the center of a volume
        # get new stencil sizes in each direction
        s1, s2 = size[-3], size[-2]

        # get midpoints of new weights tensor
        m1, m2 = s1 // 2, s2 // 2

        # input is biijk, output is boijk, elem-wise multiply along last
        # go from center - m to center + m (not including endpoints)
        inds = np.s_[
            :,
            :,
            m1 - self.modes1 + 1 : m1 + self.modes1,
            m2 - self.modes2 + 1 : m2 + self.modes2,
            0 : self.modes3,
        ]

        return inds

    def extra_repr(self):
        """Set extra print information"""
        pstr = f"in_channels={self.in_channels}, out_channels={self.out_channels}, modes={(self.modes1, self.modes2, self.modes3)}"
        return pstr

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        # shift the primary frequencies to the middle
        x_ft = torch.fft.fftshift(x_ft, dim=(-3, -2))

        # get central bit of x for multiplication
        central_inds = self.get_central_inds(x_ft.shape[-3:])

        # get previous shape
        new_shape = list(x_ft.shape)

        # set number of output channels
        new_shape[1] = self.out_channels

        # preallocate memory for output
        out_ft = x_ft.new_zeros(new_shape)

        out_ft[central_inds] = self.compl_mul3d(x_ft[central_inds], self.weights)

        out_ft = torch.fft.ifftshift(out_ft, dim=(-3, -2))

        # Return to physical space
        out = torch.fft.irfftn(out_ft, s=x.shape[-3:])

        if self.bias is not None:
            out += self.bias.reshape(1, -1, 1, 1, 1)
            # print("bias", self.bias)

        return out


class SpectralConv3d_alt(torch.nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, **kwargs):
        super(SpectralConv3d_alt, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = torch.nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = torch.nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = torch.nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = torch.nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x
