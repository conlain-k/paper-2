import torch


def herm_conj(Vh):
    return Vh.transpose(-2, -1).conj()


class TorchPCA(torch.nn.Module):
    def __init__(self, k=None, center=True, whiten=False):
        super().__init__()
        self.k = k
        self.center = center
        self.whiten = whiten

    def fit(self, X, k_override=None):
        """Fit a PCA for matrix X where each row is a sample and each column is a feature"""
        k = k_override or self.k
        assert X is not None
        self.spatial_sizes = list(X.shape[1:])
        # flatten out all dimensions except the sample dimension (row)
        X_fit = X.flatten(start_dim=1).double()

        if self.center:
            print("Centering")
            # find mean for each feature (average out rows)
            mu = X_fit.mean(dim=0)

            print(mu.shape, mu.max(), mu.min())

            # now add a singleton row dimension (broadcast the same vector to each sample)
            mu = mu.reshape(1, -1)

            # write our mu vector to the state
            self.register_buffer("mu", mu)
            X_fit = X_fit - self.mu

        if self.whiten:
            print("Whitening")
            # find std for each feature (unit variance among columns)
            sigma = X_fit.std(dim=0)

            # minimum scaling factor (to avoid numerical issues)
            sigma = torch.clamp(min=1e-4)

            print(sigma.shape, sigma.max(), sigma.min())

            # now add a singleton row dimension (broadcast the same vector to each sample)
            sigma = sigma.reshape(1, -1)

            # write our sigma vector to the state
            self.register_buffer("sigma", sigma)

            X_fit = X_fit / self.sigma

        print("Computing SVD")
        # X = U S V'
        # X V = U S
        # conduct SVD in double precision, then downcast
        U, S, Vh = torch.linalg.svd(X_fit, full_matrices=False)

        U, S, Vh = U, S, Vh
        V = herm_conj(
            Vh
        )  # get the regular version of V from its hermite conjugate (which torch returns)

        if k is not None:
            # Take first k columns (components) of V
            V = V[:, :k]
            S = S[:k]
            U = U[:, :k]

        # Save V back to state and make sure it gets saved/loaded correctly
        self.register_buffer("V", V.clone())

        evs = S**2
        self.relative_variance = evs / sum(evs)

        # check that X V = U S, including centering and up to floating point prec
        # torch.testing.assert_close(
        #     torch.mm(X_fit, self.V), torch.mm(U, torch.diag(S))
        # )
        torch.testing.assert_close(self.forward(X).double(), torch.mm(U, torch.diag(S)))

        # now project X through and back PC space (loses some data)
        X_fit_reduced = self.inverse_transform(self.forward(X), verbose=True)
        # do it again (but now we lie in the PC subspace)
        X_fit_reduced_2 = self.inverse_transform(
            self.forward(X_fit_reduced), verbose=True
        )

        # second projection pass should be lossless
        torch.testing.assert_close(X_fit_reduced, X_fit_reduced_2)

        print("Fitting finished")

    def forward(self, x):
        # first flatten out spatial dims
        x = x.flatten(start_dim=1).double()

        # normalize before transforming
        if self.center:
            x = x - self.mu
        if self.whiten:
            x = x / self.sigma

        # project onto PC components
        x_proj = torch.mm(x, self.V).float()

        return x_proj

    def transform(self, x):
        return self.forward(x)

    def inverse_transform(self, x_proj, reshape=False, verbose=False):
        """inverse projection of the PCA transform"""

        # do everything in double precision
        x_proj = x_proj.double()

        if len(x_proj.shape) == 1:
            # add an initial dimension to facilitate vector multiplication
            x_proj = x_proj.reshape(1, -1)

        x_unproj = torch.mm(x_proj, herm_conj(self.V))

        # undo normalizations
        if self.whiten:
            x_unproj = x_unproj * self.sigma
        if self.center:
            x_unproj = x_unproj + self.mu

        if reshape:
            x_unproj = x_unproj.reshape([-1] + self.spatial_sizes)

        return x_unproj.float()
