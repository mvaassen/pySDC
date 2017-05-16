from rdmodels.rdmodel_base import RDModelBase
import numpy as np


class GeneralizedFisher1D(RDModelBase):

    def __init__(self, ndofs, domain, bvp_class, bc, nspecies, diffs, lambda0, nu):
        def dirty_evil_hardcoded_bc(x, t):
            lambda1 = self.lambda0/2.0 * ((self.nu/2.0 + 1)**0.5 + (self.nu/2.0 + 1)**(-0.5))
            delta = lambda1 - np.sqrt(lambda1**2 - self.lambda0**2)
            return np.array([(1 + (2 ** (self.nu / 2.0) - 1) * np.exp(-self.nu / 2.0 * delta * (x[0] + 2 * lambda1 * t))) ** (
            -2.0 / self.nu)])
            return (1 + (2**(self.nu/2.0) - 1)*np.exp(-self.nu/2.0*delta*(x[0] + 2*lambda1*t)))**(-2.0/self.nu)

        super(GeneralizedFisher1D, self).__init__(ndofs, 1, domain, bvp_class, dirty_evil_hardcoded_bc, nspecies, diffs)

        self.lambda0 = lambda0
        self.nu = nu

    # def eval_n(self, u, at_indices=None):
    #     return self.lambda0**2 * u[0] * (1 - u**self.nu)
    #
    # def eval_n_jac(self, u):
    #     return self.lambda0**2 - self.lambda0**2 * (self.nu + 1) * u**self.nu

    def eval_n(self, u):
        return np.array([self.lambda0**2 * u[0] * (1 - u[0]**self.nu)])

    def eval_n_jac(self, u):
        return np.array([[self.lambda0**2 - self.lambda0**2 * (self.nu + 1) * u[0]**self.nu]])

    def coarsen(self, ndofs_coarse):
        return type(self)(ndofs_coarse, self.domain, self.bvp_class, self.bc, self.nspecies, self.diffs, self.lambda0, self.nu)
