import numpy as np

from projects.pyREADI.rdmodels.rdmodel_base import RDModelBase
from pySDC.core.Errors import ParameterError, ProblemError

class GeneralizedFisher1D(RDModelBase):

    def __init__(self, problem_params, dtype_u, dtype_f):
        # These parameters will be used later, so assert their existence
        essential_keys = ['nu', 'lambda0']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        problem_params['ndims'] = 1
        problem_params['nspecies'] = 1
        problem_params['diffs'] = np.array([1])

        def dirty_evil_hardcoded_bc(x, t):
            lambda1 = self.params.lambda0/2.0 * ((self.params.nu/2.0 + 1)**0.5 + (self.params.nu/2.0 + 1)**(-0.5))
            delta = lambda1 - np.sqrt(lambda1**2 - self.params.lambda0**2)
            return np.array([(1 + (2 ** (self.params.nu / 2.0) - 1) * np.exp(-self.params.nu / 2.0 * delta * (x[0] + 2 * lambda1 * t))) ** (
            -2.0 / self.params.nu)])
            return (1 + (2**(self.nu/2.0) - 1)*np.exp(-self.nu/2.0*delta*(x[0] + 2*lambda1*t)))**(-2.0/self.nu)

        problem_params['bc_func'] = dirty_evil_hardcoded_bc

        super(GeneralizedFisher1D, self).__init__(problem_params, dtype_u, dtype_f)

    # def eval_n(self, u, at_indices=None):
    #     return self.lambda0**2 * u[0] * (1 - u**self.nu)
    #
    # def eval_n_jac(self, u):
    #     return self.lambda0**2 - self.lambda0**2 * (self.nu + 1) * u**self.nu

    def eval_n(self, u):
        return np.array([self.params.lambda0**2 * u[0] * (1 - u[0]**self.params.nu)])

    def eval_n_jac(self, u):
        return np.array([[self.params.lambda0**2 - self.params.lambda0**2 * (self.params.nu + 1) * u[0]**self.params.nu]])

    # def coarsen(self, ndofs_coarse):
    #     return type(self)(ndofs_coarse, self.domain, self.bvp_class, self.bc, self.nspecies, self.diffs, self.lambda0, self.nu)

    def u_exact(self, t):
        me = self.dtype_u(self.init)
        xvalues = np.array([(i + 1 - (self.params.nvars + 1) / 2) * self.dx for i in range(self.params.nvars)])

        lam1 = self.params.lambda0 / 2.0 * ((self.params.nu / 2.0 + 1) ** 0.5 + (self.params.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1 ** 2 - self.params.lambda0 ** 2)
        me.values = (1 + (2 ** (self.params.nu / 2.0) - 1) *
                     np.exp(-self.params.nu / 2.0 * sig1 * (xvalues + 2 * lam1 * t))) ** (-2.0 / self.params.nu)
        return me
