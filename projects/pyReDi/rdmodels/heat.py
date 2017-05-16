from rdmodels.rdmodel_base import RDModelBase
import numpy as np

__author__ = 'vaassen'


# TODO: test class, complete doc
class HeatEq(RDModelBase):
    """Heat
    """

    def __init__(self, ndofs, ndims, domain, bvp_class, bc, nspecies, diffs):
        # Base class initializer call
        super(HeatEq, self).__init__(ndofs, ndims, domain, bvp_class, bc, nspecies, diffs)

    def eval_n(self, u, at_indices=None):
        return 0*u

    def eval_n_jac(self, u):
        return 0

    def coarsen(self, ndofs_coarse):
        return type(self)(ndofs_coarse, self.ndims, self.domain, self.bvp_class, self.bc, self.nspecies, self.diffs)
