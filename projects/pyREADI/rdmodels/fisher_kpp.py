from rdmodels.rdmodel_base import RDModelBase
import numpy as np

__author__ = 'vaassen'


# TODO: test class, complete doc
class FisherKPPEq(RDModelBase):
    """Heat
    """

    def __init__(self, ndofs, ndims, domain, nspecies, diffs):
        # Base class initializer call
        super(FisherKPPEq, self).__init__(), ndofs, ndims, domain, nspecies, diffs

        self._r = None

        self.r = r

    def eval_n(self, u):
        return self.r * u * (1-u)

    def eval_n_jac(self, u):
        return self.r * (1 - 2*u)

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        assert isinstance(r, float), 'Expected r to be of type float'
        assert r > 0, 'Expected r to be positive'
        self._r = r
