from rdmodels.rdmodel_base import RDModelBase
import numpy as np

__author__ = 'vaassen'


# TODO: test class, complete doc
class GrayScottEq(RDModelBase):
    """Gray-Scott
    """

    def __init__(self, ndofs, ndims, domain, bvp_class, bc, diffs, f_rate, k_rate):
        # Base class initializer call
        super(GrayScottEq, self).__init__(ndofs, ndims, domain, bvp_class, bc, 2, diffs)

        self._f_rate = None
        self._k_rate = None

        self.f_rate = f_rate
        self.k_rate = k_rate

    def eval_n(self, u):
        assert isinstance(u, np.ndarray), 'Expected u to be of type numpy.ndarray'
        #assert u.size == 2, 'The Gray-Scott system is a 2 species system'
        # R_1(u, v) := -uv**2 + F(1 - u)
        # R_2(u, v) := uv**2 - (F + k)*v
        return np.array([-u[0] * u[1]**2 + self.f_rate*(1 - u[0]), u[0] * u[1]**2 - (self.f_rate + self.k_rate)*u[1]])

    def eval_n_jac(self, u):
        return np.array([[-u[1]**2 - self.f_rate, -2*u[0]*u[1]], [u[1]**2, 2*u[0]*u[1] - (self.f_rate+self.k_rate)]])

    def coarsen(self, ndofs_coarse):
        return type(self)(ndofs_coarse, self.ndims, self.domain, self.bvp_class, self.bc, self.diffs, self.f_rate, self.k_rate)

    @property
    def f_rate(self):
        return self._f_rate

    @f_rate.setter
    def f_rate(self, f_rate):
        #assert isinstance(f_rate, float), 'Expected f_rate to be of type int'
        #assert f_rate > 0, 'Expected f_rate to be positive'
        self._f_rate = f_rate

    @property
    def k_rate(self):
        return self._k_rate

    @k_rate.setter
    def k_rate(self, k_rate):
        #assert isinstance(k_rate, float), 'Expected k_rate to be of type int'
        #assert k_rate > 0, 'Expected k_rate to be positive'
        self._k_rate = k_rate
