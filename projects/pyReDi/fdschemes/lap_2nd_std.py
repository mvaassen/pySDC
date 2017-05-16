from fdschemes.lap_explicit_base import LapExplicitBase
import scipy.sparse as sp


class Lap2ndStd(LapExplicitBase):

    def __init__(self, ndims, periodic=True):
        super(Lap2ndStd, self).__init__(ndims, periodic)

    def get_rhs_matrix(self, nx):
        stencil = [1, -2, 1]
        diags = [-1, 0, 1]
        a1d = sp.diags(stencil, diags, shape=(nx, nx), format='lil')

        if self.periodic:
            a1d[0, -1] = stencil[0]
            a1d[-1, 0] = stencil[-1]

        a1d = sp.csr_matrix(a1d)
        return sp.csr_matrix(reduce(sp.kronsum, [a1d]*self.ndims))
