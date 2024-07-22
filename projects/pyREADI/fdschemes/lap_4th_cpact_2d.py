from fdschemes.laplacian_base import LaplacianBase
import scipy.sparse as sp


class Lap4thCpact2D(LaplacianBase):

    def __init__(self, ndims, periodic=True):
        super(Lap4thCpact2D, self).__init__(ndims, periodic)

    def get_lhs_matrix(self, nx):
        # Periodic boundary
        stencil = [1./12, 4./12, 1./12]
        diags = [-1, 0, 1]
        a1d = sp.diags(stencil, diags, shape=(nx, nx), format='lil')
        a1d[0, -1] = stencil[0]
        a1d[-1, 0] = stencil[-1]
        a1d = sp.csr_matrix(a1d)
        return sp.csr_matrix(sp.kronsum(a1d, a1d))

    def get_rhs_matrix(self, nx):
        # Periodic boundary
        a = 6./5
        stencil = [a, -2 * a, a]
        diags = [-1, 0, 1]
        a1d = sp.diags(stencil, diags, shape=(nx, nx), format='lil')
        a1d[0, -1] = stencil[0]
        a1d[-1, 0] = stencil[-1]
        a1d = sp.csr_matrix(a1d)
        return a1d
