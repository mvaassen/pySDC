from projects.pyREADI.fdschemes.laplacian_base import LaplacianBase
import numpy as np
import scipy.sparse as sp


class Lap4thCpact1D(LaplacianBase):

    def __init__(self, ndims, periodic=True):
        super(Lap4thCpact1D, self).__init__(ndims, periodic)

        if periodic:
            self.lhs_handle = self.get_lhs_matrix_periodic
            self.rhs_handle = self.get_rhs_matrix_periodic
        else:
            self.lhs_handle = self.get_lhs_matrix_dirichlet
            self.rhs_handle = self.get_rhs_matrix_dirichlet

    def get_lhs_matrix(self, nx):
        return self.lhs_handle(nx)

    def get_rhs_matrix(self, nx):
        return self.rhs_handle(nx)

    def get_lhs_matrix_periodic(self, nx):
        # Periodic boundary
        stencil = [1./10, 1, 1./10]
        diags = [-1, 0, 1]
        a1d = sp.diags(stencil, diags, shape=(nx, nx), format='lil')

        if self.periodic:
            a1d[0, -1] = stencil[0]
            a1d[-1, 0] = stencil[-1]

        a1d = sp.csr_matrix(a1d)
        return a1d

    def get_rhs_matrix_periodic(self, nx):
        # Periodic boundary
        a = 6./5
        stencil = [a, -2 * a, a]
        diags = [-1, 0, 1]
        a1d = sp.diags(stencil, diags, shape=(nx, nx), format='lil')

        if self.periodic:
            a1d[0, -1] = stencil[0]
            a1d[-1, 0] = stencil[-1]

        a1d = sp.csr_matrix(a1d)
        return a1d

    def get_lhs_matrix_dirichlet(self, nx):
        stencil = [1. / 10, 1, 1. / 10]
        diags = [-1, 0, 1]
        M = sp.diags(stencil, diags, shape=(nx, nx), format='lil')[1:-1, :]
        bla = np.zeros(nx)
        bla[0:2] = np.array([1., 11.])
        M = sp.vstack([bla, M, bla[::-1]])
        # print(M.todense())
        M = sp.csr_matrix(M)

        return M

    def get_rhs_matrix_dirichlet(self, nx):
        a = 6. / 5
        stencil = [a, -2 * a, a]
        diags = [-1, 0, 1]
        a1d = sp.diags(stencil, diags, shape=(nx, nx), format='lil')[1:-1, :]

        bla = np.zeros(nx)
        bla[0:4] = np.array([13., -27., 15., -1.])
        a1d = sp.vstack([bla, a1d, bla[::-1]])
        # print(a1d.todense())
        a1d = sp.csr_matrix(a1d)

        return a1d
