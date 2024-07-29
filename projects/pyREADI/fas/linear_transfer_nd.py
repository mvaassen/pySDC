import numpy as np
import scipy.sparse as sp
from projects.pyREADI.fas.transfer_base import TransferBase


class LinearTransferND(TransferBase):
    """Implementation of the linear prolongation and restriction operators

    Attributes:
        I_2htoh (scipy.sparse.csc_matrix): prolongation matrix
        I_hto2h (scipy.sparse.csc_matrix): restriction matrix
    """

    def __init__(self, nx_fine, nx_coarse, ndims, *args, **kwargs):
        """Initialization routine for transfer operators

        Args:
            nx_fine (int): number of DOFs per dimension on the fine grid
            nxs_coarse (int): number of DOFs per dimension on the coarse grid
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        # for this particular transfer class, we need to make a few assumptions
        assert isinstance(nx_fine, int), type(nx_fine)
        assert isinstance(nx_coarse, int)
        #assert (nx_fine + 1) % 2 == 0
        #assert nx_coarse == (nx_fine + 1) / 2 - 1

        super(LinearTransferND, self).__init__(nx_fine, nx_coarse, ndims, *args, **kwargs)

        self.restriction_matrix = None
        self.restriction_matrix_w_boundary = None
        self.interpolation_matrix = None

        # pre-compute restriction and prolongation matrices
        self._compute_transfer()

    def _fwr_1d_matrix_periodic(self):
        # periodic boundary
        fwr = np.zeros((self.nx_coarse, self.nx_fine))
        np.fill_diagonal(fwr[:, 0::2], 1./2)
        np.fill_diagonal(fwr[:, 1::2], 1./4)
        np.fill_diagonal(fwr[1:, 1::2], 1./4)
        fwr[0, -1] += 1./4  # += ensures correct wrap-around for the last level

        return sp.csr_matrix(fwr)

    def _fwr_interp_1d_matrices_dirichlet(self):
        # dirichlet boundary

        # restriction
        fwr = np.zeros((self.nx_coarse, self.nx_fine))
        np.fill_diagonal(fwr[:, 1::2], 1./2)
        np.fill_diagonal(fwr[:, 2::2], 1./4)
        np.fill_diagonal(fwr[:, 0::2], 1./4)

        # restriction operator for preserving the boundary values
        fwr_wb = np.zeros((self.nx_coarse + 2, self.nx_fine + 2))
        fwr_wb[0, 0] = 1
        fwr_wb[-1, -1] = 1
        fwr_wb[1:-1, 1:-1] = fwr

        # interpolation
        # boundary condition relevant for interpolation
        interpol = 2 * fwr.transpose()
        l = np.zeros((self.nx_fine, 1))
        l[0, 0] = 1./2
        interpol = np.hstack([l, interpol, np.flipud(l)])

        return sp.csr_matrix(fwr), sp.csr_matrix(fwr_wb), sp.csr_matrix(interpol)

    def _compute_transfer(self):
        """Helper routine for the prolongation operator

        Args:
            nx_fine (int): number of DOFs per dimension on the fine grid
            nx_coarse (int): number of DOFs  per dimension on the coarse grid

        Returns:
            scipy.sparse.csc_matrix: sparse prolongation matrix of size
                `ndofs_fine**2` x `ndofs_coarse**2`
        """

        if self.bc_type == 'PeriodicProblem':
            # compute 1D full weighting restriction operator
            fwr_1d = self._fwr_1d_matrix_periodic()

            # compute N-D full weighting restriction operator
            self.restriction_matrix = sp.csr_matrix(reduce(sp.kron, [fwr_1d] * self._ndims))
            self.restriction_matrix_w_boundary = self.restriction_matrix
            # compute N-D interpolation operator by variational property
            self.interpolation_matrix = 2**self._ndims * self.restriction_matrix.transpose()
        elif self.bc_type == 'DirichletProblem':
            print('computing dirichlet transfer')
            # compute 1D full weighting restriction operator
            fwr_1d, fwr_wb_1d, interpol_1d = self._fwr_interp_1d_matrices_dirichlet()

            # compute N-D full weighting restriction operator
            self.restriction_matrix = sp.csr_matrix(reduce(sp.kron, [fwr_1d] * self._ndims))
            # compute N-D full weighting restriction operator (preserves boundary values)
            self.restriction_matrix_w_boundary = sp.csr_matrix(reduce(sp.kron, [fwr_wb_1d] * self._ndims))
            # compute N-D interpolation operator
            # self.interpolation_matrix = sp.csr_matrix(reduce(sp.kron, [interpol_1d] * self._ndims))
            # only necessary if something other than the error is being interpolated
            # error should always obey homogeneous Dirichlet bc
            self.interpolation_matrix = 2**self._ndims * self.restriction_matrix.transpose()

    def restrict(self, u_fine, with_boundary=False):
        """Routine to apply restriction

        Args:
            u_fine (numpy.ndarray): vector on fine grid
        Returns:
            numpy.ndarray: vector on coarse grid
        """
        if not with_boundary:
            return self.restriction_matrix.dot(u_fine)
        else:
            return self.restriction_matrix_w_boundary.dot(u_fine)

    def interpolate(self, u_coarse):
        """Routine to apply prolongation

        Args:
            u_coarse (numpy.ndarray): vector on coarse grid
        Returns:
            numpy.ndarray: vector on fine grid
        """
        return self.interpolation_matrix.dot(u_coarse)
