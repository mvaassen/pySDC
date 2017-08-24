from __future__ import division

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.sparse.linalg as sla

from projects.pyREADI.boundarytypes.b_type_base import BoundaryValueProblemBase
from projects.pyREADI.fdschemes.lap_2nd_std import Lap2ndStd
from projects.pyREADI.fdschemes.lap_4th_cpact_1d import Lap4thCpact1D

from projects.pyREADI.fas.fas_multigrid import FASMultigrid
from projects.pyREADI.fas.linear_transfer_nd import LinearTransferND
from projects.pyREADI.fas.nonlinear_gauss_seidel import NonlinearGaussSeidel
from projects.pyREADI.tstepping.impl_euler_new import ImplEulerScheme

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError

__author__ = 'vaassen'


# TODO: complete doc
class RDModelBase(ptype):
    __metaclass__ = ABCMeta
    """Abstract base class for modeling reaction-diffusion systems

    """

    #def __init__(self, ndofs, ndims, domain, bvp_class, bc, nspecies, diffs):
    def __init__(self, problem_params, dtype_u, dtype_f):
        """Initializer for a generic reaction-diffusion model

        Args:
            ndofs (numpy.ndarray): Number of spatial degrees of freedom (powers of 2 recommended)
            diffs (scipy.sparse.dia_matrix): Diagonal matrix containing the diffusion coefficients
        """

        # These parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'ndims', 'domain', 'nspecies', 'diffs', 'fd_laplacian', 'bc_type', 'bc_func',
                          'is_hierarchy_master_problem']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # TODO check dividability by 2 for mg
        # self.ndofs = ndofs
        # self.ndims = ndims
        # self.domain = domain
        #
        # self.nspecies = nspecies
        #
        # self.diffs = diffs
        #
        # self.domlen = domain[1] - domain[0]
        #
        # self.bvp_class = bvp_class
        # self.bc = bc

        # Invoke super init
        super(RDModelBase, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

        # Save for coarsening
        self.problem_params = problem_params

        # Begin problem specific setup
        self.domlen = self.params.domain[1] - self.params.domain[0]

        # BC handling
        # pre-setup
        self.nnodes, self.dx = self.params.bc_type.compute_nnodes_dx(self.params.nvars, self.domlen)
        self.grid = self.generate_grid()
        # actual setup
        self.bc_handler = self.params.bc_type(self)

        # Attach discrete Laplacian
        self.laplacian = self.params.fd_laplacian(self.params.ndims, periodic=False)
        self._generate_fd_laplacian()

        # Multigrid setup
        if self.params.is_hierarchy_master_problem:
            # check for compatibility
            # do setup
            self._one_multigrid_to_rule_them_all()

        # Lil matrix versions of the linear operators
        self.M_matrix_lil = np.array([self.M_matrix[i].tolil() for i in range(self.params.nspecies)])
        self.L_matrix_lil = np.array([self.L_matrix[i].tolil() for i in range(self.params.nspecies)])

        # Here comes Multigrid
        self.factor = 0
        self._one_multigrid_to_rule_them_all()

    def _one_multigrid_to_rule_them_all(self):
        # Initialize FAS multigrid solver
        self._mg = FASMultigrid(self, galerkin=False)
        self._mg.attach_transfer(LinearTransferND)

        # Set up fine grid system operator according to specified scheme and model
        sysop = ImplEulerScheme(self, lambda _: self.factor)

        # Attach FAS smoother and fine grid system operator
        # MG is responsible for the hierarchy setup
        self._mg.attach_smoother(NonlinearGaussSeidel, sysop)

    @staticmethod
    def slicing_helper(A, u, i):
        return sum(val * u[j] for j, val in zip(A.rows[i], A.data[i]))

    def eval_m_i_lil(self, v_red, i):
        return np.array([RDModelBase.slicing_helper(self.M_matrix_lil[j], v_red[j], i) for j in range(self.params.nspecies)])

    def eval_l_i_lil(self, u_ext, i):
        """ (Lu)_i
        """
        return np.array([RDModelBase.slicing_helper(self.L_matrix_lil[j], u_ext[j], i) for j in range(self.params.nspecies)])

    def eval_m_n_i_lil(self, u_ext, i):
        return np.array([sum(val * self.eval_n(u_ext[:, j])[k] for j, val in zip(self.M_matrix_lil[k].rows[i], self.M_matrix_lil[k].data[i])) for k in range(self.params.nspecies)])

    @staticmethod
    def identity(u):
        return u

    def eval_f(self, u, t):
        f = self.dtype_f(self.init)
        uext = self.extend(np.array([u.values]), t)
        f.values, = self.eval_rhs_wb(uext)
        return f

    def solve_system(self, rhs, factor, u0, t):
        u = self.dtype_u(u0)
        self.factor = factor

        u.values, = self._mg.mg_iter(np.array([u0.values]), np.array([rhs.values]), self.params.mg_restol, t, nu1=1, nu2=1)

        return u

    def eval_rhs(self, u_ext):
        return self.eval_minv_l(u_ext) + self.eval_n(self.reduce(u_ext))

    def eval_rhs_wb(self, u_ext):
        lu = self.eval_l(u_ext)
        return np.array([sla.spsolve(self.M_matrix[i], lu[i]) for i in range(self.params.nspecies)]) + self.eval_n(u_ext)

    def eval_m_rhs(self, u_ext):
        return self.eval_l(u_ext) + self.eval_m_n(u_ext)

    def eval_m(self, v_ext):
        return np.array([self.M_matrix[i].dot(v_ext[i]) for i in range(self.params.nspecies)])

    def eval_m_i(self, v_red, i):
        return np.array([self.M_matrix[j][i, :].dot(v_red[j]) for j in range(self.params.nspecies)])

    def eval_l(self, u_ext):
        """ In case of linear operator modification in subclasses ALWAYS modify this method
        """
        return np.array([self.L_matrix[i].dot(u_ext[i]) for i in range(self.params.nspecies)])

    def eval_l_i(self, u_ext, i):
        """ (Lu)_i
        """
        return np.array([self.L_matrix[j][i, :].dot(u_ext[j]) for j in range(self.params.nspecies)])

    def eval_minv_l(self, u_ext):
        lu = self.eval_l(u_ext)
        return np.array([sla.spsolve(self.M_matrix[i], lu[i])[1:-1] for i in range(self.params.nspecies)])

    @abstractmethod
    def eval_n(self, u_inner):
        """ In case of nonlinear operator modification in subclasses ALWAYS modify this method
        """
        pass

    def eval_m_n(self, u_ext):
        nu = self.eval_n(u_ext)
        return np.array([self.M_matrix[i].dot(nu[i]) for i in range(self.params.nspecies)])

    def eval_m_n_i(self, u_ext, i):
        nu = self.eval_n(u_ext)
        return self.eval_m_i(nu, i)
        #altnu = self.eval_n(u_ext)
        nu = np.zeros_like(u_ext)
        ind = self.M_matrix[0][i, :].nonzero()[1]
        #print(nu)
        nu[:, ind] = self.eval_n(u_ext[:, ind])
        #print nu
        #print altnu
        #exit()
        return self.eval_m_i(nu, i)

    @abstractmethod
    def eval_n_jac(self, u):
        pass

    def eval_jac(self, u):
        pass

    def coarsen(self, ndofs):
        # To be implemented by specific problem classes
        # Creates a coarse grid version of the problem
        self.problem_params['nvars'] = ndofs
        return type(self)(self.dtype_u, self.dtype_f, self.problem_params)

    def generate_grid(self):
        grid_axis = self.params.domain[0] + np.arange(self.nnodes) * self.dx
        return np.meshgrid(*[grid_axis]*self.params.ndims, indexing='ij')

    def generate_ndofs_list(self):
        #return [int(self.ndofs / 2 ** l) for l in range(1)]
        if type(self.bc_handler).__name__ == 'DirichletProblem':
            nlevels = int(np.log2(self.params.nvars)) + 1 - 1
            return [int((self.params.nvars + 1) / 2**l) - 1 for l in range(nlevels)]
        elif type(self.bc_handler).__name__ == 'PeriodicProblem':
            nlevels = int(np.log2(self.params.nvars))
            return [int(self.params.nvars / 2**l) for l in range(nlevels)]
        else:
            pass

    # masks need fixing
    def generate_periodic_b_mask(self):
        mask_1d = np.zeros(self.params.nvars, dtype=bool)
        return reduce(np.kron, [mask_1d] * self.ndims)

    def _generate_fd_laplacian(self):
        lhs_matrix = self.laplacian.get_lhs_matrix(self.nnodes)
        self.M_matrix = np.array([lhs_matrix]*self.params.nspecies)

        rhs_matrix = self.laplacian.get_rhs_matrix(self.nnodes) / self.dx**2
        #print(rhs_matrix.todense())
        #print(lhs_matrix.todense())
        #rhs_matrix = self.fdscheme.get_rhs_matrix(self.nnodes)[1:self.nnodes-1, :] / self.dx ** 2
        #print(rhs_matrix.todense())
        self.L_matrix = np.array([d * rhs_matrix for d in self.params.diffs])

    def extend(self, u_red, t):
        # delegate call
        return self.bc_handler.extend(u_red, t)

    def extend_sdc(self, u_red, t):
        u = self.dtype_u(u_red)

        # delegate call
        uext, = self.bc_handler.extend(np.array([u.values]), t)

        bla = self.dtype_u(init=uext.shape[0])
        bla.values = uext

        return bla

    def reduce(self, u_ext):
        # delegate call
        return self.bc_handler.reduce(u_ext)

    def u_exact(self, t):
        pass
