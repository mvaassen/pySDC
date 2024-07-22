from __future__ import division

import sys
sys.path.append('/home/zam/vaassen/PycharmProjects/pyRD/')

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError

from time import time
import projects.pyREADI.tests.module_diagnostics as diagn

from rdmodels.generalized_fisher_1d import GeneralizedFisher1D
from rdmodels.gray_scott import GrayScottEq
from tstepping.impl_euler_new import ImplEulerScheme
from tstepping.implicit_timestepper import ImplicitTimeIntegrator
from tstepping.fwd_euler import solve_t
from fdschemes.lap_4th_cpact_1d import Lap4thCpact1D
from boundarytypes.periodic_problem import PeriodicProblem
from fas.fas_multigrid import FASMultigrid
from fas.linear_transfer_nd import LinearTransferND
from fas.nonlinear_gauss_seidel import NonlinearGaussSeidel

# noinspection PyUnusedLocal
class GrayScott_MG(ptype):
    """
    Example implementing the generalized Fisher's equation in 1D with finite differences

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'nvars_axis', 'diff_mat', 'feedrate', 'killrate', 'mg_maxiter', 'mg_restol', 'interval']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars_axis']) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(GrayScott_MG, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

        # compute dx and get discretization matrix A
        self.dx = (self.params.interval[1] - self.params.interval[0])/self.params.nvars_axis
        # print(self.dx)
        # exit()
        self.A = self.__get_A(self.params.nvars_axis, self.dx)
        self.A = sp.kron(self.params.diff_mat, self.A)
        #print(self.A.todense())
        #self.A = sp.kron(self.A, self.params.diff_mat)
        # if self.params.expl_boundary:
        #     print('hello')
        #     #self.A = self.__get_A_cmpct_expl_boundary(self.params.nvars, self.dx)
        # else:
        #     print('hello')
        #     self.A = self.__get_A_cmpct(self.params.nvars, self.dx)

        # cast model into my own data structure
        self.rdmodel = GrayScottEq(self.params.nvars_axis, 2, self.params.interval, PeriodicProblem, None,
                                   np.array([self.params.diff_mat[0, 0], self.params.diff_mat[1, 1]]),
                                   self.params.feedrate, self.params.killrate)

        # Here comes Multigrid
        self.factor = 0
        self._one_multigrid_to_rule_them_all()

    def _one_multigrid_to_rule_them_all(self):
        # Initialize FAS multigrid solver
        self._mg = FASMultigrid(self.rdmodel, galerkin=False)
        self._mg.attach_transfer(LinearTransferND)

        # Set up fine grid system operator according to specified scheme and model
        sysop = ImplEulerScheme(self.rdmodel, lambda _: self.factor)

        # Attach FAS smoother and fine grid system operator
        # MG is responsible for the hierarchy setup
        self._mg.attach_smoother(NonlinearGaussSeidel, sysop)

    @staticmethod
    def __get_A(N, dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (int): number of dofs
            dx (float): distance between two spatial nodes

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """
        stencil = [1, -2, 1]
        diags = [-1, 0, 1]
        a1d = sp.diags(stencil, diags, shape=(N, N), format='lil')

        a1d[0, -1] = stencil[0]
        a1d[-1, 0] = stencil[-1]

        a1d = sp.csr_matrix(a1d)/dx**2
        #return a1d/dx**2
        return sp.csr_matrix(sp.kronsum(a1d, a1d))

    @staticmethod
    def __get_A_cmpct(N, dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (int): number of dofs
            dx (float): distance between two spatial nodes

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        a = 6. / 5
        stencil = [a, -2 * a, a]
        diags = [-1, 0, 1]
        a1d = sp.diags(stencil, diags, shape=(N + 2, N + 2), format='lil')[1:-1, :]

        bla = np.zeros(N + 2)
        bla[0:4] = np.array([13., -27., 15., -1.])
        a1d = sp.vstack([bla, a1d, bla[::-1]])
        #print(a1d.todense())
        a1d *= 1.0 / (dx ** 2)
        a1d = sp.csr_matrix(a1d)

        stencil = [1. / 10, 1, 1. / 10]
        diags = [-1, 0, 1]
        M = sp.diags(stencil, diags, shape=(N + 2, N + 2), format='lil')[1:-1, :]
        bla = np.zeros(N + 2)
        bla[0:2] = np.array([1., 11.])
        M = sp.vstack([bla, M, bla[::-1]])
        #print(M.todense())
        M = sp.csc_matrix(M)

        bla = sla.inv(M)[1:-1, 1:-1].dot(a1d[1:-1, 1:-1])

        return sla.inv(M).dot(a1d)

    # @staticmethod
    # def __get_A_cmpct_expl_boundary(N, dx):
    #     """
    #     Helper function to assemble FD matrix A in sparse format
    #
    #     Args:
    #         N (int): number of dofs
    #         dx (float): distance between two spatial nodes
    #
    #     Returns:
    #         scipy.sparse.csc_matrix: matrix A in CSC format
    #     """
    #
    #     a = 6. / 5
    #     stencil = [a, -2 * a, a]
    #     diags = [-1, 0, 1]
    #     a1d = sp.diags(stencil, diags, shape=(N + 2, N + 2), format='lil')[1:-1, :]
    #
    #     bla = np.zeros(N + 2)
    #     bla[0:5] = np.array([35./12, -26./3, 19./2, -14./3, 11./12])
    #     a1d = sp.vstack([bla, a1d, bla[::-1]])
    #     # print(a1d.todense())
    #     a1d *= 1.0 / (dx ** 2)
    #     a1d = sp.csr_matrix(a1d)
    #
    #     stencil = [1. / 10, 1, 1. / 10]
    #     diags = [-1, 0, 1]
    #     M = sp.diags(stencil, diags, shape=(N + 2, N + 2), format='lil')[1:-1, :]
    #     bla = np.zeros(N + 2)
    #     bla[0] = 1.
    #     M = sp.vstack([bla, M, bla[::-1]])
    #     # print(M.todense())
    #     M = sp.csr_matrix(M)
    #
    #     bla = sla.inv(M)[1:-1, 1:-1].dot(a1d[1:-1, 1:-1])
    #
    #     return sla.inv(M).dot(a1d)

    def solve_t(self, t_interval, dt, u0):
        t = t_interval[0]
        u = u0

        # Tolerance for last time step
        ttol = 1e-16 * dt

        u_bla = self.dtype_u(self.init)

        while t < t_interval[1]:
            tn = t + dt

            if tn > t_interval[1] - ttol:
                # if upper time bound (te) exceeded fit last time step to te
                tn = t_interval[1]
                dt = tn - t

            u_bla.values = u
            un = u + dt*self.eval_f(u_bla, 0).values

            t = tn
            u = un

            # Test
            # u_hat[0, ind] = 0
            # u_hat[1, ind] = 0

        return u

    # noinspection PyTypeChecker
    def solve_system(self, rhs, factor, u0, t):
        u = self.dtype_u(u0)
        self.factor = factor

        u.values = self._mg.mg_iter(u0.values.reshape((2, self.params.nvars//2)), rhs.values.reshape((2, self.params.nvars//2)), self.params.mg_restol, t, nu1=1, nu2=1).flatten()

        return u

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        u_species = u.values.reshape((2, self.params.nvars//2))

        f = self.dtype_f(self.init)
        f.values = self.A.dot(u.values) + np.array(
            [-u_species[0] * u_species[1] ** 2 + self.params.feedrate * (1 - u_species[0]),
             u_species[0] * u_species[1] ** 2 - (self.params.feedrate + self.params.killrate) * u_species[1]]).flatten()

        return f

    def u_init(self):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)
        # resh = me.values.reshape(2, self.params.ndofs, self.params.ndofs)
        # print(resh)
        grid_axis = self.params.interval[0] + np.arange(self.params.nvars_axis) * self.dx
        xvalues, yvalues = np.meshgrid(grid_axis, grid_axis, indexing='ij')

        u = np.array([0 * xvalues + 1, 0 * xvalues])
        center = self.params.nvars_axis // 2
        u[0, center - 5: center + 5, center - 5: center + 5] = 0.5
        u[1, center - 5: center + 5, center - 5: center + 5] = 0.25
        # plt.contourf(xvalues, yvalues, u[0], 200)
        # plt.show()
        me.values = u.flatten()

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)

        grid_axis = self.params.interval[0] + np.arange(self.params.nvars_axis) * self.dx
        xvalues, yvalues = np.meshgrid(grid_axis, grid_axis, indexing='ij')

        u = np.array([np.exp(-self.params.diff_mat[0, 0] * 2 * t) * np.sin(xvalues) * np.sin(yvalues),
             np.exp(-self.params.diff_mat[1, 1] * 2 * t) * np.sin(xvalues) * np.sin(yvalues)])

        me.values = u.flatten()

        return me
