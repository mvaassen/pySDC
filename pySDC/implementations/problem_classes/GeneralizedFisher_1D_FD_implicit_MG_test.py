from __future__ import division

import sys
sys.path.append('/home/zam/vaassen/PycharmProjects/pyRD/')

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError

from projects.pyREADI.rdmodels.generalized_fisher_1d import GeneralizedFisher1D
from projects.pyREADI.tstepping.impl_euler_new import ImplEulerScheme
from projects.pyREADI.tstepping.implicit_timestepper import ImplicitTimeIntegrator
from projects.pyREADI.tstepping.fwd_euler import solve_t
from projects.pyREADI.fdschemes.lap_4th_cpact_1d import Lap4thCpact1D
from projects.pyREADI.boundarytypes.dirichlet_problem import DirichletProblem
from projects.pyREADI.fas.fas_multigrid import FASMultigrid
from projects.pyREADI.fas.linear_transfer_nd import LinearTransferND
from projects.pyREADI.fas.nonlinear_gauss_seidel import NonlinearGaussSeidel

# from rdmodels.generalized_fisher_1d import GeneralizedFisher1D
# from tstepping.impl_euler_new import ImplEulerScheme
# from tstepping.implicit_timestepper import ImplicitTimeIntegrator
# from tstepping.fwd_euler import solve_t
# from fdschemes.lap_4th_cpact_1d import Lap4thCpact1D
# from boundarytypes.dirichlet_problem import DirichletProblem
# from fas.fas_multigrid import FASMultigrid
# from fas.linear_transfer_nd import LinearTransferND
# from fas.nonlinear_gauss_seidel import NonlinearGaussSeidel


# noinspection PyUnusedLocal
class generalized_fisher_MG(ptype):
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
        essential_keys = ['nvars', 'nu', 'lambda0', 'mg_maxiter', 'mg_restol', 'interval']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars'] + 1) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p - 1')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(generalized_fisher_MG, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

        # compute dx and get discretization matrix A
        self.dx = (self.params.interval[1] - self.params.interval[0]) / (self.params.nvars + 1)
        self.A = self.__get_A(self.params.nvars, self.dx)

        # cast model into my own data structure
        self.rdmodel = GeneralizedFisher1D(self.params.nvars, self.params.interval, DirichletProblem, None, 1, np.array([1]), self.params.lambda0, self.params.nu)

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
        A = sp.diags(stencil, [-1, 0, 1], shape=(N + 2, N + 2), format='lil')
        A *= 1.0 / (dx ** 2)

        return A

    # noinspection PyTypeChecker
    def solve_system(self, rhs, factor, u0, t):
        u = self.dtype_u(u0)
        self.factor = factor

        u.values, = self._mg.mg_iter(np.array([u0.values]), np.array([rhs.values]), self.params.mg_restol, t, nu1=1, nu2=1)

        return u

    # noinspection PyTypeChecker
    def solve_system_newton(self, rhs, factor, u0, t):
        """
        Simple Newton solver

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (required here for the BC)

        Returns:
            dtype_u: solution u
        """

        u = self.dtype_u(u0)

        nu = self.params.nu
        lambda0 = self.params.lambda0

        # set up boundary values to embed inner points
        lam1 = lambda0 / 2.0 * ((nu / 2.0 + 1) ** 0.5 + (nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1 ** 2 - lambda0 ** 2)
        ul = (1 + (2 ** (nu / 2.0) - 1) *
              np.exp(-nu / 2.0 * sig1 * (self.params.interval[0] + 2 * lam1 * t))) ** (-2.0 / nu)
        ur = (1 + (2 ** (nu / 2.0) - 1) *
              np.exp(-nu / 2.0 * sig1 * (self.params.interval[1] + 2 * lam1 * t))) ** (-2.0 / nu)

        # start newton iteration
        n = 0
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            uext = np.concatenate(([ul], u.values, [ur]))
            g = u.values - \
                factor * (self.A.dot(uext)[1:-1] + lambda0 ** 2 * u.values * (1 - u.values ** nu)) - rhs.values

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = sp.eye(self.params.nvars) - factor * \
                (self.A[1:-1, 1:-1] + sp.diags(lambda0 ** 2 - lambda0 ** 2 * (nu + 1) * u.values ** nu, offsets=0))

            # newton update: u1 = u0 - g/dg
            u.values -= spsolve(dg, g)

            # increase iteration count
            n += 1

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
        f = self.dtype_f(self.init)
        uext = self.rdmodel.extend(np.array([u.values]), t)
        f.values, = self.rdmodel.eval_rhs_wb(uext)
        return f

    def eval_f_newton(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        # set up boundary values to embed inner points
        lam1 = self.params.lambda0 / 2.0 * ((self.params.nu / 2.0 + 1) ** 0.5 + (self.params.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1 ** 2 - self.params.lambda0 ** 2)
        ul = (1 + (2 ** (self.params.nu / 2.0) - 1) *
              np.exp(-self.params.nu / 2.0 * sig1 * (self.params.interval[0] + 2 * lam1 * t))) ** (-2 / self.params.nu)
        ur = (1 + (2 ** (self.params.nu / 2.0) - 1) *
              np.exp(-self.params.nu / 2.0 * sig1 * (self.params.interval[1] + 2 * lam1 * t))) ** (-2 / self.params.nu)

        uext = np.concatenate(([ul], u.values, [ur]))

        f = self.dtype_f(self.init)
        f.values = self.A.dot(uext)[1:-1] + self.params.lambda0 ** 2 * u.values * (1 - u.values ** self.params.nu)
        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)
        xvalues = np.array([(i + 1 - (self.params.nvars + 1) / 2) * self.dx for i in range(self.params.nvars)])

        lam1 = self.params.lambda0 / 2.0 * ((self.params.nu / 2.0 + 1) ** 0.5 + (self.params.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1 ** 2 - self.params.lambda0 ** 2)
        me.values = (1 + (2 ** (self.params.nu / 2.0) - 1) *
                     np.exp(-self.params.nu / 2.0 * sig1 * (xvalues + 2 * lam1 * t))) ** (-2.0 / self.params.nu)
        return me
