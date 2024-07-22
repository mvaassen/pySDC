from __future__ import division

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError


# noinspection PyUnusedLocal
class generalized_fisher(ptype):
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
        essential_keys = ['nvars', 'nu', 'lambda0', 'newton_maxiter', 'newton_tol', 'interval', 'expl_boundary']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars'] + 1) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p - 1')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(generalized_fisher, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

        # compute dx and get discretization matrix A
        self.dx = (self.params.interval[1] - self.params.interval[0]) / (self.params.nvars + 1)

        #self.A = self.__get_A(self.params.nvars, self.dx)
        if self.params.expl_boundary:
            print('hello')
            #self.A = self.__get_A_cmpct_expl_boundary(self.params.nvars, self.dx)
        else:
            print('hello')
            self.A = self.__get_A_cmpct(self.params.nvars, self.dx)


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

    @staticmethod
    def __get_A_cmpct_expl_boundary(N, dx):
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
        bla[0:5] = np.array([35./12, -26./3, 19./2, -14./3, 11./12])
        a1d = sp.vstack([bla, a1d, bla[::-1]])
        # print(a1d.todense())
        a1d *= 1.0 / (dx ** 2)
        a1d = sp.csr_matrix(a1d)

        stencil = [1. / 10, 1, 1. / 10]
        diags = [-1, 0, 1]
        M = sp.diags(stencil, diags, shape=(N + 2, N + 2), format='lil')[1:-1, :]
        bla = np.zeros(N + 2)
        bla[0] = 1.
        M = sp.vstack([bla, M, bla[::-1]])
        # print(M.todense())
        M = sp.csr_matrix(M)

        bla = sla.inv(M)[1:-1, 1:-1].dot(a1d[1:-1, 1:-1])

        return sla.inv(M).dot(a1d)

    # noinspection PyTypeChecker
    def solve_system(self, rhs, factor, u0, t):
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

            #plt.ylim(ymax=1e-6, ymin=-1e-6)
            #plt.plot(g)
            #plt.show()
            #plt.pause(1)
            #plt.close()

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
