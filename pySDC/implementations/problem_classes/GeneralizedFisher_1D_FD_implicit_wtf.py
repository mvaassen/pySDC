from __future__ import division

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError


# noinspection PyUnusedLocal
class generalized_fisher_wtf(ptype):
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
        super(generalized_fisher_wtf, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

        # compute dx and get discretization matrix A
        self.dx = (self.params.interval[1] - self.params.interval[0]) / (self.params.nvars + 1)

        #self.A = self.__get_A(self.params.nvars, self.dx)
        self.M, self.A, self.L = self.__get_A_cmpct(self.params.nvars, self.dx)


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
        Ma1d = sp.csr_matrix(1.0/(dx ** 2) * a1d)

        bla = np.zeros(N + 2)
        bla[0:4] = np.array([13., -27., 15., -1.])
        a1d = sp.vstack([bla, a1d, bla[::-1]])
        # print(a1d.todense())
        a1d *= 1.0 / (dx ** 2)

        a1d = sp.csr_matrix(a1d)

        stencil = [1. / 10, 1, 1. / 10]
        diags = [-1, 0, 1]
        M = sp.diags(stencil, diags, shape=(N + 2, N + 2), format='lil')[1:-1, :]
        MM = sp.csr_matrix(M)

        bla = np.zeros(N + 2)
        bla[0:2] = np.array([1., 11.])
        M = sp.vstack([bla, M, bla[::-1]])
        # print(M.todense())

        M = sp.csr_matrix(M)

        return MM, Ma1d, sla.inv(M).dot(a1d)

    def extend(self, u0, t):
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

        uext = np.concatenate(([ul], u.values, [ur]))

        bla = self.dtype_u(init=uext.shape[0])
        bla.values = uext

        return bla

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
        #uext = np.concatenate(([ul], u.values, [ur]))
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            uext = np.concatenate(([ul], u.values, [ur]))
            g = self.M.dot(uext) - \
                factor * (self.A.dot(uext) + self.M.dot(lambda0 ** 2 * uext * (1 - uext ** nu))) - self.M.dot(rhs.values)

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            #plt.ylim(ymax=1e-6, ymin=-1e-6)
            #plt.plot(g)
            #plt.pause(1)

            if res < self.params.newton_tol:
                break

            # assemble dg
            #m_diag = sp.diags(np.diagonal(self.M))
            dg = self.M[:, 1:-1] - factor * \
                (self.A[:, 1:-1] + self.M[:, 1:-1].dot(sp.diags(lambda0 ** 2 - lambda0 ** 2 * (nu + 1) * u.values ** nu, offsets=0)))


            import scipy.linalg as la

            #print(la.norm(self.M.dot(self.L).todense()-self.A.todense(), np.inf))
            # print(self.A.todense().dtype)
            # print('bla', la.norm(self.A.todense(), 2))
            # print('mult', la.norm(self.M.dot(self.L).todense(), 2))
            # print('err', la.norm((self.M.dot(self.L) - self.A).todense(), 2))
            #print(self.A.todense())

            # newton update: u1 = u0 - g/dg
            tmp = np.concatenate(([ul], u.values - spsolve(dg, g), [ur]))
            res = self.M.dot(tmp) - \
                factor * (self.A.dot(tmp) + self.M.dot(lambda0 ** 2 * tmp * (1 - tmp ** nu))) - self.M.dot(rhs.values)
            res2 = tmp - \
                  factor * (self.L.dot(tmp) + lambda0 ** 2 * tmp * (1 - tmp ** nu)) - rhs.values

            u.values -= spsolve(dg, g)
            res3 = rhs.values - (tmp - factor*self.eval_f(u, t).values)

            #print(n)
            #plt.plot(res+self.M.dot(res3))
            #plt.plot(-res2)
            #plt.plot(-self.M.dot(res3))
            #plt.pause(1)
            #plt.show()
            #plt.close()

            #uext = tmp

            #u.values -= spsolve(dg, g)

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
        f.values = self.L.dot(uext) + self.params.lambda0 ** 2 * uext * (1 - uext ** self.params.nu)
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
