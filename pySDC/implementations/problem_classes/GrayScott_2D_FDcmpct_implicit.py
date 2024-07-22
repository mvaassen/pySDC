from __future__ import division

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError

from time import time
import projects.pyREADI.tests.module_diagnostics as diagn

class GmresCounter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            #print('iter %3i\trk = %s' % (self.niter, str(rk)))
            print('iter %3i\trk = %s' % (self.niter, np.linalg.norm(rk, np.inf)))


# noinspection PyUnusedLocal
class GrayScott(ptype):
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
        essential_keys = ['nvars', 'nvars_axis', 'diff_mat', 'feedrate', 'killrate', 'newton_maxiter', 'newton_tol', 'interval', 'expl_boundary']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars_axis']) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(GrayScott, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

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

        # expl = self.solve_t([0, 200], 1, rhs.values)
        # grid_axis = self.params.interval[0] + np.arange(self.params.nvars_axis) * self.dx
        # xvalues, yvalues = np.meshgrid(grid_axis, grid_axis, indexing='ij')
        # plt.contourf(xvalues, yvalues, expl.reshape((2, self.params.nvars_axis, self.params.nvars_axis))[0], 200)
        # plt.colorbar()
        # plt.show()

        # Logging
        start = time()
        counter = GmresCounter()
        # End Logging

        u = self.dtype_u(u0)

        # dg = sp.eye(self.params.nvars) - factor * self.A
        #
        # me = self.dtype_u(self.init)
        # me.values = spsolve(dg, rhs.values)
        #
        # walltime = time() - start
        # diagn.diagnostics_dict['newton_wtime'].append(walltime)
        #
        # return me

        # grid_axis = self.params.interval[0] + np.arange(self.params.nvars_axis) * self.dx
        # xvalues, yvalues = np.meshgrid(grid_axis, grid_axis, indexing='ij')
        # plt.contourf(xvalues, yvalues, me.values.reshape((self.params.nvars_axis, self.params.nvars_axis)), 200)
        # plt.colorbar()
        # plt.show()

        # Start init residual

        # if g is close to 0, then we are done
        uext = u.values
        u_species = uext.reshape((2, self.params.nvars // 2))

        g = u.values - factor * (self.A.dot(uext) + np.array(
            [-u_species[0] * u_species[1] ** 2 + self.params.feedrate * (1 - u_species[0]),
             u_species[0] * u_species[1] ** 2 - (self.params.feedrate + self.params.killrate) * u_species[
                 1]]).flatten()) - rhs.values

        res0_newton = np.linalg.norm(g, np.inf)
        # End init residual

        # start newton iteration
        n = 0
        while n < self.params.newton_maxiter:

            # form the function g with g(u) = 0
            uext = u.values
            u_species = uext.reshape((2, self.params.nvars // 2))

            g = u.values - factor * (self.A.dot(uext) + np.array(
                [-u_species[0] * u_species[1] ** 2 + self.params.feedrate * (1 - u_species[0]),
                 u_species[0] * u_species[1] ** 2 - (self.params.feedrate + self.params.killrate) * u_species[
                     1]]).flatten()) - rhs.values

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            #plt.ylim(ymax=1e-6, ymin=-1e-6)
            #plt.plot(g)
            #plt.show()
            #plt.pause(1)
            #plt.close()

            if res < self.params.newton_tol * res0_newton:
                break

            # assemble dg
            dg = sp.eye(self.params.nvars) - factor * self.A

            # newton update: u1 = u0 - g/dg
            #u.values -= spsolve(dg, g)

            # lu = sla.splu(sp.csc_matrix(dg))
            # u.values -= lu.solve(g)
            # counter.niter += 1

            # CG
            update, info = sla.cg(dg, g, tol=1e-8, callback=counter)
            if info != 0:
                print('Warning! Something went wrong in CG')
                exit()
            u.values -= update
            # End CG

            # GMRES
            # update, info = sla.lgmres(dg, g, tol=1e-11, maxiter=1000000, callback=counter)
            # if info != 0:
            #     print('Warning! Something went wrong in GMRES')
            #     #exit()
            # u.values -= update
            # End GMRES

            # increase iteration count
            n += 1

        print(n)
        print('iter %3i' % counter.niter)
        walltime = time() - start
        diagn.diagnostics_dict['newton_wtime'].append(walltime)
        diagn.diagnostics_dict['inner_niter'].append(counter.niter)

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

    def u_init_(self):
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
        lo = int(0.4*(self.params.nvars_axis // 2))
        hi = int(0.6*(self.params.nvars_axis // 2))
        domlen = self.params.interval[1] -  self.params.interval[0]
        u[0, center - lo: center + hi, center - lo: center + hi] = 0.5
        u[1, center - lo: center + hi, center - lo: center + hi] = 0.25
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
