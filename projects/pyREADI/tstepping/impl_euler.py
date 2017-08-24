from fas.fas_multigrid import FASMultigrid
from fas.linear_transfer_nd import LinearTransferND
from fas.nonlinear_gauss_seidel import NonlinearGaussSeidel
from rdmodels.rdmodel_base import RDModelBase
from fas.system_operator import SystemOperator
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time


class ImplicitEuler(object):

    def __init__(self, rdmodel):
        assert isinstance(rdmodel, RDModelBase)

        self._rdmodel = rdmodel

        # Initialize FAS multigrid solver
        self._mg = FASMultigrid(self._rdmodel, True)
        self._mg.attach_transfer(LinearTransferND)

    def coarsening(self, nx, dt):
        # Dirty, please fix me
        lin = self._rdmodel.l_matrix(nx)
        return SystemOperator([sp.eye(lin[i].shape[0]) - dt * lin[i] for i in range(self._rdmodel.nspecies)],
                              lambda x: -dt * self._rdmodel.eval_n(x),
                              lambda x: -dt * self._rdmodel.eval_n_jac(x))

    def solve(self, t_interval, dt, u0):
        # Diagnostics
        itercount = []
        residuals = []
        cpu_time = []

        # Set up system operator according to the implicit Euler timestepping scheme
        # (I - dt*L)u_new - dt*N(u_new) = u_old
        sysop = SystemOperator([sp.eye(self._rdmodel.L[i].shape[0]) - dt * self._rdmodel.L[i] for i in range(self._rdmodel.nspecies)],
                               lambda x: -dt * self._rdmodel.eval_n(x),
                               lambda x: -dt * self._rdmodel.eval_n_jac(x))

        # Attach FAS smoother and system operator
        self._mg.attach_smoother(NonlinearGaussSeidel, sysop, lambda nx: self.coarsening(nx, dt))

        t = t_interval[0]
        u = u0

        # Tolerance for last time step
        ttol = 1e-16 * dt

        while t < t_interval[1]:
            tn = t + dt

            if tn > t_interval[1] - ttol:
                # if upper time bound (te) exceeded fit last time step to te
                tn = t_interval[1]
                dt = tn - t

            # Here comes Multigrid
            #un_ex = np.array([spla.spsolve(sysop.lin[0], u[0])])
            #un_ex = opt.fsolve(lambda x: sysop.eval_sysop(x) - u, u)
            #print(un_ex)

            nvcycles = 0

            un = u
            init_res = (sysop.eval_sysop(un) - u).flatten()
            res = init_res
            #err = (un - un_ex).flatten()
            print('init residual: %e' % la.norm(init_res, 2))
            #print('init err: %e' % la.norm(err, np.inf))
            #un = self._mg.do_fmg_cycle_recursive_bs(u, 1, 1, 1, 0, 1)
            #res = (sysop.eval_sysop(un) - u).flatten()
            #print('residual (post FMG): %e' % la.norm(res, np.inf))
            #residuals.append(la.norm(res, ord=2)/la.norm(init_res, 2))
            start = time.time()
            #while la.norm(res, ord=2) > 1e-8 * la.norm(init_res, 2):
            while la.norm(res, ord=np.inf) > 1e-6:
                nvcycles += 1

                un = self._mg.do_v_cycle_bs(un, u, 1, 1, 0, 1)
                res = (sysop.eval_sysop(un) - u).flatten()
                #err = (un - un_ex).flatten()
                print('residual: %e' % la.norm(res, ord=np.inf))
                #print('err: %e' % la.norm(err, np.inf))

                #residuals.append(la.norm(res, ord=2) / la.norm(init_res, 2))

            cpu_time.append(time.time() - start)
            itercount.append(nvcycles)

            t = tn
            u = un

        return u

