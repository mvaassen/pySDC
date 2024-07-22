from projects.pyREADI.fas.fas_multigrid import FASMultigrid
from projects.pyREADI.fas.linear_transfer_nd import LinearTransferND
from projects.pyREADI.fas.nonlinear_gauss_seidel import NonlinearGaussSeidel
from projects.pyREADI.rdmodels.rdmodel_base import RDModelBase
from projects.pyREADI.fas.system_operator import SystemOperator
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time


class ImplicitTimeIntegrator(object):

    def __init__(self, scheme_class, rdmodel):
        assert isinstance(rdmodel, RDModelBase)

        self._dt = None

        self.scheme_class = scheme_class
        self._rdmodel = rdmodel

        # Initialize FAS multigrid solver
        self._mg = FASMultigrid(self._rdmodel, galerkin=False)
        self._mg.attach_transfer(LinearTransferND)

    def _perform_step(self, u_start, rhs, mg_res_bound, t0, t, nu1=1, nu2=1):
        return self._mg.mg_iter(u_start, rhs, mg_res_bound, t0, t, nu1, nu2)

    def solve(self, t_interval, dt, u0, mg_res_bound=1e-12):

        # Diagnostics
        itercount = []
        residuals = []
        cpu_time = []

        # Set up fine grid system operator according to specified scheme and model
        sysop = self.scheme_class(self._rdmodel, lambda _: dt)

        # Attach FAS smoother and fine grid system operator
        # MG is responsible for the hierarchy setup
        self._mg.attach_smoother(NonlinearGaussSeidel, sysop)

        # Hang on. This is just boring setup
        t = t_interval[0]
        u = u0

        # Tolerance for last time step
        ttol = 1e-16 * dt

        # Actual timestepping begins
        # Timestepper loop from t_start to t_end
        while t < t_interval[1]:
            tn = t + dt

            if tn > t_interval[1] - ttol:
                # if upper time bound (t_end) exceeded fit last time step to t_end
                tn = t_interval[1]
                dt = tn - t

            # u of last time step
            rhs = u

            # initial value for the iteration
            un = self._perform_step(u, rhs, mg_res_bound, t, tn)

            t = tn
            u = un

        return u

