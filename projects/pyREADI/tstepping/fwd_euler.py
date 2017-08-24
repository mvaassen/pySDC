import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__author__ = 'vaassen'


def solve_t(rhs, t_interval, dt, u0):
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

        # Evolution forward in time
        #un = u + dt*rhs(u)
        u_ext = rhs.extend(u, t)
        #un = u + dt*rhs.eval_rhs(u_ext)
        b = rhs.eval_m(u_ext) + dt * rhs.eval_m_rhs(u_ext)
        un = [spla.spsolve(rhs.M_matrix[0], b[i])[1:-1] for i in range(1)]

        t = tn
        u = un

        # Test
        #u_hat[0, ind] = 0
        #u_hat[1, ind] = 0

    return u
