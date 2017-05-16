# coding=utf-8
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt

from fas.smoother_base import SmootherBase
from fas.system_operator import SystemOperator


class NonlinearGaussSeidel(SmootherBase):
    """Implementation of the scalar newton iteration

    Attributes:
        A : system matrix
        gamma: gamma
    """

    def __init__(self, sysop, *args, **kwargs):
        """Initialization routine for the smoother

        Args:
            A: matrix A of the system to solve
            gamma: non-linear part
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """

        super(NonlinearGaussSeidel, self).__init__(sysop, *args, **kwargs)

    def smooth_(self, rhs, u_old):
        """
        DEBUG VERSION
        """
        u_new = u_old.copy()

        non_boundary_indices = np.arange(u_new.shape[1])[self.sysop.rdmodel.inv_b_mask]

        m_rhs = self.sysop.rdmodel.eval_m(rhs)

        g = self.sysop.eval_sysop(u_new) - self.sysop.rdmodel.reduce(m_rhs)

        # if g is close to 0, then we are done
        res = np.linalg.norm(g.flatten(), np.inf)


        # assemble dg
        # m_diag = sp.diags(np.diagonal(self.M))
        dg = self.sysop.rdmodel.M_matrix[0][1:-1, 1:-1] - self.sysop.dt(None) * \
                               (self.sysop.rdmodel.L_matrix[0][1:-1, 1:-1] + self.sysop.rdmodel.M_matrix[0][1:-1, 1:-1].dot(
                                   sp.diags(self.sysop.rdmodel.eval_n_jac(self.sysop.rdmodel.reduce(u_new))[0, 0], offsets=0)))

        #print(self.sysop.rdmodel.eval_n_jac(self.sysop.rdmodel.reduce(u_new))[0, 0])

        # newton update: u1 = u0 - g/dg
        u_new[:, non_boundary_indices] -= np.array([sla.spsolve(dg, g.flatten())])

        #plt.plot(-(self.sysop.eval_sysop(u_new) - self.sysop.rdmodel.reduce(m_rhs)).flatten())
        #plt.show()

        return u_new

    def smooth(self, rhs, u_old):
        """
        Routine to perform a smoothing step

        Args:
            rhs (numpy.ndarray): the right-hand side vector, size
                :attr:`pymg.problem_base.ProblemBase.ndofs`
            u_old (numpy.ndarray): the initial value for this step, size
                :attr:`pymg.problem_base.ProblemBase.ndofs`

        Returns:
            numpy.ndarray: the smoothed solution u_new of size
                :attr:`pymg.problem_base.ProblemBase.ndofs`
        """
        u_new = u_old.copy()

        m_rhs = self.sysop.rdmodel.eval_m(rhs)

        non_boundary_indexes = np.arange(u_new.shape[1])[self.sysop.rdmodel.inv_b_mask]

        for i in range(len(non_boundary_indexes)):
            jac = self.sysop.eval_jac_i(u_new, non_boundary_indexes[i])
            #print(jac)
            #exit()
            #test = self.sysop.eval_sysop(u_new)[:, i]
            #print(test)
            #print(self.sysop.eval_sysop_i(u_new, i).flatten())
            #if i == 3:
            #    exit()
            #print(test)
            # flatten is a workaround, might cause problems
            #u_new[:, non_boundary_indexes[i]] -= la.solve(jac, self.sysop.eval_sysop_i(u_new, non_boundary_indexes[i]) - m_rhs[:, non_boundary_indexes[i]])
            u_new[:, non_boundary_indexes[i]] -= (self.sysop.eval_sysop_i(u_new, non_boundary_indexes[i]) - m_rhs[:, non_boundary_indexes[i]])/jac.flatten()

        return u_new
