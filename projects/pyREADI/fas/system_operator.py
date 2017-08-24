from projects.pyREADI.fas.transfer_base import TransferBase
import numpy as np
import scipy.sparse as sp


class SystemOperator(object):

    def __init__(self, lin, nonlin, nonlin_jac):
        self.lin = lin
        self.nonlin = nonlin
        self.nonlin_jac = nonlin_jac

    def eval_sysop(self, u):

        return np.array([self.lin[i].dot(u[i]) for i in range(len(self.lin))]) + self.nonlin(u)

    def eval_sysop_i(self, u, i):
        return np.array([self.lin[j][i, :].dot(u[j]) for j in range(len(self.lin))]).flatten() + self.nonlin(u[:, i])

    def eval_jac_i(self, u, i):
        diag = np.zeros((len(self.lin), len(self.lin)))
        for j in range(diag.shape[0]):
            diag[j, j] = self.lin[j][i, i]
        return diag + self.nonlin_jac(u[:, i])

    def get_coarse_sysop(self, transfer):
        assert isinstance(transfer, TransferBase)

        # Here comes Galerkin
        return SystemOperator([transfer.restriction_matrix.dot(self.lin[i].dot(transfer.interpolation_matrix)) for i in range(len(self.lin))], self.nonlin, self.nonlin_jac)
