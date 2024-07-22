from projects.pyREADI.fas.transfer_base import TransferBase
#from projects.pyREADI.rdmodels.rdmodel_base import RDModelBase
import numpy as np
import scipy.sparse.linalg as spla


class ImplEulerScheme(object):

    def __init__(self, rdmodel, dt):
        from projects.pyREADI.rdmodels.rdmodel_base import RDModelBase
        assert isinstance(rdmodel, RDModelBase)

        self.rdmodel = rdmodel
        self.dt = dt

        # (M - dt*B)
        #self.lin = rdmodel.M_matrix - self.dt(None) * rdmodel.L_matrix

    def eval_sysop(self, u_ext):
        return self.rdmodel.reduce(self.rdmodel.eval_m(u_ext) - self.dt(None)*self.rdmodel.eval_m_rhs(u_ext))

    def eval_sysop_alt(self, u_ext):
        return self.rdmodel.reduce(u_ext) - self.dt(None)*self.rdmodel.eval_rhs(u_ext)

    def eval_sysop_alt_wb(self, u_ext):
        return u_ext - self.dt(None)*self.rdmodel.eval_rhs_wb(u_ext)

    def eval_sysop_i(self, u, i):
        #return np.array([self.lin[j][i, :].dot(u[j]) - self.dt(None)*self.rdmodel.M_matrix[j][i, :].dot(self.rdmodel.eval_n(u[j])) for j in range(self.rdmodel.nspecies)]).flatten()
        #print(self.rdmodel.eval_m_i(u, i))
        #print(self.rdmodel.eval_m_i_lil(u, i))
        #exit()
        #return self.rdmodel.eval_m_i(u, i) - self.dt(None)*self.rdmodel.eval_l_i(u, i) - self.dt(None)*self.rdmodel.eval_m_n_i(u, i)

        return self.rdmodel.eval_m_i_lil(u, i) - self.dt(None) * self.rdmodel.eval_l_i_lil(u, i) - self.dt(None) * self.rdmodel.eval_m_n_i_lil(u, i)

    def eval_sysop_i_alt(self, u, i):
        return self.eval_sysop_alt(u)[:, i]

    def eval_jac_i(self, u, i):
        diag = np.zeros((self.rdmodel.nspecies, self.rdmodel.nspecies))
        for j in range(diag.shape[0]):
            diag[j, j] = self.rdmodel.M_matrix[j][i, i] - self.dt(None)*self.rdmodel.L_matrix[j][i, i]  #self.lin[j][i, i]
        #print(self.dt(None) * self.rdmodel.M_matrix[0][i, i] * self.rdmodel.eval_n_jac(u[:, i]))
        return diag - self.dt(None)*self.rdmodel.M_matrix[0][i, i]*self.rdmodel.eval_n_jac(u[:, i])  # might need fixing

    def coarsen(self, ndofs_coarse):
        return ImplEulerScheme(self.rdmodel.coarsen(ndofs_coarse), self.dt)
