import numpy as np

from boundarytypes.i_bvp import BoundaryValueProblemBase


class DirichletProblem(BoundaryValueProblemBase):

    def __init__(self, rdmodel):
        super(DirichletProblem, self).__init__(rdmodel)

    def extend(self, u_red, t):
        # extends u_inner to the boundary
        u_ext = np.array([np.empty_like(self.rdmodel.b_mask, dtype=float)]*self.rdmodel.nspecies)
        u_ext[:, self.rdmodel.inv_b_mask] = u_red
        u_ext[:, self.rdmodel.b_mask] = self.rdmodel.bc(self.rdmodel.b_grid, t)
        return u_ext

    def reduce(self, u_ext):
        return np.copy(u_ext[:, self.rdmodel.inv_b_mask])
