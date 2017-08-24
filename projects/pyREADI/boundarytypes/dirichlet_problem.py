import numpy as np

from projects.pyREADI.boundarytypes.b_type_base import BoundaryValueProblemBase


class DirichletProblem(BoundaryValueProblemBase):

    def __init__(self, rdmodel):
        super(DirichletProblem, self).__init__(rdmodel)

        self.non_b_mask = self._generate_dirichlet_non_b_mask()
        print(self.non_b_mask)
        self.b_mask = np.logical_not(self.non_b_mask)
        self.b_grid = [l.flatten()[self.b_mask] for l in self.rdmodel.grid]

    @staticmethod
    def compute_nnodes_dx(nvars, domlen):
        """Maps nvars to nnodes (actual number of grid nodes, including the boundary nodes
        """
        nnodes = nvars + 2
        dx = domlen / (nvars+1)
        return nnodes, dx

    def _generate_dirichlet_non_b_mask(self):
        mask_1d = np.concatenate(([False], np.ones(self.rdmodel.params.nvars, dtype=bool), [False]))
        return reduce(np.kron, [mask_1d] * self.rdmodel.params.ndims)

    def extend(self, u_red, t):
        # extends u_inner to the boundary
        u_ext = np.array([np.empty_like(self.b_mask, dtype=float)]*self.rdmodel.params.nspecies)
        u_ext[:, self.non_b_mask] = u_red
        u_ext[:, self.b_mask] = self.rdmodel.params.bc_func(self.b_grid, t)
        return u_ext

    def reduce(self, u_ext):
        return np.copy(u_ext[:, self.non_b_mask])
