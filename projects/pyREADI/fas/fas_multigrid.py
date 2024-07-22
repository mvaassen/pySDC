import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from projects.pyREADI.tools.plot_tools import eigvalue_plot_list

from projects.pyREADI.fas.multigrid_base import MultigridBase


class FASMultigrid(MultigridBase):
    """Implementation of a multigrid solver with different cycle implementations
    """

    def __init__(self, rdmodel, galerkin):
        """Initialization routine
        """
        #assert np.log2(nx+1) >= nlevels
        self.err = []
        super(FASMultigrid, self).__init__(rdmodel, galerkin)

    def mg_iter(self, v, rhs, mg_res_bound, t, nu1, nu2):
        v_ext = self._rdmodel.extend(v, t)
        residual = rhs - self.smoo[0].sysop.eval_sysop_alt_wb(v_ext)
        residual = self._rdmodel.reduce(self._rdmodel.eval_m(residual))
        #print('init residual: %e' % la.norm(residual.flatten(), np.inf))
        niter = 0

        grid = self._rdmodel.generate_grid()
        inner_grid = [l.flatten()[self._rdmodel.inv_b_mask] for l in grid]

        # w = self.do_v_cycle_bs(w, rhs, t, nu1, 10, 0, 1)
        # v_ext = self._rdmodel.extend(w, t)
        # residual = self._rdmodel.reduce(self._rdmodel.eval_m(rhs)) - self.smoo[0].sysop.eval_sysop(v_ext)
        # print('pre residual: %e' % la.norm(residual.flatten(), ord=np.inf))
        # err = (w - self._rdmodel.bc(inner_grid, t)).flatten()
        # print('pre error: %e' % la.norm(err, ord=np.inf))

        v_curr = v.copy()

        while la.norm(residual.flatten(), ord=np.inf) > mg_res_bound:
            v_prev = v_curr
            v_curr = self.do_v_cycle_bs(v_prev, rhs, t, nu1, nu2, 0, 1)
            #v_ext = self._rdmodel.extend(v, t)
            #residual = self._rdmodel.reduce(self._rdmodel.eval_m(rhs)) - self.smoo[0].sysop.eval_sysop(v_ext)
            #plt.plot(residual.flatten())

            # preconditioner residual
            residual = v_curr - v_prev

            #err = (w - self._rdmodel.bc(inner_grid, t)).flatten()

            print('residual: %e' % la.norm(residual.flatten(), ord=np.inf))
            #print('error: %e' % la.norm(err, ord=np.inf))
            #plt.show()

            niter += 1

        return v_curr

    def do_v_cycle_bs(self, v0, rhs, t, nu1, nu2, lstart, innerNewton):
        """Straightforward implementation of a V-cycle
        This can also be used inside an FMG-cycle!
        Args:
            v0 (numpy.array): initial values on finest level
            rhs (numpy.array): right-hand side on finest level
            nu1 (int): number of downward smoothing steps
            nu2 (int): number of upward smoothing steps
            lstart (int): starting level
        Returns:
            numpy.array: solution vector on finest level
        """

        assert self.nlevels >= lstart >= 0
        assert v0.shape == self.vh[lstart].shape

        # set intial conditions (note: resetting vectors here is important!)
        #self.reset_vectors(lstart)
        self.vh[lstart] = v0
        self.fh[lstart] = rhs

        # downward cycle
        for l in range(lstart, self.nlevels-1):

            # pre pre-smoothing extension
            vh_ext = self.smoo[l].sysop.rdmodel.extend(self.vh[l], t)
            tmp = self.smoo[l].sysop.rdmodel.extend(self.vh[l], t)

            # pre-smoothing
            for i in range(nu1):
                vh_ext = self.smoo[l].smooth(self.fh[l], vh_ext)

            # post pre-smoothing reduction
            self.vh[l] = self.smoo[l].sysop.rdmodel.reduce(vh_ext)

            # restrict
            self.vh[l + 1] = np.array([self.trans[l].restrict(self.vh[l][i]) for i in range(self._rdmodel.nspecies)])
            self.fh[l + 1] = np.array([self.trans[l].restrict(self.fh[l][i], with_boundary=True) for i in range(self._rdmodel.nspecies)])

            # apply tau-correction
            vh_coarse_ext = self.smoo[l + 1].sysop.rdmodel.extend(self.vh[l + 1], t)
            tau = self.smoo[l + 1].sysop.eval_sysop_alt_wb(vh_coarse_ext) - np.array(
                [self.trans[l].restrict(self.smoo[l].sysop.eval_sysop_alt_wb(vh_ext)[i], with_boundary=True) for i in
                 range(self._rdmodel.nspecies)])
            #tau = self.smoo[l + 1].sysop.eval_sysop(self.vh[l + 1]) - np.array([self.trans[l].restrict(self.smoo[l].sysop.eval_sysop(self.vh[l])[i]) for i in range(self._rdmodel.nspecies)])
            self.fh[l + 1] += tau

        # coarse grid smoothing
        vh_ext = self.smoo[-1].sysop.rdmodel.extend(self.vh[-1], t)
        self.vh[-1] = self.smoo[-1].sysop.rdmodel.reduce(self.smoo[-1].smooth(self.fh[-1], vh_ext))
        #plt.plot(vh_ext.flatten())
        #plt.show()

        # upward cycle
        for l in reversed(range(lstart, self.nlevels-1)):
            # correct
            e = self.vh[l + 1] - np.array([self.trans[l].restrict(self.vh[l][i]) for i in range(self._rdmodel.nspecies)])
            self.vh[l] += np.array([self.trans[l].interpolate(e[i]) for i in range(self._rdmodel.nspecies)])

            # pre post-smoothing extension
            vh_ext = self.smoo[l].sysop.rdmodel.extend(self.vh[l], t)

            # post-smoothing
            for i in range(nu2):
                vh_ext = self.smoo[l].smooth(self.fh[l], vh_ext)

            # post post-smoothing reduction (yay, permutations)
            self.vh[l] = self.smoo[l].sysop.rdmodel.reduce(vh_ext)

        return self.vh[lstart]

    def do_v_cycle(self, v0, rhs, nu1, nu2, lstart, innerNewton):
        """Straightforward implementation of a V-cycle
        This can also be used inside an FMG-cycle!
        Args:
            v0 (numpy.array): initial values on finest level
            rhs (numpy.array): right-hand side on finest level
            nu1 (int): number of downward smoothing steps
            nu2 (int): number of upward smoothing steps
            lstart (int): starting level
        Returns:
            numpy.array: solution vector on finest level
        """

        assert self.nlevels >= lstart >= 0
        assert v0.shape == self.vh[lstart].shape

        # set intial conditions (note: resetting vectors here is important!)
        #self.reset_vectors(lstart)
        self.vh[lstart] = v0
        self.fh[lstart] = rhs

        # downward cycle
        for l in range(lstart, self.nlevels-1):
            # pre-smoothing
            for i in range(nu1):
                self.vh[l] = self.smoo[l].smooth(self.fh[l], self.vh[l])
            # restrict
            self.vh[l + 1] = self.trans[l].restrict(self.vh[l])
            self.fh[l + 1] = self.trans[l].restrict(self.fh[l])
            # apply tau-correction
            tau = self.smoo[l + 1].sysop.eval_sysop(self.vh[l + 1]) - self.trans[l].restrict(self.smoo[l].sysop.eval_sysop(self.vh[l]))
            self.fh[l+1] += tau

        # coarse grid smoothing
        self.vh[-1] = self.smoo[-1].smooth(self.fh[-1], self.vh[-1])

        # upward cycle
        for l in reversed(range(lstart, self.nlevels-1)):
            # correct
            self.vh[l] += self.trans[l].interpolate(self.vh[l + 1] - self.trans[l].restrict(self.vh[l]))
            # post-smoothing
            for i in range(nu2):
                self.vh[l] = self.smoo[l].smooth(self.fh[l], self.vh[l])

        return self.vh[lstart]

    def do_fmg_cycle_recursive_bs_init(self, rhs, guess, nu0, nu1, nu2, level, innerNewton):
        """Recursive implementation of an FMG-cycle
                Args:
                    v0 (numpy.array): initial values on finest level
                    rhs (numpy.array): right-hand side on finest level
                    nu1 (int): number of downward smoothing steps
                    nu2 (int): number of upward smoothing steps
                    level (int): current level
                Returns:
                    numpy.array: solution vector on current level
                """

        assert self.nlevels > level >= 0

        # set intial conditions
        self.fh[level] = rhs

        # downward cycle
        if level < self.nlevels - 1:

            # restrict
            self.fh[level + 1] = np.array([self.trans[level].restrict(self.fh[level][i]) for i in range(self._rdmodel.ncomps)])

            # guess test
            guess_coarse = np.array([self.trans[level].restrict(guess[i]) for i in range(self._rdmodel.ncomps)])

            # recursive call to fmg-cycle
            self.vh[level + 1] = self.do_fmg_cycle_recursive_bs_init(self.fh[level + 1], guess_coarse, nu0, nu1, nu2, level + 1, innerNewton)
        # on coarsest level
        else:
            # 'solve' on coarsest level
            self.vh[level] = self.smoo[level].smooth(self.fh[level], guess)
            #print(self.vh[level])
            #print(guess)
            #exit()
            return self.vh[level]

        # correct
        self.vh[level] = np.array([self.trans[level].interpolate(self.vh[level + 1][i]) for i in range(self._rdmodel.ncomps)])

        # v-cycles
        for i in range(nu0):
            self.vh[level] = self.do_v_cycle_bs(self.vh[level], self.fh[level], nu1, nu2, level, innerNewton)

        return self.vh[level]

    def do_fmg_cycle_recursive_bs(self, rhs, nu0, nu1, nu2, level, innerNewton):
        """Recursive implementation of an FMG-cycle
                Args:
                    v0 (numpy.array): initial values on finest level
                    rhs (numpy.array): right-hand side on finest level
                    nu1 (int): number of downward smoothing steps
                    nu2 (int): number of upward smoothing steps
                    level (int): current level
                Returns:
                    numpy.array: solution vector on current level
                """

        assert self.nlevels > level >= 0

        # set intial conditions
        self.fh[level] = rhs

        # downward cycle
        if level < self.nlevels - 1:

            # restrict
            self.fh[level + 1] = np.array([self.trans[level].restrict(self.fh[level][i]) for i in range(self._rdmodel.ncomps)])

            # recursive call to fmg-cycle
            self.vh[level + 1] = self.do_fmg_cycle_recursive_bs(self.fh[level + 1], nu0, nu1, nu2, level + 1, innerNewton)
        # on coarsest level
        else:
            # 'solve' on coarsest level
            self.vh[level] = self.smoo[level].smooth(self.fh[level], self.vh[level])
            return self.vh[level]

        # correct
        self.vh[level] = np.array([self.trans[level].interpolate(self.vh[level + 1][i]) for i in range(self._rdmodel.ncomps)])

        # v-cycles
        for i in range(nu0):
            self.vh[level] = self.do_v_cycle_bs(self.vh[level], self.fh[level], nu1, nu2, level, innerNewton)

        return self.vh[level]

    # TODO
    def do_fmg_cycle_recursive(self, rhs, nu0, nu1, nu2, level, innerNewton):
        """Recursive implementation of an FMG-cycle
                Args:
                    v0 (numpy.array): initial values on finest level
                    rhs (numpy.array): right-hand side on finest level
                    nu1 (int): number of downward smoothing steps
                    nu2 (int): number of upward smoothing steps
                    level (int): current level
                Returns:
                    numpy.array: solution vector on current level
                """

        assert self.nlevels > level >= 0

        # set intial conditions
        self.fh[level] = rhs

        # downward cycle
        if level < self.nlevels - 1:

            # restrict
            self.fh[level + 1] = self.trans[level].restrict(self.fh[level])

            # recursive call to fmg-cycle
            self.vh[level + 1] = self.do_fmg_cycle_recursive(self.fh[level + 1], nu0, nu1, nu2, level + 1, innerNewton)
        # on coarsest level
        else:
            # 'solve' on coarsest level
            self.vh[level] = self.smoo[level].smooth(self.fh[level], self.vh[level])
            return self.vh[level]

        # correct
        self.vh[level] = self.trans[level].interpolate(self.vh[level + 1])

        # v-cycles
        for i in range(nu0):
            self.vh[level] = self.do_v_cycle(self.vh[level], self.fh[level], nu1, nu2, level, innerNewton)

        return self.vh[level]
