import numpy as np

from pySDC.core.Sweeper import sweeper


class generic_implicit_compact_alt(sweeper):
    """
    Generic implicit sweeper, expecting lower triangular matrix type as input

    Attributes:
        QI: lower triangular matrix
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if 'QI' not in params:
            params['QI'] = 'IE'

        # call parent's initialization routine
        super(generic_implicit_compact_alt, self).__init__(params)

        # get QI matrix
        self.QI = self.get_Qdelta_implicit(self.coll, qd_type=self.params.QI)

    def myintegrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init + 2, val=0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1] += L.dt * self.coll.Qmat[m, j] * L.f[j]

        return me

    def integrate(self):

        L = self.level
        P = L.prob

        bla = self.myintegrate()

        me = []

        # integrate RHS over all collocation nodes
        for m in range(0, self.coll.num_nodes):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0))
            me[-1].values += P.M.dot(bla[m].values)

        return me

    # OVERRIDE
    def compute_residual(self):
        """
        Computation of the residual using the collocation matrix Q
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if there are new values (e.g. from a sweep)
        # assert L.status.updated

        # compute the residual for each node

        # build QF(u)
        res_norm = []
        res = self.integrate()
        for m in range(self.coll.num_nodes):
            # add u0 and subtract u at current node
            res[m].values += P.M.dot(P.extend_sdc(L.u[0], L.time).values - P.extend_sdc(L.u[m + 1], L.time + L.dt * self.coll.nodes[m]).values)
            # add tau if associated
            if L.tau is not None:
                res[m] += L.tau[m]
            # use abs function from data type here
            res_norm.append(abs(res[m]))

        # find maximal residual over the nodes
        L.status.residual = max(res_norm)

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        # get QF(u^k)
        integral = self.myintegrate()
        for m in range(M):

            # get -QdF(u^k)_m
            for j in range(M + 1):
                integral[m] -= L.dt * self.QI[m + 1, j] * L.f[j]

            # add initial value
            integral[m] += P.extend_sdc(L.u[0], L.time)
            # add tau if associated
            if L.tau is not None:
                integral[m] += L.tau[m]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(m + 1):
                rhs += L.dt * self.QI[m + 1, j] * L.f[j]

            # implicit solve with prefactor stemming from the diagonal of Qd
            L.u[m + 1] = P.solve_system(rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1],
                                        L.time + L.dt * self.coll.nodes[m])
            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if (self.coll.right_is_node and not self.params.do_coll_update):
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt * self.coll.weights[m] * L.f[m + 1]
            # add up tau correction of the full interval (last entry)
            if L.tau is not None:
                L.uend += L.tau[-1]

        return None
