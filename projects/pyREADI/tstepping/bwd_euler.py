

def bkwd_euler(rhs, t_interval, dt, u0):
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
        u = ifftn(u_hat, axes=self._fftnaxes)
        # DFT of discrete elliptic operator (Lu = Laplacian*u)
        lu = self.laplacian*u_hat
        d = self.rdmodel.diffs
        return np.array([d[i]*lu[i] for i in range(len(d))]) + fftn(self.rdmodel.eval_n(u), axes=self._fftnaxes)

        t = tn
        u = un
        # Test
        #u_hat[0, ind] = 0
        #u_hat[1, ind] = 0

    return u
