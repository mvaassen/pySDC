import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import matplotlib.pyplot as plt

def get_A_cmpct(N, dx):
    """
    Helper function to assemble FD matrix A in sparse format

    Args:
        N (int): number of dofs
        dx (float): distance between two spatial nodes

    Returns:
        scipy.sparse.csc_matrix: matrix A in CSC format
    """

    a = 6. / 5
    stencil = [a, -2 * a, a]
    diags = [-1, 0, 1]
    a1d = sp.diags(stencil, diags, shape=(N + 2, N + 2), format='lil')[1:-1, :]
    Ma1d = sp.csr_matrix(1.0 / (dx ** 2) * a1d)

    bla = np.zeros(N + 2)
    bla[0:4] = np.array([13., -27., 15., -1.])
    a1d = sp.vstack([bla, a1d, bla[::-1]])
    # print(a1d.todense())
    a1d *= 1.0 / (dx ** 2)

    a1d = sp.csr_matrix(a1d)

    stencil = [1. / 10, 1, 1. / 10]
    diags = [-1, 0, 1]
    M = sp.diags(stencil, diags, shape=(N + 2, N + 2), format='lil')[1:-1, :]
    MM = sp.csr_matrix(M)

    bla = np.zeros(N + 2)
    bla[0:2] = np.array([1., 11.])
    M = sp.vstack([bla, M, bla[::-1]])
    # print(M.todense())

    M = sp.csc_matrix(M)

    return M, a1d, sla.inv(M).dot(a1d)


def get_A_cmpct_periodic(N, dx_periodic):
    """
    Helper function to assemble FD matrix A in sparse format

    Args:
        N (int): number of dofs
        dx (float): distance between two spatial nodes

    Returns:
        scipy.sparse.csc_matrix: matrix A in CSC format
    """

    a = 6. / 5
    stencil = [a, -2 * a, a]
    diags = [-1, 0, 1]
    a1d = sp.diags(stencil, diags, shape=(N, N), format='lil')
    a1d[0, -1] = a
    a1d[-1, 0] = a
    a1d *= 1.0 / (dx_periodic ** 2)
    a1d = sp.csc_matrix(a1d)

    stencil = [1. / 10, 1, 1. / 10]
    diags = [-1, 0, 1]
    M = sp.diags(stencil, diags, shape=(N, N), format='lil')
    M[0, -1] = 1./10
    M[-1, 0] = 1./10
    M = sp.csc_matrix(M)

    return M, a1d, sla.inv(M).dot(a1d)


def get_A_cmpct_expl_boundary(N, dx):
    """
    Helper function to assemble FD matrix A in sparse format

    Args:
        N (int): number of dofs
        dx (float): distance between two spatial nodes

    Returns:
        scipy.sparse.csc_matrix: matrix A in CSC format
    """

    a = 6. / 5
    stencil = [a, -2 * a, a]
    diags = [-1, 0, 1]
    a1d = sp.diags(stencil, diags, shape=(N + 2, N + 2), format='lil')[1:-1, :]

    bla = np.zeros(N + 2)
    bla[0:5] = np.array([35./12, -26./3, 19./2, -14./3, 11./12])
    #bla[0:4] = np.array([2, -5, 4, -1])
    a1d = sp.vstack([bla, a1d, bla[::-1]])
    # print(a1d.todense())
    a1d *= 1.0 / (dx ** 2)
    a1d = sp.csc_matrix(a1d)

    stencil = [1. / 10, 1, 1. / 10]
    diags = [-1, 0, 1]
    M = sp.diags(stencil, diags, shape=(N + 2, N + 2), format='lil')[1:-1, :]
    bla = np.zeros(N + 2)
    bla[0] = 1.
    M = sp.vstack([bla, M, bla[::-1]])
    # print(M.todense())
    M = sp.csc_matrix(M)

    bla = sla.inv(M)[1:-1, 1:-1].dot(a1d[1:-1, 1:-1])

    return M, a1d

def get_A_expl(N, dx):
    """
    Helper function to assemble FD matrix A in sparse format

    Args:
        N (int): number of dofs
        dx (float): distance between two spatial nodes

    Returns:
        scipy.sparse.csc_matrix: matrix A in CSC format
    """

    stencil = [-1, 16, -30, 16, -1]
    A = sp.diags(stencil, [-2, -1, 0, 1, 2], shape=(N + 2, N + 2), format='lil')
    A *= 1. / (12*(dx ** 2))

    return A

nxs = [2**i for i in range(2, 10)]
err_compact = []
err_expl = []
err_noncompact = []

err_periodic = []

for nx in nxs:
    N = nx
    dx = 1./(N + 1)

    # periodic stuff
    dx_periodic = 1./N
    x_periodic = dx_periodic*np.arange(N)
    u_periodic = np.sin(2*np.pi*x_periodic)
    lap_u_exact_periodic = -4 * (np.pi) ** 2 * np.sin(2*np.pi * x_periodic)
    # end periodic stuff

    x = np.linspace(0, 1, N + 2)
    u = np.sin(1*np.pi*x)

    S_lhs, S_rhs, L = get_A_cmpct(N, dx)
    S_lhs_expl, S_rhs_expl = get_A_cmpct_expl_boundary(N, dx)
    A_expl = get_A_expl(N, dx)

    lap_u_exact = -1*(np.pi)**2*np.sin(np.pi*x)
    lap_u_compact = sla.spsolve(S_lhs, S_rhs.dot(u))
    lap_u_expl = sla.spsolve(S_lhs_expl, S_rhs_expl.dot(u))
    lap_u_noncompact = A_expl.dot(u)

    # periodic stuff
    S_lhs_p, S_rhs_p, L_p = get_A_cmpct_periodic(N, dx_periodic)
    lap_u_compact_p = sla.spsolve(S_lhs_p, S_rhs_p.dot(u_periodic))

    err_periodic.append(la.norm((lap_u_exact_periodic - lap_u_compact_p), ord=np.inf))
    # plt.plot(x_periodic, (lap_u_exact_periodic - lap_u_compact_p))
    # plt.show()
    # end periodic stuff

    err_compact.append(la.norm((lap_u_exact - lap_u_compact)[2:-2], ord=np.inf))
    err_expl.append(la.norm((lap_u_exact - lap_u_expl)[2:-2], ord=np.inf))
    err_noncompact.append(la.norm((lap_u_exact - lap_u_noncompact)[2:-2], ord=np.inf))

    # plt.plot(x, (lap_u_exact - lap_u_compact))
    # plt.plot(x, (lap_u_exact - lap_u_expl))
    # plt.plot(x, (lap_u_exact - lap_u_noncompact))
    # plt.show()


ref, = plt.loglog(nxs, 1./np.array(nxs)**4, linestyle='--', color='gray')
plt.xlabel('nx', fontsize=16)
plt.ylabel('error norm', fontsize=16)
plt.grid(True, which='both')
plt.loglog(nxs, err_compact)
plt.loglog(nxs, err_expl)
plt.loglog(nxs, err_noncompact)
plt.loglog(nxs, err_periodic)
plt.show()

plt.plot(nxs, np.array(err_expl)/np.array(err_compact))
plt.show()
