import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
from matplotlib import rc
import matplotlib as mpl
mpl.use("pgf")
pgf_with_custom_preamble = {
    "font.family": "serif", # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    "pgf.preamble": [
         "\\usepackage{units}",         # load additional packages
         "\\usepackage{metalogo}",
         "\\usepackage{unicode-math}",  # unicode math setup
         #r"\setmathfont{xits-math.otf}",
         #r"\setmainfont{DejaVu Serif}", # serif font via preamble
         ]
}
mpl.rcParams.update(pgf_with_custom_preamble)
# rc('font',size=11)
# rc('font',family='serif')
# rc('axes',labelsize=14)

import matplotlib.pyplot as plt


def iterate(iter_mat, v):
    return iter_mat.dot(v)


def iter_mat_jac(mat):
    precond = sp.dia_matrix((mat.diagonal(), 0), shape=mat.shape).tocsc()
    omega = 2./3
    return sp.eye(mat.shape[0]) - omega*sla.inv(precond).dot(A)


def iter_mat_gs(mat):
    precond = sp.tril(mat).tocsc()
    return sp.eye(mat.shape[0]) - sla.inv(precond).dot(A)


def sysmat(nx):
    stencil = [1, -2, 1]
    diags = [-1, 0, 1]
    a1d = sp.diags(stencil, diags, shape=(nx, nx), format='lil')

    a1d = sp.csr_matrix(a1d)
    return (nx+1)**2 * sp.csr_matrix(sp.kronsum(a1d, a1d))


def get_mode(nx, k, l):
    x = [(np.pi*k*i)/(nx+1) for i in range(1, nx+1)]
    y = [(np.pi*l*i)/(nx + 1) for i in range(1, nx+1)]
    x, y = np.meshgrid(x, y)

    return np.sin(x)*np.sin(y)

nx = 15
dx = 1./(nx+1)
grid_axis = 0 + np.arange(1, nx+1) * dx
x, y = np.meshgrid(grid_axis, grid_axis)

A = sysmat(nx)
# R = iter_mat_jac(A)
R = iter_mat_gs(A)
print(type(R))

# eigvals, eigvecs = sla.eigs(A, k=4)
# print(eigvecs)
# for elem in eigvecs.T:
#     plt.plot(elem)
#     plt.show()

cmap = cm.get_cmap('viridis')

v = np.random.uniform(0, 1, nx**2)
err_vec = [la.norm(v, np.inf)]
mode = get_mode(nx, 14, 14)
for i in range(50):
    v = iterate(R, v)
    err_vec.append(la.norm(v, np.inf))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.set_zlim(0, 0.08)
    #ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    #ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=35, azim=153)
    #ax.plot_surface(x, y, v.reshape((nx, nx)), cmap=cmap, rstride=1, cstride=1, alpha=0.1)
    ax.plot_surface(x, y, mode.reshape((nx, nx)), cmap=cmap, rstride=1, cstride=1, alpha=0.15)
    #ax.scatter(x[::2, ::2], y[::2, ::2], mode.reshape((nx, nx))[::2, ::2], s=150, c='w')
    ax.scatter(x[1::2, 1::2], y[1::2, 1::2], mode.reshape((nx, nx))[1::2, 1::2], s=90, c='w')
    ax.scatter(x[1::2, 1::2], y[1::2, 1::2], mode.reshape((nx, nx))[1::2, 1::2], s=30, c=np.array([[0, 0, 0]]))
    #ax.plot_surface(x[::2], y[::2], mode.reshape((nx, nx))[::2], rstride=1, cstride=1, alpha=0.1)
    cset = ax.contour(x, y, mode.reshape((nx, nx)), 20, zdir='z', offset=10, cmap=cmap)
    ax.set_xlabel('$x$', fontsize=35)
    ax.set_ylabel('$y$', fontsize=35)
    # ax.xaxis._axinfo['label']['space_factor'] = 2.8
    ax.dist = 10.5
    # ax.set_zlabel('Error')
    # [t.set_va('center') for t in ax.get_yticklabels()]
    # [t.set_ha('left') for t in ax.get_yticklabels()]
    # [t.set_va('center') for t in ax.get_xticklabels()]
    # [t.set_ha('right') for t in ax.get_xticklabels()]
    # [t.set_va('center') for t in ax.get_zticklabels()]
    # [t.set_ha('left') for t in ax.get_zticklabels()]
    # ax.xaxis.set_rotate_label(True)
    ax.tick_params(axis='both', which='major', pad=-1)
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.xaxis._axinfo['label']['space_factor'] = 10.0
    ax.yaxis._axinfo['label']['space_factor'] = 10.0
    # ax.zaxis._axinfo['label']['space_factor'] = 3.0
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(20)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(20)
    ax.set_zticks([])
    print('hello')
    #plt.show()
    plt.savefig('/home/zam/vaassen/master_thesis/figures/coarse_grid_approx/hf_alias.pdf')
    plt.savefig('/home/zam/vaassen/master_thesis/figures/coarse_grid_approx/hf_alias.pgf')
    exit()
