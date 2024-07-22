import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la

import pickle

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
from matplotlib import rc
import matplotlib as mpl
#mpl.use("pgf")
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
    omega = 4./5
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

nx = 63
dx = 1./(nx+1)
grid_axis = 0 + np.arange(1, nx+1) * dx
x, y = np.meshgrid(grid_axis, grid_axis)

A = sysmat(nx)
R = iter_mat_jac(A)
# R = iter_mat_gs(A)
print(type(R))

# eigvals, eigvecs = sla.eigs(A, k=4)
# print(eigvecs)
# for elem in eigvecs.T:
#     plt.plot(elem)
#     plt.show()

cmap = cm.get_cmap('viridis')

# v = np.random.uniform(0, 1, nx**2)
# v = iterate(R, v)
# pickle.dump(v, open("/home/zam/vaassen/master_thesis/figures/smoothing/init_noise.p", "wb"))
# exit()
v = pickle.load(open("/home/zam/vaassen/master_thesis/figures/smoothing/init_noise.p", "rb"))
err_vec = [la.norm(v, np.inf)]
for i in range(50):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlim(0, 0.8)
    #ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    #ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=35, azim=153)
    ax.plot_surface(x, y, v.reshape((nx, nx)), cmap=cmap, rstride=1, cstride=1, alpha=0.15)
    cset = ax.contour(x, y, v.reshape((nx, nx)), 20, zdir='z', offset=0, cmap=cmap)
    ax.set_xlabel('$x$', fontsize=35)
    ax.set_ylabel('$y$', fontsize=35)
    #ax.xaxis._axinfo['label']['space_factor'] = 2.8
    ax.dist = 10.5
    # ax.set_zlabel('Error')
    # [t.set_va('center') for t in ax.get_yticklabels()]
    # [t.set_ha('left') for t in ax.get_yticklabels()]
    # [t.set_va('center') for t in ax.get_xticklabels()]
    # [t.set_ha('right') for t in ax.get_xticklabels()]
    # [t.set_va('center') for t in ax.get_zticklabels()]
    # [t.set_ha('left') for t in ax.get_zticklabels()]
    #ax.xaxis.set_rotate_label(True)
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
    plt.show()
    # plt.savefig('/home/zam/vaassen/master_thesis/figures/smoothing/gauss_seidel/iter%d.pdf' % i)
    # plt.savefig('/home/zam/vaassen/master_thesis/figures/smoothing/gauss_seidel/iter%d.pgf' % i)

    v = iterate(R, v)
    err_vec.append(la.norm(v, np.inf))

# plt.ylim(0)
#
# plt.plot([i for i in range(51)], err_vec, '-ro', markevery=[0], markersize=15, color='#ca0020')
# # plt.axvline(x=5, linestyle='dotted', color='r')
# #plt.vlines([0], [0], err_vec[0], linestyles='dashed', lw=2, color='gray')
# plt.xlabel('iteration cycle $(k)$', fontsize=40)
# plt.ylabel('$L^\infty$-error $\|e^{(k)}\|_\infty$', fontsize=40)
# plt.tick_params(axis='both', which='major', labelsize=30)
# plt.tick_params(axis='both', which='minor', labelsize=8)
# plt.tight_layout()
# #plt.show()
# plt.savefig('/home/zam/vaassen/master_thesis/figures/smoothing/w_jacobi/2/error0.pdf')
# plt.savefig('/home/zam/vaassen/master_thesis/figures/smoothing/w_jacobi/2/error0.pgf')
# plt.close()
#
# plt.ylim(0)
#
# plt.plot([i for i in range(51)], err_vec, '-ro', markevery=[5], markersize=15, color='#ca0020')
# # plt.axvline(x=5, linestyle='dotted', color='r')
# plt.vlines([5], [0], err_vec[5], linestyles='dashed', lw=2, color='gray')
# plt.xlabel('iteration cycle $(k)$', fontsize=40)
# plt.ylabel('$L^\infty$-error $\|e^{(k)}\|_\infty$', fontsize=40)
# plt.tick_params(axis='both', which='major', labelsize=30)
# plt.tick_params(axis='both', which='minor', labelsize=8)
# plt.tight_layout()
# plt.savefig('/home/zam/vaassen/master_thesis/figures/smoothing/w_jacobi/2/error5.pdf')
# plt.savefig('/home/zam/vaassen/master_thesis/figures/smoothing/w_jacobi/2/error5.pgf')
# plt.close()
#
# plt.ylim(0)
#
# plt.plot([i for i in range(51)], err_vec, '-ro', markevery=[10], markersize=15, color='#ca0020')
# # plt.axvline(x=10, linestyle='dotted', color='r')
# plt.vlines([10], [0], err_vec[10], linestyles='dashed', lw=2, color='gray')
# plt.xlabel('iteration cycle $(k)$', fontsize=40)
# plt.ylabel('$L^\infty$-error $\|e^{(k)}\|_\infty$', fontsize=40)
# plt.tick_params(axis='both', which='major', labelsize=30)
# plt.tick_params(axis='both', which='minor', labelsize=8)
# plt.tight_layout()
# plt.savefig('/home/zam/vaassen/master_thesis/figures/smoothing/w_jacobi/2/error10.pdf')
# plt.savefig('/home/zam/vaassen/master_thesis/figures/smoothing/w_jacobi/2/error10.pgf')
# plt.close()
