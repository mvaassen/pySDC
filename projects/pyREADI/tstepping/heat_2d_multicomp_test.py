from rdmodels.gray_scott import GrayScottEq
from rdmodels.heat import HeatEq
from spectral.fourier_solver import FourierSpectral
from tstepping.fwd_euler import *
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.linalg as la
import matplotlib.pyplot as plt
import time

plt.set_cmap('RdBu')


def u_exact(u0, t, diff_coeff, k, l):
    return u0*np.exp(-diff_coeff*((2*np.pi*k)**2 + (2*np.pi*l)**2)*t)

domain = [0, 1]
ndofs = 256
t_interval = [0, 10]
dt = 10

diff = np.array([0.01, 0.02, 0.04])

model = HeatEq(, ndofs, 2, domain, diff
x, y = model.generate_grid()

u0 = np.array([np.sin(2*np.pi*x) * np.sin(2*np.pi*y), np.sin(2*np.pi*x) * np.sin(2*np.pi*y), np.sin(2*np.pi*x) * np.sin(2*np.pi*y)])
u0_ex = np.array([u_exact(u0[0], 0, diff[0], 1, 1), u_exact(u0[1], 0, diff[1], 1, 1), u_exact(u0[2], 0, diff[2], 1, 1)])

plt.contourf(x, y, u0[0], 200)
plt.colorbar()
plt.show()

spectral = FourierSpectral(model)

err1 = []
err2 = []
err3 = []
dts = [dt/(2**i) for i in range(12)]
for delta in dts:
    print(delta)
    sol = spectral.fssolve(solve_t, t_interval, delta, u0, tspace='fourier')
    sol_ex = np.array([u_exact(u0[0], t_interval[1], diff[0], 1, 1), u_exact(u0[1], t_interval[1], diff[1], 1, 1), u_exact(u0[2], t_interval[1], diff[2], 1, 1)])
    try:
        err1.append(la.norm((sol_ex[0]-sol[0]).flatten(), ord=np.inf))
    except ValueError:
        print('triggered')
        err1.append(0)
    try:
        err2.append(la.norm((sol_ex[1]-sol[1]).flatten(), ord=np.inf))
    except ValueError:
        print('triggered')
        err2.append(0)
    try:
        err3.append(la.norm((sol_ex[2]-sol[2]).flatten(), ord=np.inf))
    except ValueError:
        print('triggered')
        err3.append(0)

diff1, = plt.loglog(dts, err1, marker='o', color='red', label='diff=0.01')
diff2, = plt.loglog(dts, err2, marker='o', color='green', label='diff=0.02')
diff3, = plt.loglog(dts, err3, marker='o', color='blue', label='diff=0.04')
ref, = plt.loglog(dts, dts, linestyle='--', color='orange', label='h (1st order ref.)')
plt.grid(True, which='both')
plt.ylim(0, 10**10)
plt.legend(handles=[diff1, diff2, diff3, ref])
plt.suptitle('2D heat eq., ndofs=256, domain=[0, 2pi],\n diff_coeffs: 0.01, 0.02, 0.04, te=10, init=sin(x)*sin(y), init_dt=10')
plt.show()
#plt.savefig('heat_multicomp_bwd_euler.png')


