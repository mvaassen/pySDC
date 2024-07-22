import sys
#sys.path.append('/home/zam/vaassen/PycharmProjects/pySDC/')

from pySDC.core.Step import step

from pySDC.implementations.problem_classes.GeneralizedFisher_1D_FD_implicit import generalized_fisher
from pySDC.implementations.problem_classes.GeneralizedFisher_1D_FD_implicit_MG_test import generalized_fisher_MG
from pySDC.implementations.problem_classes.GrayScott_2D_FDcmpct_implicit_MG import GrayScott_MG
from pySDC.implementations.problem_classes.GeneralizedFisher_1D_FD_implicit_wtf import generalized_fisher_wtf
from pySDC.implementations.problem_classes.GrayScott_2D_FDcmpct_implicit import GrayScott
from pySDC.implementations.problem_classes.HeatEquation_2D_FD_periodic import heat2d_periodic
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.generic_implicit_compact import generic_implicit_compact
from pySDC.implementations.sweeper_classes.generic_implicit_compact_alternative import generic_implicit_compact_alt

from pySDC.helpers.stats_helper import filter_stats, sort_stats, get_list_of_types
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import collections
from operator import itemgetter

from projects.pyREADI.rdmodels.generalized_fisher_1d import GeneralizedFisher1D
from projects.pyREADI.tstepping.impl_euler_new import ImplEulerScheme
from projects.pyREADI.tstepping.implicit_timestepper import ImplicitTimeIntegrator
from projects.pyREADI.tstepping.fwd_euler import solve_t
from projects.pyREADI.fdschemes.lap_4th_cpact_1d import Lap4thCpact1D
from projects.pyREADI.boundarytypes.dirichlet_problem import DirichletProblem
from time import time
import projects.pyREADI.tests.module_diagnostics as diagn

import pickle

import warnings
warnings.filterwarnings('error')

sdc_evals = []
my_evals = []


def main():
    nxs = [2 ** (i+1) for i in range(1, 11)]
    dts = [1 ** i for i in range(1, 2)]

    print('starting SDC sim')

    start1 = time()

    sdc_err = collections.defaultdict(dict)

    for nx in nxs:
        for dt in dts:
            print('dt=%12.8e, ndofs=%d' % (dt, nx))
            #sdc_err[dt][nx] = sdc_sim(nx, dt)

    dur1 = time() - start1
    #
    # aggr = [sum(diagn.diagnostics_dict['inner_niter'][i:i+3]) for i in range(0, len(diagn.diagnostics_dict['inner_niter']), 3)]
    # print(aggr)
    # plt.loglog(nxs, aggr, marker='o')
    # plt.loglog(nxs, np.array(nxs)**1)
    # plt.show()
    #
    # aggr = [sum(diagn.diagnostics_dict['newton_wtime'][i:i + 3]) for i in
    #         range(0, len(diagn.diagnostics_dict['newton_wtime']), 3)]
    # print(aggr)
    # plt.loglog(nxs, aggr,  marker='o')
    # plt.loglog(nxs, 1e-3*np.array(nxs)**1)
    # plt.show()
    #
    # plot_err(sdc_err)
    # plt.show()
    #
    # pickle.dump([nxs, dts, diagn.diagnostics_dict['inner_niter'], diagn.diagnostics_dict['newton_wtime']], open(
    #     '/home/zam/vaassen/master_thesis/numerical_results/mg_efficiency/gray_scott/other_solvers/cg/newton_cg_alternative_problem.p',
    #     'wb'))
    #
    # exit()

    print('starting SDC sim')

    sdc_err2 = []

    start = time()

    for nx in nxs:
        sdc_dt_err2 = []
        for dt in dts:
            print('dt=%12.8e, ndofs=%d' % (dt, nx))
            #sdc_dt_err2.append(sdc_sim_2(nx, dt))
        sdc_err2.append(sdc_dt_err2)

    dur2 = time() - start

    #plot_n_err(nxs, dts, sdc_err2)
    #plt.show()

    print('starting MG SDC sim')

    mgsdc_err = []

    start = time()

    for nx in nxs:
        mgsdc_dt_err = []
        for dt in dts:
            print('dt=%12.8e, ndofs=%d' % (dt, nx))
            mgsdc_dt_err.append(sdc_sim_MG(nx, dt))
        mgsdc_err.append(mgsdc_dt_err)

    dur3 = time() - start

    aggr_mg = [sum(diagn.diagnostics_dict['fas_wtime'][i:i + 3]) for i in
               range(0, len(diagn.diagnostics_dict['fas_wtime']), 3)]
    print(aggr_mg)
    plt.loglog(np.array(nxs)**2, aggr_mg)
    plt.loglog(np.array(nxs)**2, 1e-3*np.array(nxs)**2)
    plt.show()

    aggr = [sum(diagn.diagnostics_dict['fas_niter'][i:i + 3]) for i in
            range(0, len(diagn.diagnostics_dict['fas_niter']), 3)]
    print(aggr)
    plt.loglog(nxs, aggr, marker='o')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(diagn.diagnostics_dict['res_process'])
    for i in range(0, len(diagn.diagnostics_dict['res_process']), 3):
        ax.semilogy(diagn.diagnostics_dict['res_process'][i], marker='o')
    p1 = patches.Rectangle((0, 1e-11), 12, 1e-9, fill=None, hatch='//')
    p2 = patches.Rectangle((0, 1e-11), 12, 1e-9)
    p1.set_color([0.984375, 0.7265625, 0])
    p2.set_color([1, 1, 0.9])
    ax.add_patch(p2)
    ax.add_patch(p1)
    legend = plt.legend([p1], ['target domain'])
    plt.show()

    # pickle.dump([nxs, dts, mgsdc_err, diagn.diagnostics_dict['fas_wtime'], diagn.diagnostics_dict['fas_niter']
    #                 , diagn.diagnostics_dict['res_process']],
    #             open(
    #                 '/home/zam/vaassen/master_thesis/numerical_results/mg_efficiency/gray_scott/sdc_fas_gray_scott_efficiency_alternative_problem.p',
    #                 'wb'))

    print(dur1, dur2, dur3)


    #print('starting MG impl. Euler sim')

    # mg_impl_euler_err = []
    #
    # for nx in nxs:
    #     mg_impl_euler_dt_err = []
    #     for dt in dts:
    #         print('dt=%12.8e, ndofs=%d' % (dt, nx))
    #         mg_impl_euler_dt_err.append(mystuff_sim(nx, dt))
    #     mg_impl_euler_err.append(mg_impl_euler_dt_err)
    #
    # plot_n_err(nxs, mg_impl_euler_err)
    # plt.show()

    #print(np.array(sdc_evals) - np.array(my_evals))


def sdc_sim(nx, dt, expl_boundary=False):
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-6
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    # problem_params['diff_mat'] = sp.diags([[16*2e-5, 16*1e-5]], [0]).tocsr()  # diffusion coefficient
    # problem_params['feedrate'] = 0.04
    # problem_params['killrate'] = 0.06

    problem_params['diff_mat'] = sp.diags([[1e-4, 1e-5]], [0]).tocsr()  # diffusion coefficient
    problem_params['feedrate'] = 0.0367
    problem_params['killrate'] = 0.0649

    problem_params['nvars'] = nx**2 * 2  # nvars (ndofs**ndims * nspecies)
    problem_params['nvars_axis'] = nx  # number of degrees of freedom per axis
    problem_params['newton_maxiter'] = 100
    problem_params['newton_tol'] = 1e-8
    problem_params['interval'] = (0, 2.5)
    problem_params['expl_boundary'] = expl_boundary

    # problem_params['nu'] = 1e-4
    # problem_params['freq'] = 2

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 1

    # initialize controller parameters (<-- this is new!)
    controller_params = dict()
    controller_params['logger_level'] = 20  # reduce verbosity of each run

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = GrayScott
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0
    Tend = 1

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()

    sdc_evals.append(P.eval_f(uinit, 1).values)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    dx = 1./nx
    grid_axis = 0 + np.arange(nx) * dx
    xvalues, yvalues = np.meshgrid(grid_axis, grid_axis, indexing='ij')

    # plt.contourf(xvalues, yvalues, uend.values.reshape((2, nx, nx))[0], 200)
    # plt.colorbar()
    # plt.show()
    # plt.contourf(xvalues, yvalues, uend.values.reshape((2, nx, nx))[1], 200)
    # plt.colorbar()
    # plt.show()
    # plt.contourf(xvalues, yvalues, P.u_exact(Tend).values.reshape((nx, nx)), 200)
    # plt.colorbar()
    # plt.show()

    # return (la.norm(uend.values[0:nx**2] - P.u_exact(Tend).values[0:nx**2], ord=np.inf),
    #         la.norm(uend.values[nx**2:2*nx**2] - P.u_exact(Tend).values[nx**2:2*nx**2], ord=np.inf))
    return la.norm(uend.values - P.u_exact(Tend).values, ord=np.inf)


def sdc_sim_2(nx, dt, expl_boundary=False):
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 1  # diffusion coefficient
    problem_params['lambda0'] = 0.5
    problem_params['nvars'] = nx  # number of degrees of freedom
    problem_params['newton_maxiter'] = 100
    problem_params['newton_tol'] = 1e-8
    problem_params['interval'] = (-10, 10)
    problem_params['expl_boundary'] = expl_boundary

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 100

    # initialize controller parameters (<-- this is new!)
    controller_params = dict()
    controller_params['logger_level'] = 20  # reduce verbosity of each run

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = generalized_fisher
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0
    Tend = 10

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    sdc_evals.append(P.eval_f(uinit, 1).values)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # solution from my own code
    #my_uend, = run_sim()

    #plt.plot(uend.values - P.u_exact(Tend).values)
    #plt.show()

    return la.norm(uend.values - P.u_exact(Tend).values, ord=np.inf)


def sdc_sim_MG(nx, dt):
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-6
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    # problem_params['diff_mat'] = sp.diags([[16 * 2e-5, 16 * 1e-5]], [0]).tocsr()  # diffusion coefficient
    # problem_params['feedrate'] = 0.04
    # problem_params['killrate'] = 0.06

    problem_params['diff_mat'] = sp.diags([[1e-4, 1e-5]], [0]).tocsr()  # diffusion coefficient
    problem_params['feedrate'] = 0.0367
    problem_params['killrate'] = 0.0649

    problem_params['nvars'] = nx ** 2 * 2  # nvars (ndofs**ndims * nspecies)
    problem_params['nvars_axis'] = nx  # number of degrees of freedom per axis
    problem_params['mg_maxiter'] = 100
    problem_params['mg_restol'] = 1e-8
    problem_params['interval'] = (0, 2.5)

    # problem_params['nu'] = 1e-4
    # problem_params['freq'] = 2

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 1

    # initialize controller parameters (<-- this is new!)
    controller_params = dict()
    controller_params['logger_level'] = 20  # reduce verbosity of each run

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = GrayScott_MG
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0
    Tend = 1

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()

    sdc_evals.append(P.eval_f(uinit, 1).values)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    dx = 1. / nx
    grid_axis = 0 + np.arange(nx) * dx
    xvalues, yvalues = np.meshgrid(grid_axis, grid_axis, indexing='ij')

    # plt.contourf(xvalues, yvalues, uend.values.reshape((2, nx, nx))[0], 200)
    # plt.colorbar()
    # plt.show()
    # plt.contourf(xvalues, yvalues, uend.values.reshape((2, nx, nx))[1], 200)
    # plt.colorbar()
    # plt.show()
    # plt.contourf(xvalues, yvalues, P.u_exact(Tend).values.reshape((nx, nx)), 200)
    # plt.colorbar()
    # plt.show()

    # return (la.norm(uend.values[0:nx**2] - P.u_exact(Tend).values[0:nx**2], ord=np.inf),
    #         la.norm(uend.values[nx**2:2*nx**2] - P.u_exact(Tend).values[nx**2:2*nx**2], ord=np.inf))
    return la.norm(uend.values - P.u_exact(Tend).values, ord=np.inf)


def sdc_sim_MG_w_pyReDi_problem(nx, dt):
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-8
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 1
    problem_params['lambda0'] = 0.5
    problem_params['nvars'] = nx  # number of degrees of freedom
    #problem_params['mg_maxiter'] = 100
    #problem_params['mg_restol'] = 1e-9
    problem_params['domain'] = (-40, 40)
    problem_params['fd_laplacian'] = Lap4thCpact1D
    problem_params['bc_type'] = DirichletProblem
    problem_params['is_hierarchy_master_problem'] = True

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 10

    # initialize controller parameters (<-- this is new!)
    controller_params = dict()
    controller_params['logger_level'] = 20  # reduce verbosity of each run

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = GeneralizedFisher1D
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['sweeper_class'] = generic_implicit_compact_alt
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0
    Tend = 1

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    #eigvalue_plot_list([(P.rdmodel.M_matrix[0] - P.rdmodel.L_matrix[0]).todense()])

    #sdc_evals.append(P.eval_f(uinit, 1).values)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # solution from my own code
    #my_uend, = run_sim()

    #plt.plot(uend.values - P.u_exact(Tend).values)
    #plt.show()

    return la.norm(uend.values - P.u_exact(Tend).values, ord=np.inf)


def mystuff_sim(nx, dt):
    domain = [-40, 40]
    ndofs = nx
    t_interval = [0, 1]
    diff = np.array([1])

    lambda0 = 0.5
    nu = 1

    model = GeneralizedFisher1D(ndofs, domain, DirichletProblem, None, 1, diff, lambda0, nu)

    x = model.generate_grid()
    u0 = model.bc(x, 0)
    u0inner = model.reduce(u0)

    u_ex, = model.reduce(model.bc(x, t_interval[1]))

    bla = model.extend(u0inner, 1)
    my_evals.append(model.eval_rhs(bla)[0])

    tstepper = ImplicitTimeIntegrator(ImplEulerScheme, model)

    #sol, = solve_t(model, t_interval, dt, u0inner)

    sol, = tstepper.solve(t_interval, dt, u0inner, mg_res_bound=1e-8)

    return la.norm(sol - u_ex, ord=np.inf)

def plot_err(err_dict):
    for k, v in err_dict.iteritems():

        data = [(nx, err) for nx, err in v.iteritems()]
        data = sorted(data, key=itemgetter(0))

        plt.loglog([item[0] for item in data], [item[1] for item in data], marker='o', label='dt='+str(k))
        plt.loglog([item[0] for item in data], [1./item[0] ** 2 for item in data], linestyle='--', color='gray', label='ref')

    plt.legend()
    plt.xlabel('N')
    plt.ylabel('err')
    plt.grid(True, which='both')
    plt.show()

def plot_t_err(dts, err):
    handles = []
    for elem in err:
        handle, = plt.loglog(dts, elem, marker='o')
        handles.append(handle)
    ref, = plt.loglog(dts, np.array(dts)**2, linestyle='--', color='gray')
    handles.append(ref)
    # plt.legend(handles, [r'$\Delta t = 1\mathrm{E}{-05}$', r'$\Delta t = 5\mathrm{E}{-06}$', r'$\Delta t = 2.5\mathrm{E}{-06}$', r'$\Delta t = 1.25\mathrm{E}{-06}$', r'$\mathcal{O}(N^{-4})$'])
    plt.xlabel('dt', fontsize=16)
    plt.ylabel('error', fontsize=16)
    plt.grid(True, which='both')
    plt.show()


def plot_n_err(nxs, dt_list, err):
    sdc_err = np.transpose(err)
    handles = []
    for elem in sdc_err:
        handle, = plt.loglog(nxs, elem, marker='o')
        handles.append(handle)
    ref, = plt.loglog(nxs, 1e-3*1./np.array(nxs)**4, linestyle='--', color='gray')
    #plt.loglog(nxs, 1e-3 * 1. / np.array(nxs) ** 3, linestyle='--', color='gray')
    #plt.loglog(nxs, 1e-3 * 1. / np.array(nxs) ** 2, linestyle='--', color='gray')
    handles.append(ref)
    legendstr = [r'$\Delta t = %e$' % elem for elem in dt_list]
    plt.legend(handles, legendstr)
    plt.xlabel('N (DoF)', fontsize=16)
    plt.ylabel('error', fontsize=16)
    plt.grid(True, which='both')
    plt.show()

if __name__ == "__main__":
    main()
