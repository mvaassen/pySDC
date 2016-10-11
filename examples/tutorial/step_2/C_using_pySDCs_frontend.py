from implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

from implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from implementations.sweeper_classes.imex_1st_order import imex_1st_order


def main():
    """
    A simple test program to run IMEX SDC for a single time step
    """
    # initialize level parameters
    level_params = {}
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.1

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    # sweeper_params['do_LU'] = True      # for this sweeper we can use the LU trick for the implicit part!

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = 1023  # number of degrees of freedom

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 20

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = heat1d_forced
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = allinclusive_classic_nonMPI(num_procs=1, controller_params={}, description=description)

    # set time parameters
    t0 = 0.1
    Tend = 0.3   # note that we are requesting 2 time steps here (dt is 0.1)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)
    err = abs(uex - uend)
    print('Error: %12.8e' %(err))

    assert err <= 2E-5,"ERROR: controller doing IMEX SDC iteration did not reduce the error enough, got %s" %err


if __name__ == "__main__":
    main()