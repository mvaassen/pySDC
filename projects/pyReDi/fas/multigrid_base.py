import abc
from abc import ABCMeta, abstractmethod
from rdmodels.rdmodel_base import RDModelBase
from fas.system_operator import SystemOperator
from tstepping.impl_euler_new import ImplEulerScheme
import numpy as np
import scipy.sparse as sp
import time


class MultigridBase(object):
    """Base class for multigrid cycles

    This mainly includes the data structure required to cycle through the levels:
    the vectors vh and fh, a list of smoothers (i.e. the correct matrices) and
    a list of transfer operators

    Attributes:
        nlevels (int): number of levels in the MG hierarchy
        vh (list of numpy.ndarrays): data structure for the solution vectors
        fh (list of numpy.ndarrays): data structure for the rhs vectors
        trans (list of :class:`pymg.transfer_base.TransferBase`): list of transfer operators
        smoo (list of :class:`pymg.smoother_base.SmootherBase`): list of smoothers
        Acoarse (scipy.sparse.csc_matrix): system matrix on the coarsest level
    """

    def __init__(self, rdmodel, galerkin, *args, **kwargs):
        """Initialization routine for a multigrid solver

        Note:
            instantiation of smoothers and transfer operators is separated to allow
            passing parameters more easily

        Args:
            ndofs (int): number of degrees of freedom (see
                :attr:`pymg.problem_base.ProblemBase.ndofs`)
            nlevels (int): number of levels in the hierarchy
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        assert isinstance(rdmodel, RDModelBase)
        #assert 0 <= nlevels <= np.log2(nx+1)

        self._rdmodel = rdmodel
        self.galerkin = galerkin

        self._nx_list = self._rdmodel.generate_ndofs_list()
        self.nlevels = len(self._nx_list)

        print(self._nx_list)

        self.vh = [np.array([np.zeros(i**self._rdmodel.ndims)] * self._rdmodel.nspecies) for i in self._nx_list]
        self.fh = [np.array([np.zeros(i**self._rdmodel.ndims)] * self._rdmodel.nspecies) for i in self._nx_list]

        self.trans = []
        self.smoo = []
        self.Acoarse = None

    def reset_vectors(self, lstart):
        """Routine to (re)set the solution and rhs vectors to zero

        Args:
            lstart (int): level to start from (all below will be set to zero)
        """
        self.vh[lstart:] = [np.zeros(i*i) for i in self._nx_list[lstart:]]
        self.fh[lstart:] = [np.zeros(i*i) for i in self._nx_list[lstart:]]

    def attach_smoother(self, smoother_class, sysop, *args, **kwargs):
        """Routine to attach a smoother to each level (except for the coarsest)

        Args:
            smoother_class (see :class:`pymg.smoother_base.SmootherBase`): the class of smoothers
            A (scipy.sparse.csc_matrix): system matrix of the problem
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        # for the Galerkin approach: check if transfer operators are attached
        assert len(self.trans) == self.nlevels - 1

        assert isinstance(sysop, ImplEulerScheme)

        self.smoo = []

        # work your way through the hierarchy
        sysop_curr = sysop
        self.smoo.append(smoother_class(sysop_curr, *args, **kwargs))
        for l in range(0, self.nlevels-1):
            if self.galerkin:
                sysop_curr = sysop_curr.get_coarse_sysop(self.trans[l])
                self.smoo.append(smoother_class(sysop_curr, *args, **kwargs))
            else:
                print('No Galerkin, level %d' % l)
                sysop_curr = sysop_curr.coarsen(self._nx_list[l + 1])
                self.smoo.append(smoother_class(sysop_curr, *args, **kwargs))

        # in case we want to do smoothing instead of solving on the coarsest level:
        #self.smoo.append(smoother_class(sysop_curr.get_coarse_sysop(self.trans[-1]), *args, **kwargs))

    def attach_transfer(self, transfer_class, *args, **kwargs):
        """Routine to attach transfer operators to each level (except for the coarsest)

        Args:
            transfer_class (see :class:`pymg.transfer_base.TransferBase`): the class of transfer ops
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        for l in range(self.nlevels-1):
            self.trans.append(transfer_class(nx_fine=self._nx_list[l],
                                             nx_coarse=self._nx_list[l + 1], ndims=self._rdmodel.ndims,
                                             bc_type=type(self._rdmodel.bc_handler).__name__, *args, **kwargs))
