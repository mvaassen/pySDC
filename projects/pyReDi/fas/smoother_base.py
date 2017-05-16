# coding=utf-8
import abc
from abc import ABCMeta, abstractmethod
from fas.system_operator import SystemOperator
from tstepping.impl_euler_new import ImplEulerScheme
import scipy.sparse as sp


class SmootherBase(object):

    __metaclass__ = ABCMeta

    """Base class for smoothers

    Derive from this class to ensure consistent handling of smoothers throughout the code.

    """
    def __init__(self, sysop, *args, **kwargs):
        """Initialization routine for a smoother

        Args:
            A (scipy.sparse.csc_matrix): sparse matrix A of the system to solve
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        # we use private attributes with getters and setters here to avoid overwriting by accident
        self._sysop = None

        # now set the attributes using potential validation through the setters defined below
        self.sysop = sysop

    @abc.abstractmethod
    def smooth(self, rhs, u_old):
        """Abstract method to be overwritten by implementation
        """
        pass

    @property
    def sysop(self):
        """scipy.sparse.csc_matrix: system matrix A
        """
        return self._sysop

    @sysop.setter
    def sysop(self, sysop):
        assert isinstance(sysop, ImplEulerScheme), 'Please'
        self._sysop = sysop
