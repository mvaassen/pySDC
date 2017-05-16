# coding=utf-8
import abc
from abc import ABCMeta, abstractmethod


class TransferBase(object):
    __metaclass__ = ABCMeta
    """Base class for restriction and prolongation operators

    Derive from this class to ensure consistent handling of transfers throughout the code.

    """
    def __init__(self, nx_fine, nx_coarse, ndims, bc_type, *args, **kwargs):
        """Initialization routine for transfer operators

        Args:
            ndofs_fine (int): number of DoFs per spatial axis on the fine grid
            ndofs_coarse (int): number of DoFs per spatial axis on the coarse grid
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """

        self.nx_fine = nx_fine
        self.nx_coarse = nx_coarse
        self._ndims = ndims
        self.bc_type = bc_type

        self.restriction_matrix = None
        self.interpolation_matrix = None

    @abstractmethod
    def restrict(self, u_fine):
        """Abstract restriction method to be overwritten by implementation
        """
        pass

    @abstractmethod
    def interpolate(self, u_coarse):
        """Abstract prolongation method to be overwritten by implementation
        """
        pass
