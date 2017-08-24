from abc import ABCMeta, abstractmethod

__author__ = 'vaassen'


# TODO: complete doc
class LaplacianBase(object):

    __metaclass__ = ABCMeta
    """Abstract base class for FD approximations of the Laplace operator

    """

    def __init__(self, ndims, periodic=True):
        self.ndims = ndims
        self.periodic = periodic

    @abstractmethod
    def get_lhs_matrix(self, nx):
        pass

    @abstractmethod
    def get_rhs_matrix(self, nx):
        pass
