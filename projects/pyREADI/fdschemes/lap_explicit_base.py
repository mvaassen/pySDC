from abc import ABCMeta, abstractmethod
from projects.pyREADI.fdschemes.laplacian_base import LaplacianBase
import scipy.sparse as sp

__author__ = 'vaassen'


# TODO: complete doc
class LapExplicitBase(LaplacianBase):

    __metaclass__ = ABCMeta
    """Abstract base class for FD approximations of the Laplace operator

    """

    def __init__(self, ndims, periodic=True):
        super(LapExplicitBase, self).__init__(ndims, periodic)

    def get_lhs_matrix(self, nx):
        return sp.csr_matrix(sp.eye(nx**self.ndims, nx**self.ndims))
