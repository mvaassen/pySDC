from abc import ABCMeta, abstractmethod
import numpy as np


class BoundaryValueProblemBase(object):

    __metaclass__ = ABCMeta

    def __init__(self, rdmodel):
        self.rdmodel = rdmodel

        assert self.rdmodel.nnodes is not None
        assert self.rdmodel.dx is not None

    @abstractmethod
    def compute_nnodes_dx(nvars, domlen):
        # leaving out 'self' forces this to be implemented as a staticmethod in subclasses
        pass

    @abstractmethod
    def extend(self, u_red, t):
        pass

    @abstractmethod
    def reduce(self, u_ext):
        pass
