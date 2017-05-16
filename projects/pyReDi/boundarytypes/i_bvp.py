from abc import ABCMeta, abstractmethod
import numpy as np


class BoundaryValueProblemBase(object):

    __metaclass__ = ABCMeta

    def __init__(self, rdmodel):
        self.rdmodel = rdmodel

    @abstractmethod
    def extend(self, u_red, t):
        pass

    @abstractmethod
    def reduce(self, u_ext):
        pass
