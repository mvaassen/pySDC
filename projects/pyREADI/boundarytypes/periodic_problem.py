from boundarytypes.i_bvp import BoundaryValueProblemBase


class PeriodicProblem(BoundaryValueProblemBase):
    """Essentially does nothing (it's a feature, though!)
    """
    def __init__(self, rdmodel):
        super(PeriodicProblem, self).__init__(rdmodel)

    def extend(self, u_red, t):
        return u_red

    def reduce(self, u_ext):
        return u_ext
