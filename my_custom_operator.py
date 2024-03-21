import numpy as np
from cil.optimisation.operators import LinearOperator

class MyCustomOperator(LinearOperator):
    def __init__(self, A, domain_geometry, range_geometry, order):
        """
        Custom linear operator that applies a matrix A to an input array.

        Parameters:
        A (ndarray): The matrix to apply in the direct and adjoint operations.
        domain_geometry (ImageGeometry): The geometry of the input space.
        range_geometry (ImageGeometry): The geometry of the output space.
        order (str): The order of flattening and reshaping operations.
                     'C' for row-major (C-style),
                     'F' for column-major (Fortran-style).
        """
        super(MyCustomOperator, self).__init__(domain_geometry=domain_geometry, 
                                               range_geometry=range_geometry)
        self.A = A
        self.order = order

    def direct(self, x, out=None):
        flattened_x = x.as_array().flatten(order=self.order)
        result_1d = np.dot(self.A, flattened_x)
        result_2d = result_1d.reshape((self.range_geometry().voxel_num_y, 
                                        self.range_geometry().voxel_num_x), 
                                       order=self.order)

        if out is None:
            result = self.range_geometry().allocate()
            result.fill(result_2d)
            return result
        else:
            out.fill(result_2d)

    def adjoint(self, y, out=None):
        flattened_y = y.as_array().flatten(order=self.order)
        result_1d = np.dot(self.A.T, flattened_y)
        result_2d = result_1d.reshape((self.domain_geometry().voxel_num_y, 
                                        self.domain_geometry().voxel_num_x), 
                                       order=self.order)

        if out is None:
            result = self.domain_geometry().allocate()
            result.fill(result_2d)
            return result
        else:
            out.fill(result_2d)