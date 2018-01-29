from pyspark.tests import ReusedPySparkTestCase
from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import MatrixEntry, CoordinateMatrix
import numpy as np
import labelpropagation
from labelpropagation import lp_matrix_multiply


class TestNaive_multiplication_rdd(ReusedPySparkTestCase):
    def setUp(self):
        spark = SparkSession(sparkContext=self.sc)
        y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        X_triangle = np.array([[1, 0, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0],
                               [3, 4, 1, 0, 0, 0], [5, 6, 7, 1, 0, 0],
                               [1, 4, 2, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
        self.y_shape = y.shape
        self.longMessage = True
        self.X_shape = X_triangle.shape
        self.X_real = X_triangle+X_triangle.T-np.eye(6)
        self.product = self.X_real.dot(y)
        self.rdd_y = (self.sc.parallelize(y)
            .map(lambda x: x.tolist()).map(lambda x: list(enumerate(x)))
            .zipWithIndex()
            .flatMap(lambda x: [MatrixEntry(i=x[1], j=jdx, value=val) for jdx, val in x[0]])
            .filter(lambda x: x.value != 0.))

        self.rdd_X = (self.sc.parallelize(X_triangle)
            .map(lambda x: x.tolist()).map(lambda x: list(enumerate(x)))
            .zipWithIndex()
            .flatMap(lambda x: [MatrixEntry(i=x[1], j=jdx, value=val) for jdx, val in x[0]])
            .filter(lambda x: x.value != 0.))

    def test_naive_multiplication_rdd(self):
        computed_result = lp_matrix_multiply.naive_multiplication_rdd(
            mat_a=self.rdd_X, mat_b=self.rdd_y, is_triangle=True).collect()
        actual_result = self.product
        for element in computed_result:
            self.assertEqual(actual_result[element.i, element.j], element.value)

    def test_naive_multiplication_coord_matrix(self):
        mat_a = CoordinateMatrix(self.rdd_X, *self.X_shape)
        mat_b = CoordinateMatrix(self.rdd_y, *self.y_shape)
        computed_result = lp_matrix_multiply.naive_multiplication_rdd(
            mat_a=mat_a, mat_b=mat_b, is_triangle=True).collect()
        actual_result = self.product
        for element in computed_result:
            self.assertEqual(actual_result[element.i, element.j], element.value,
                             msg='i {}, j {}, computed {} - actual_value: {}'.format(
                                 element.i, element.j, element.value, actual_result[element.i, element.j]))



