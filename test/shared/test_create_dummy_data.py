from unittest import TestCase
from pyspark.tests import ReusedPySparkTestCase, PySparkTestCase
from shared import create_dummy_data
import functools
import numpy as np
import pandas as pd

class TestCreateOutliers(ReusedPySparkTestCase):

    def setUp(self):
        super().setUp()
        self.n_samples = [10, 3, 27]
        self.means = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [5.0, 5.0, 5.0, 5.0, 5.0, -5, -5, -5, -5, -5],
                      [ 1,  5, -4, -9, -9, -5,  8,  0,  6, -2]]
        self.stds = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                     [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                     [ 8.16, 9.42, 4.27, 0.34 , 8.65, 7.90, 6.14, 5.31, 1.37, 9.96]]
        self.m = 10
        self.n = functools.reduce(lambda a, b: a+b, self.n_samples)
        self.features = [chr(c) for c in range(ord('a'), ord('a')+self.m, 1)]

    def test_create_norm_cluster_data_pandas(self):
        partial_create_dummy = functools.partial(
            create_dummy_data.create_norm_cluster_data_pandas,
            n_amounts=self.n_samples, means=self.means, std=None)

        test_instances = [self.m, ['a','b','d','e','c','f','h','u','t','e'], None]
        result_instances_columns = [self.m, 10, 10]
        for test_i, result_i in zip(test_instances, result_instances_columns):
            pdf = partial_create_dummy(features=test_i)
            self.assertEqual(len(pdf.columns), result_i+2) # check columns

        self.assertEqual(len(pdf), self.n) # check data point

    def test_create_norm_cluster_data_spark(self):
        data_frame = create_dummy_data.create_norm_cluster_data_spark(
            sc=self.sc, n_amounts = self.n_samples, means = self.means, std = None)

        self.assertEqual(len(data_frame.columns), self.m+2)
        self.assertEqual(data_frame.count(), self.n)

