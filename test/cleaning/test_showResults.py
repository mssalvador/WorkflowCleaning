from unittest import TestCase

from pyspark.tests import ReusedPySparkTestCase, PySparkTestCase
from pyspark.sql import Window
from pyspark.sql import SparkSession
from pyspark.ml.linalg import DenseVector
import pandas as pd
import numpy as np
from pyspark.sql import functions as F

from cleaning.ShowCleaning import ShowResults


class TestShowResults(PySparkTestCase):

    def setUp(self):
        super().setUp()
        self.spark = SparkSession(self.sc)

        df = pd.DataFrame(
            {'predictionCol': [0, 0, 0, 0, 0, 0, 1, 2, 1, 1],
             'distance': [0.5, 1.5, 0.5, 0.1, 0.01, 6.0, 20.0, 13, 2, 1],
             'point_col': [DenseVector(np.array([1.0, 1.0])),
                           DenseVector(np.array([0.0, 0.0])),
                           DenseVector(np.array([3.0, 3.0])),
                           DenseVector(np.array([1.0, 1.0])),
                           DenseVector(np.array([0.0, 0.0])),
                           DenseVector(np.array([0.0, 0.0])),
                           DenseVector(np.array([3.0, 3.0])),
                           DenseVector(np.array([1.0, 1.0])),
                           DenseVector(np.array([0.0, 0.0])),
                           DenseVector(np.array([3.0, 3.0]))],
             'centers': [DenseVector(np.array([2.0, 2.0])),
                         DenseVector(np.array([2.0, 2.0])),
                         DenseVector(np.array([6.0, 6.0])),
                         DenseVector(np.array([2.0, 2.0])),
                         DenseVector(np.array([2.0, 2.0])),
                         DenseVector(np.array([6.0, 6.0])),
                         DenseVector(np.array([2.0, 2.0])),
                         DenseVector(np.array([2.0, 2.0])),
                         DenseVector(np.array([6.0, 6.0])),
                         DenseVector(np.array([6.0, 6.0]))]},
            columns=['predictionCol', 'distance', 'point_col', 'centers'])

        self.dataframe = self.spark.createDataFrame(df)

    def test_add_row_index(self):

        computed_dataframe = ShowResults._add_row_index(self.dataframe)
        self.assertIn(('rowId', 'bigint'), computed_dataframe.dtypes)

        computed_dataframe = ShowResults._add_row_index(self.dataframe, rowId='roow')
        self.assertIn(('roow', 'bigint'), computed_dataframe.dtypes)

    def test_add_distances(self):
        from math import sqrt
        computed_dataframe = ShowResults._add_distances(self.dataframe, point_col='point_col')
        self.assertIn(('distance', 'double'), computed_dataframe.dtypes)

        p_computed_dataframe = computed_dataframe.toPandas()
        actual_distances = [sqrt(2.0), sqrt(8.0), sqrt(18.0)]
        for idx, val in enumerate(actual_distances):
            self.assertEqual(val, p_computed_dataframe['distance'][idx])

    def test_add_outliers(self):
        computed_pdf = ShowResults._add_outliers(self.dataframe).toPandas()

        # Boundary pre calculated mean for prediction 0: mean+2*stddev = 8.37
        actual_values = [False]*5+[True]+4*[False]
        self.assertListEqual(list(computed_pdf['is_outlier']), actual_values)

    def test_compute_summary(self):
        computed_df = ShowResults._add_outliers(self.dataframe)
        summary_pdf = ShowResults.compute_summary(computed_df).toPandas()

        # counts from predictionCol
        actual_count_prediction = [6, 3, 1]
        # counts from outliers in distance
        actual_count_outliers = [1, 0, 0]
        # percentage from actual_count_outliers / actual_count_prediction
        actual_count_percentage = list(map(float, ['%.f' % elem for elem in
                                                   [out/pre*100 for out, pre in
                                                    zip(actual_count_outliers, actual_count_prediction)]]))

        self.assertEqual(list(summary_pdf['count']), actual_count_prediction)
        self.assertEqual(list(summary_pdf['outlier_count']), actual_count_outliers)
        self.assertEqual(list(summary_pdf['outlier percentage']), actual_count_percentage)

    def test_prepare_table_data(self):

        table_df = ShowResults.prepare_table_data(self.dataframe, point_col='point_col').toPandas()
        print(table_df)






