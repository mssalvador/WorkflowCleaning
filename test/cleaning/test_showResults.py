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
        self.result = ShowResults(
            self.sc, {'predictionCol': [1, 1], 'k': 2},
            ['feat1', 'feat2'], ['lab1', 'lab2']
        )

        d = {'predictionCol': [1, 1, 2],
             'points' : [DenseVector(np.array([1.0,1.0])),
                         DenseVector(np.array([0.0, 0.0])),
                         DenseVector(np.array([3.0,3.0]))
                         ],
             'centers': [DenseVector(np.array([2.0, 2.0])),
                         DenseVector(np.array([2.0, 2.0])),
                         DenseVector(np.array([6.0,6.0]))
                         ]
             }
        df = pd.DataFrame(data=d)

        self.dataframe = self.spark.createDataFrame(df)

    def test_add_row_index(self):

        computed_dataframe = self.result.add_row_index(self.dataframe)
        self.assertIn(('rowId', 'bigint'), computed_dataframe.dtypes)

        computed_dataframe = self.result.add_row_index(self.dataframe, rowId='roow')
        self.assertIn(('roow','bigint'), computed_dataframe.dtypes)

    def test_add_distances(self):
        from math import sqrt
        computed_dataframe = self.result.add_distances(self.dataframe, feature_col= 'points')
        self.assertIn(('distance', 'double'), computed_dataframe.dtypes)

        p_computed_dataframe = computed_dataframe.toPandas()
        print(p_computed_dataframe)
        actual_distances = [sqrt(2.0), sqrt(8.0), sqrt(18.0)]
        for idx, val in enumerate(actual_distances):
            self.assertEqual(val, p_computed_dataframe['distance'][idx])

    def test_add_outliers(self):
        # self.fail()
        mini_pdf = pd.DataFrame(
            {'predictionCol': [1,1,1,1,1,1,2], 'distance': [0.5, 1.5, 0.5, 0.1, 0.01, 6.0, 20.0]},
            columns=['predictionCol','distance']
        )
        computed_df = self.spark.createDataFrame(mini_pdf)
        # computed_df.show()
        computed_pdf = ShowResults.add_outliers(computed_df).toPandas()

        print(computed_pdf)
        actual_values = [False]*5+[True]+[False]
        self.assertListEqual( list(computed_pdf['is_outlier']), actual_values)






