from unittest import TestCase

from pyspark.tests import ReusedPySparkTestCase, PySparkTestCase
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from pyspark.sql import functions as F

from cleaning.ShowCleaning import ShowResults


class TestShowResults(PySparkTestCase):

    def setUp(self):
        super().setUp()
        self.spark = SparkSession(self.sc)
        self.result = ShowResults(
            self.sc, {'predictionCol': [1, 1], 'distance': [2, 2]},
            ['feat1', 'feat2'], ['lab1', 'lab2']
        )

    def test_add_row_index(self):
        d = {'predictionCol': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(data=d)
        # df['rowIndex'] = np.array([0, 1])

        dataframe = self.spark.createDataFrame(df)
        computed_dataframe = self.result.add_row_index(dataframe)
        self.assertIn(('rowId', 'bigint'), computed_dataframe.dtypes)

        computed_dataframe = self.result.add_row_index(dataframe, rowId='roow')
        self.assertIn(('roow','bigint'), computed_dataframe.dtypes)

