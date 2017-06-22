from unittest import TestCase
import shared.create_dummy_data as dd
from pyspark.ml.linalg import Vectors
from pyspark import SparkContext, SQLContext
from pyspark.sql import functions as F

sc = SparkContext.getOrCreate()
sqlCtx = SQLContext(sc)


class TestCreateOutliers(TestCase):

    def setUp(self):
        self.dummy = dd.DummyData()
        self.column = [(Vectors.dense([0.0, 0.0]),), (Vectors.dense([1.0, 1.0]),),
                       (Vectors.dense([9.0, 8.0]),), (Vectors.dense([8.0, 9.0]),)]
        #self._df = sqlCtx.createDataFrame(self.column, ["features"])
        self.dummy._df = sqlCtx.createDataFrame(self.column, ["features"])
        self.dummy_outlier_factor = 100

    def test_outlier_factor(self):
        self.assertEqual(self.dummy_outlier_factor, 100)

    def test_create_outliers_not_same(self):
        new_column = self.dummy.create_outliers(self.column, self.dummy_outlier_factor)
        self.assertNotEqual(new_column, self.column, str(new_column)+" is equal to: "+str(self.column))
        #
        # def test_create_outliers_as_dummy_outlier_factor(self):
        #     new_column = DummyData.DummyData.create_outliers(self, self.column, self.dummy_outlier_factor)
        #     element_wise_udf = F.udf(lambda x: x)
        #     self.assertEqual(element_wise_udf(new_column), element_wise_udf(self.column*self.dummy_outlier_factor))
