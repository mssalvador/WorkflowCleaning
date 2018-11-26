from pyspark.sql import SparkSession
from pyspark.sql import dataframe
from pyspark import tests
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.mllib.linalg import VectorUDT
from pyspark.mllib.linalg import Vectors
from semisupervised.label2propagation.compute_distances import compute_distances
import numpy as np
import math

class TestCompute_distances(tests.ReusedPySparkTestCase):
    def setUp(self):
        self.spark = SparkSession(self.sc)

        self.val = 5
        self.sample = [np.nan, 1, 2]
        prob = [.8, .1, .1]
        data = [{"id":i,
                 "feature": Vectors.dense(np.random.rand(3)),
                 "label": float(np.random.choice(self.sample, 1, replace=True, p=prob))} for i in range(self.val)]
        schema = T.StructType([
            T.StructField("id", T.IntegerType()),
            T.StructField("feature", VectorUDT()),
            T.StructField("label", T.FloatType())])

        self.input = self.spark.createDataFrame(data,schema)

    def test_compute_distances(self):
        self.input.show()
        compute_distances(sc=self.sc, data_frame=self.input).show(truncate=False)
