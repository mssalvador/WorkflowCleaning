from pyspark.sql import SparkSession
from pyspark.sql import dataframe
from pyspark import tests
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.mllib.linalg import VectorUDT
from pyspark.mllib.linalg import Vectors
from semisupervised.label2propagation.label_propagation import label_propagation
import numpy as np
import math

class TestLabel_propagation(tests.ReusedPySparkTestCase):
    def setUp(self):
        self.spark = SparkSession(self.sc)

        self.val = 20
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

    def test_label_propagation(self):
        output = label_propagation(sc=self.sc, data_frame=self.input, features="feature", id="id", label="label")
        output.show(truncate=False)
