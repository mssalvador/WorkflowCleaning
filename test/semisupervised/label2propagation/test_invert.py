from typing import Any, Union

from pyspark import tests
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import types as T
from pyspark.mllib.linalg import VectorUDT, Vectors
import pprint
import itertools as it
import numpy as np
from semisupervised.label2propagation.matrix_inversion import invert

class TestInvert(tests.ReusedPySparkTestCase):
    def setUp(self):
        self.spark = SparkSession(self.sc)

        self.val = 10
        sample = [np.nan, 0, 1]
        prob = [.8, .1, .1]
        data = [{"id":i,
                 "feature": Vectors.dense(np.random.randint(0, high=10, size=self.val)),
                 "label": float(np.random.choice(sample, 1, replace=True, p=prob))} for i in range(self.val)]
        schema = T.StructType([
            T.StructField("id", T.IntegerType()),
            T.StructField("feature", VectorUDT()),
            T.StructField("label", T.FloatType())])

        self.input = (self.spark.
                      createDataFrame(data,schema).
                      sort("label").
                      rdd.
                      zipWithIndex().
                      map(lambda x: [*x[0], x[1]]).
                      toDF(["old_id", "feature", "label", "id"]))

    def test_invert(self):
        inverted_col = "inverted_array"
        output = invert(sc=self.sc, data_frame=self.input, column="feature", output_cols=inverted_col, id_col="id")
        inverted_matrix = np.asarray(list(map(lambda x: x["feature"], self.input.collect())))
        output.show()
        # Is the input data consistent?
        for idx, (a, b) in enumerate(zip(list(map(lambda x: x["feature"].tolist(), self.input.collect())), inverted_matrix)):
            self.assertListEqual(a, b.tolist(), "Error in line {} for Matrix".format(idx))

        # Does each value in both matrices correspond to each other?
        computed_invert_output = it.chain(*list(map(lambda x: x[inverted_col], output.collect())))
        real_inverted_output = it.chain(*np.linalg.inv(inverted_matrix))
        for computed, real in zip(computed_invert_output, real_inverted_output):
            self.assertAlmostEqual(
                computed, real,
                delta=0.0001,
                msg="The computed value is not equal to the real value with 4 decimals"
            )

