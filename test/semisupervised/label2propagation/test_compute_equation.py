from pyspark.sql import SparkSession
from pyspark.sql import dataframe
from pyspark import tests
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.mllib.linalg import VectorUDT
from pyspark.mllib.linalg import Vectors
from semisupervised.label2propagation.equation import compute_equation, create_eye
from semisupervised.label2propagation.split_to_submatrix import to_submatries
import numpy as np
import math

class TestCompute_equation(tests.ReusedPySparkTestCase):
    def setUp(self):
        self.spark = SparkSession(self.sc)

        self.val = 30
        sample = [np.nan, 1, 2]
        prob = [.8, .1, .1]
        data = [{"id":i,
                 "feature": Vectors.dense(np.random.rand(self.val)),
                 "label": float(np.random.choice(sample, 1, replace=True, p=prob))} for i in range(self.val)]
        schema = T.StructType([
            T.StructField("id", T.IntegerType()),
            T.StructField("feature", VectorUDT()),
            T.StructField("label", T.FloatType())])

        self.input = self.spark.createDataFrame(data,schema)
        l = list( filter(lambda x: not math.isnan(x["label"]) ,data))
        self.broad_l = self.sc.broadcast(len(l))
        self.broad_u =  self.sc.broadcast(len(data)-len(l))

        self.T_ll_df, self.T_lu_df, self.T_ul_df, self.T_uu_df = to_submatries(
            df=self.input,
            broadcast_l=self.broad_l, feature="feature")

    def test_create_eye(self):
        n = 10
        b_u = self.sc.broadcast(n)
        eye = create_eye(sc=self.sc, broadcast_u=b_u)
        computed_eye = eye.toLocalMatrix().toArray()
        self.assertEqual(np.linalg.det(computed_eye), 1.0)
        for i in range(n):
            self.assertEqual(computed_eye[i,i], 1.0)
            self.assertEqual(computed_eye[i-1, i], 0.0)


    def test_compute_equation(self):
        # self.T_ll_df.show()
        result = compute_equation(
            sc=self.sc,
            T_uu=self.T_uu_df,
            T_ll=self.T_ll_df,
            T_ul=self.T_ul_df,
            u_broadcast=self.broad_u,
        )

        result.show(30, truncate=False)
