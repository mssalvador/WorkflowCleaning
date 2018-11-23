from pyspark.tests import ReusedPySparkTestCase
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql import functions as F
from semisupervised.label2propagation.create_label_matrix import determine_class
from semisupervised.label2propagation.create_label_matrix import label_matrix

import numpy as np
import math

class TestLabel_matrix(ReusedPySparkTestCase):
    def setUp(self):
        spark = SparkSession(self.sc)
        data = [{"n_id": i, "label": float(np.random.choice([float("nan"), 1., 2.], size=1, p=[0.33, 0.33, 0.34]))} for i in range(10)]
        schema = T.StructType([T.StructField("n_id", T.IntegerType()), T.StructField("label", T.DoubleType())])
        self.df = spark.createDataFrame(data, schema)

    def test_label_matrix(self):
        self.df.show()
        broadcast_count = self.sc.broadcast(self.df.filter(~F.isnan("label")).distinct().count())
        output = label_matrix(data_frame=self.df, broadc_classes=broadcast_count, label_col="label")
        l_output = output.collect()
        for label, vector in map(lambda x: (x["label"], x["vector_labels"].tolist()) ,l_output):
            if math.isnan(label):
                self.assertListEqual(vector, [0.0]*broadcast_count.value)
            else:
                l = [0.0]*broadcast_count.value
                l[int(label)-1] = 1.0
                self.assertListEqual(vector, l)


    def test_determine_class(self):
        first_cls = determine_class(1, 2)
        self.assertListEqual([1,0], first_cls.tolist())

        second_cls = determine_class(2,5)
        self.assertListEqual([0, 1, 0, 0, 0], second_cls.tolist())

        no_cls = determine_class(float("nan"), 3)
        self.assertListEqual([0]*3, no_cls.tolist())