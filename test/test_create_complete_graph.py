from unittest import TestCase
from pyspark.tests import ReusedPySparkTestCase, PySparkTestCase
from pyspark.sql import SparkSession
from pyspark.ml.linalg import DenseVector
import pandas as pd
import numpy as np
from semisupervised import LabelPropagation

class TestCreate_complete_graph(PySparkTestCase):


    def test_complete_graph_no_points(self):

        spark = SparkSession(self.sc)

        data = {'id' : [1., 2.], 'a' : list(np.random.rand(2)), 'b' : list(np.random.rand(2))}
        p_data_frame = pd.DataFrame(data, columns=['id', 'a', 'b'])

        data_frame = spark.createDataFrame(data=p_data_frame)

        self.assertRaises(AssertionError, LabelPropagation.create_complete_graph, data_frame, None)

    def test_complete_graph_cross_join(self):
        spark = SparkSession(self.sc)
        spark.conf.set("spark.sql.crossJoin.enabled", "true")
        data = {'id' : [1., 2.], 'a' : list(np.random.rand(2)), 'b' : list(np.random.rand(2))}
        p_data_frame = pd.DataFrame(data, columns=['id', 'a', 'b'])

        data_frame = spark.createDataFrame(data=p_data_frame)
        df_result = LabelPropagation.create_complete_graph(data_frame=data_frame, points=['a', 'b'])
        dict_result = df_result.select('a_id','b_id').rdd.map(lambda x: (x[0],x[1])).collectAsMap()

        self.assertEqual(
            dict_result,
            {1:1, 1:2, 2:1, 2:2})


    def test_proper_sorts(self):

        spark = SparkSession(self.sc)
        indexs = list(range(3))*3
        weights = list(range(9))
        data = {'row': sorted(indexs),
                'column': indexs,
                'transition_ab': weights}
        p_data_frame = pd.DataFrame(data,columns=['row','column','transition_ab'])
        data_frame = spark.createDataFrame(p_data_frame)
        #print(p_data_frame)
        result = LabelPropagation.generate_transition_matrix(data_frame,'column','row','transition_ab')
        dict_result = result.rdd.map(lambda x: (x[0],x[1])).collectAsMap()
        desired_output = dict(zip(range(3),[DenseVector(range(i*3, i*3+3)) for i in range(0, 3)]))

        self.assertEqual(dict_result, desired_output)

    def test_generate_label_matrix(self):
        """tests if the definition outputs the right elements"""

        data = {'id': np.linspace(0,4,4,dtype='int'),
                'label': [0.0, 1.0, None, None],
                'a': np.random.normal(size=4),
                'b': np.random.normal(size=4)
                }

        spark = SparkSession(self.sc)
        pdf = pd.DataFrame(data, columns=['id', 'label', 'a', 'b'])
        test_df = spark.createDataFrame(pdf)
        y_c, y_u = LabelPropagation.generate_label_matrix(data_frame=test_df, id_col='id', label_col='label')
        y_c.show()
        y_u.show()

