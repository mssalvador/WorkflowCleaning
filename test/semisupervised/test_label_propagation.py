from unittest import TestCase
from pyspark.tests import ReusedPySparkTestCase, PySparkTestCase
from pyspark.sql import SparkSession
from shared.context import JobContext
from pyspark.sql import functions as F
from pyspark.sql import types as T
from functools import partial

from pyspark.ml.linalg import DenseVector
import pandas as pd
import numpy as np
from semisupervised import LabelPropagation

class TestCreate_complete_graph(PySparkTestCase):

    def setUp(self):
        super().setUp()
        # self.sc.addPyFile('/home/svanhmic/workspace/DABAI/Workflows/semisupervised/LabelPropagation.py')
        self.spark = SparkSession(self.sc)
        self.spark.conf.set("spark.sql.crossJoin.enabled", "true")
        self.label_context = JobContext(self.sc)
        self.label_context_set = partial(self.label_context.set_constant, self.sc)
        self.label_context_set('k', 2)
        self.label_context_set('sigma', np.sqrt(3)/3)

        id_col = np.array([5, 3, 1, 2, 6,4])
        # np.random.shuffle(id_col)
        data = {
                'label': [0.0, 1.0] + 4 * [None],
                'a': np.array([1.0, 2.0, 3.0, -1.0, -0.5, 4.0]),
                'b': np.array([1.0, 2.0, 3.0, -1.0, -0.5, 4.0]),
                'c': np.array([1.0, 2.0, 3.0, -1.0, -0.5, 4.0]),
                }
        pdf = pd.DataFrame(data, columns=['label', 'a', 'b','c'])
        pdf['id'] = id_col
        self.label_context_set('n',len(pdf['id']))
        self.test_df = self.spark.createDataFrame(pdf)

    def test_jobcontext(self):

        self.assertEqual(self.label_context.constants['k'].value, 2)
    
    def test_cross_joining_length(self):

        df_crossed = LabelPropagation.create_complete_graph(
            data_frame= self.test_df,
            id_col= 'id',
            points= ['a', 'b', 'c']
        )
        pdf_crossed = df_crossed.orderBy('a_id','b_id').toPandas()

        self.assertEqual(df_crossed.count(), self.label_context.constants['n'].value**2)
        #print(pdf_crossed)

    def test_all_dist_in_cross_join(self):
        df_crossed = LabelPropagation.create_complete_graph(
            data_frame=self.test_df,
            id_col='id',
            points=['a', 'b', 'c']
        )
        sigma = self.label_context.constants['sigma'].value
        compute_weights = F.udf(lambda x, y: LabelPropagation._compute_weights(
            x, y, sigma), T.DoubleType())
        pdf_computed_weights = (df_crossed
                                .withColumn('weights_ab', compute_weights('a_features', 'b_features'))
                                .toPandas())

        pdf_computed_weights['actual_weights'] = pdf_computed_weights['a_features']-pdf_computed_weights['b_features']
        pdf_computed_weights['actual_weights'] = pdf_computed_weights['actual_weights'].map(
            lambda x: np.exp(-np.linalg.norm(x)**2/sigma**2))

        # Lets check our distances
        for i in (pdf_computed_weights['weights_ab'] == pdf_computed_weights['actual_weights']).tolist():
            self.assertTrue(i)

    def test_compute_distributed_weights(self):

        df_crossed = LabelPropagation.create_complete_graph(
            data_frame=self.test_df,
            id_col='id',
            points=['a', 'b', 'c']
        )

        pdf_crossed = df_crossed.toPandas().groupby('b_id', as_index=False)['weights_ab'].sum()
        # print(pdf_crossed)
        dict_test_compute_distributed_weights = LabelPropagation.compute_distributed_weights(
            columns= 'b_id', weight_col= 'weights_ab', df_weights= df_crossed)

        pdf_crossed['comp_summed_weights'] = pdf_crossed['b_id'].map(dict_test_compute_distributed_weights)

        for i in (pdf_crossed['comp_summed_weights'] == pdf_crossed['weights_ab']).tolist():
            self.assertTrue(i)

    def test_add_broadcasted_summed_weight(self):
        df_crossed = LabelPropagation.create_complete_graph(
            data_frame=self.test_df,
            id_col='id',
            points=['a', 'b', 'c']
        )
        LabelPropagation.generate_summed_weights(self.label_context_set, df_crossed, column_col='b_id')

        weights_dict = self.label_context.constants['summed_row_weights'].value
        self.assertTrue(isinstance(weights_dict, dict))
        self.assertEqual(self.test_df.count(), len(weights_dict)) # summed by column should render the same length as the original df

    def test_distance_mesaure(self):

        # Dummy testing!
        xd3 = np.array([1.0]*3)
        yd3 = np.array([2.0]*3)
        print('vector x: {} and vector y: {}'.format(xd3, yd3))

        acutual_weight = float(np.exp(-(np.linalg.norm(xd3-yd3, 2)**2)/(self.label_context.constants['sigma'].value**2)))
        computed_weight = LabelPropagation._compute_weights(xd3, yd3, self.label_context.constants['sigma'].value)
        self.assertEqual(acutual_weight,computed_weight)

