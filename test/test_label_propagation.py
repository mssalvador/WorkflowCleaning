from unittest import TestCase
from pyspark.tests import ReusedPySparkTestCase, PySparkTestCase
from pyspark.sql import SparkSession
from shared.context import JobContext
from functools import partial

from pyspark.ml.linalg import DenseVector
import pandas as pd
import numpy as np
from semisupervised import LabelPropagation


class TestCreate_complete_graph(PySparkTestCase):

    def setUp(self):
        super().setUp()
        self.spark = SparkSession(self.sc)
        self.spark.conf.set("spark.sql.crossJoin.enabled", "true")
        self.label_context = JobContext(self.sc)
        self.label_context_set = partial(self.label_context.set_constant, self.sc)
        self.label_context_set('k', 2)

        data = {'id': np.array(range(6)),
                'label': [0.0, 1.0] + 4 * [None],
                'a': np.random.normal(size=6),
                'b': np.random.normal(size=6)
                }
        pdf = pd.DataFrame(data, columns=['id', 'label', 'a', 'b'])
        self.label_context_set('n',len(pdf['id']))
        self.test_df = self.spark.createDataFrame(pdf)


    def test_jobcontext(self):

        self.assertEqual(self.label_context.constants['k'].value, 2)

    def test_generate_transition_matrix_length(self):
        """
        Test transition matrix
        :return:
        """
        df_crossed = LabelPropagation.create_complete_graph(
            data_frame= self.test_df,
            id_col= 'id',
            points= ['a', 'b']
        )
        df_crossed.show()
    
    def test_cross_joining_length(self):

        df_crossed = LabelPropagation.create_complete_graph(
            data_frame= self.test_df,
            id_col= 'id',
            points= ['a', 'b']
        )

        self.assertEqual(df_crossed.count(), self.label_context.constants['n'].value**2)

    def test_differenece(self):

        dict_old_label = {}
        n = 10
        self.label_context_set('n', n)

        for key in range(2):
            dict_old_label[key] = np.random.randint(1, 10, n)

        self.label_context_set('initial_label', dict_old_label)
        self.label_context_set('tol', 0.01)


        dict_new_label = {}
        for key in range(2):
            dict_new_label[key] = dict_old_label[key]+np.array([0.001, 0.3, 0.01 ]+[1.0]*7)


        difference = LabelPropagation._delta_func(self.label_context, dict_new_label)
        print(difference[0])
        self.assertEqual(len(difference), 2)
        self.assertEqual(len(difference[0]), 10)
        self.assertListEqual(difference[0],
                             list(zip(range(n), dict_old_label[0]+np.array([0.001, 0.3, 0.01 ]+[1.0]*7), [True, False, True]+[False]*7)),
                             str(difference[0]))