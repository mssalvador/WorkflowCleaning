from pyspark.tests import PySparkTestCase
from pyspark.sql import SparkSession
from shared.context import JobContext
from pyspark.sql import functions as F
from functools import partial
from itertools import product

from pyspark.ml.linalg import DenseVector, SparseVector
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
        self.label_context_set('sigma', 0.5)

        id_col = np.array([0,1,2,3])
        # np.random.shuffle(id_col)
        data = {
                'label': [0.0, 1.0] + 2 * [None],
                'a': np.array([0., 0.9, 0.1, 0.85]),
                'b': np.array([0., 0.9, 0.1, 0.85]),
                'c': np.array([0., 0.9, 0.1, 0.85]),
                }
        pdf = pd.DataFrame(data, columns=['label', 'a', 'b','c'])
        pdf['id'] = id_col
        self.label_context_set('n',len(pdf['id']))
        self.test_df = self.spark.createDataFrame(pdf)

        x_r = [1, 0.00006007, 0.88692, 0.00017]
        y_r = [0.00006007, 1, 0.00046, 0.97045]
        z_r = [0.88692, 0.00046, 1, 0.00117]
        v_r = [0.00017, 0.97045, 0.00117, 1]
        self.results = [x_r, y_r, z_r, v_r]

    def test_jobcontext(self):
        self.assertEqual(self.label_context.constants['k'].value, 2)
    
    def test_cross_joining_length(self):

        df_crossed = LabelPropagation.create_complete_graph(
            data_frame= self.test_df,
            id_col= 'id',
            points= ['a', 'b', 'c'],
            sigma= self.label_context.constants['sigma'].value,
            standardize= False
        )
        pdf_crossed = df_crossed.orderBy('a_id','b_id').toPandas()

        self.assertEqual(df_crossed.count(), self.label_context.constants['n'].value**2)
        #print(pdf_crossed)

    def test_all_dist_in_cross_join(self):
        sigma = self.label_context.constants['sigma'].value
        n = self.label_context.constants['n'].value
        df_crossed = LabelPropagation.create_complete_graph(
            data_frame=self.test_df,
            id_col='id',
            points=['a', 'b', 'c'],
            sigma= sigma,
            standardize= False
        )
        for idx, val in enumerate(df_crossed.select('weights_ab').toPandas().values):
            jdx = idx % n
            self.assertAlmostEqual(val[0], self.results[int(jdx)][int(idx/4)], n)

    def test_compute_distributed_weights(self):

        df_crossed = LabelPropagation.create_complete_graph(
            data_frame=self.test_df,
            id_col='id',
            points=['a', 'b', 'c'],
            sigma= self.label_context.constants['sigma'].value,
            standardize= False
        )
        dict_test_compute_distributed_weights = LabelPropagation.compute_distributed_weights(
            columns= 'a_id', weight_col= 'weights_ab', df_weights= df_crossed)
        print(dict_test_compute_distributed_weights)
        list_actual_computed_weights = [1.88715007, 1.97097007, 1.88858, 1.97178]

        self.assertEqual(len(dict_test_compute_distributed_weights), self.label_context.constants['n'].value)

        for idx, val in dict_test_compute_distributed_weights.items() :
            self.assertAlmostEqual(val, list_actual_computed_weights[idx], 4)

    def test_add_broadcasted_summed_weight(self):
        df_crossed = LabelPropagation.create_complete_graph(
            data_frame=self.test_df,
            id_col='id',
            points=['a', 'b', 'c'],
            sigma=self.label_context.constants['sigma'].value,
            standardize= False
        )
        LabelPropagation.generate_summed_weights(self.label_context_set, df_crossed, column_col='b_id')
        list_actual_computed_weights = [1.88715007, 1.97097007, 1.88858, 1.97178]

        weights_dict = self.label_context.constants['summed_row_weights'].value
        self.assertTrue(isinstance(weights_dict, dict))
        self.assertEqual(self.test_df.count(), len(weights_dict)) # summed by column should render the same length as the original df
        for i,v in weights_dict.items():
            self.assertAlmostEqual(v, list_actual_computed_weights[i], 4)

    def test_generate_transition_mat(self):
        actual_results = [[0.529900, 0.000030, 0.469622, 0.000086],
                          [0.000032, 0.507366, 0.000245, 0.492164],
                          [0.469979, 0.000234, 0.529498, 0.000593],
                          [0.000090, 0.492369, 0.000635, 0.507156]]


        df_crossed = LabelPropagation.create_complete_graph(
            data_frame=self.test_df,
            id_col='id',
            points=['a', 'b', 'c'],
            sigma=self.label_context.constants['sigma'].value,
            standardize= False
        )
        LabelPropagation.generate_summed_weights(self.label_context_set, df_crossed, column_col='b_id')
        # print(self.label_context.constants['summed_row_weights'].value)

        initial_weights = df_crossed.select('a_id', 'b_id','weights_ab').toPandas()
        for i in range(len(initial_weights)):
            computed_value = LabelPropagation.compute_transition_values(
                self.label_context,
                initial_weights.loc[i]['weights_ab'],
                initial_weights.loc[i]['b_id']
            )
            actual_value = actual_results[int(initial_weights.loc[i]['a_id'])][int(initial_weights.loc[i]['b_id'])]
            self.assertAlmostEqual(computed_value, actual_value, 4)

    def test_distance_mesaure(self):

        # Dummy testing!
        x = np.array([0., 0., 0.,])
        y = np.array([0.9, 0.9, 0.9,])
        z = np.array([0.1, 0.1, 0.1,])
        v = np.array([0.85, 0.85, 0.85,])
        data = [x,y,z,v]
        sigma = self.label_context.constants['sigma'].value

        for i,j in product(range(4),range(4)):
            computed_weight = LabelPropagation._compute_weights(data[i], data[j], sigma)
            self.assertAlmostEqual(self.results[i][j], computed_weight, 5)


        # Check for sparse data
        sparse_data = [SparseVector(3,[],[]),
                       DenseVector(y),
                       DenseVector(z),
                       DenseVector(v)]

        for i,j in product(range(4),range(4)):
            computed_weight = LabelPropagation._compute_weights(sparse_data[i], sparse_data[j], sigma)
            self.assertAlmostEqual(self.results[i][j], computed_weight, 5)

    # Test label generation

    def test_label_by_row(self):
        initial_labels = [0, 1, 2, 3, 0, None, None]
        actual_labels = [[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0]]
        k = 4
        gen_label_row = partial(LabelPropagation._label_by_row, k)
        computed_label_vec = map(gen_label_row, initial_labels)
        for idx, vec in enumerate(list(computed_label_vec)):
            self.assertListEqual(actual_labels[idx], vec)

    def test_label_gen(self):

        actual_labels = [[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0]]

        pdf = pd.DataFrame(actual_labels, columns=['x', 'y', 'z', 'v'])
        df = self.spark.createDataFrame(pdf).withColumn('initial_label', F.array('x','y','z','v'))

        self.label_context_set('k', 4)
        computed_label = LabelPropagation.generate_label(self.label_context, df)

        t_pdf = pdf.transpose()
        for idx, val in computed_label.items():
            self.assertListEqual(val, list(t_pdf.iloc[idx,:]))

    # Test one iteration

    def test_one_iteration(self):
        actual_new_label = [
            [1.0, 0.0], [0.0, 1.0],
            [0.46984, 0.00023], [0.00009, 0.49225]]

        computed_labels = LabelPropagation.label_propagation(
            self.sc, self.test_df, 'label', 'id',
            ['a', 'b', 'c'], k= 2, sigma= 0.5, max_iters= 1,
            standardize= False
        )
        pandas_comp_labels = computed_labels.toPandas()

        for idx, vec in enumerate(actual_new_label):
            computed_value = list(pandas_comp_labels['initial_label'][idx])
            for jdx ,val in enumerate(vec):
                self.assertAlmostEqual(val, computed_value[jdx], 4)

        print(computed_labels.toPandas())