from pyspark import tests
from pyspark.sql import SparkSession
from examples import SemisupervisedMnist
from pyspark.sql import functions as F
import math

class Test_Semisupervised_Mnist(tests.ReusedPySparkTestCase):
    def setUp(self):
        str_input = '/home/svanhmic/workspace/data/DABAI/mnist/train.csv'
        self.spark = SparkSession(sparkContext=self.sc)
        self.data_frame = self.spark.read.csv(
        path=str_input, header=True, inferSchema=True,
        mode='PERMISSIVE', nullValue=float('NAN'), nanValue=float('NAN'))

    def test_create_nan_labels(self):
        fraction = 0.1
        input_data_frame = self.data_frame.filter(F.col('label').isin([0,1]))
        output_data_frame = SemisupervisedMnist.create_nan_labels(
            self.sc, dataframe=input_data_frame, label_col='label', fraction=fraction)

        # TEST 1: Does it contain missing_*label_name*?
        self.assertIn(member='missing_label', container=output_data_frame.columns)

        # TEST 2: Does the missing_factor correspond to the actual amount of missings?

        computed_fractions = (output_data_frame.filter(~F.isnan('missing_label'))
            .groupBy('missing_label').count().rdd.collectAsMap())

        desired_frac = input_data_frame.groupBy('label').count().collect()
        desired_fractions = dict(map(lambda x: (x['label'], fraction*x['count']), desired_frac))

        for key, val in computed_fractions.items():
            self.assertAlmostEqual(val, desired_fractions[key], delta=input_data_frame.count()*0.01) # 1 percent deviation

    def test_enlarge_dataset(self):
        original_size = 1000
        input_df = self.data_frame.limit(original_size)

        # Test 1: reduce the size to 90
        new_size = 900
        output_data_frame = SemisupervisedMnist.enlarge_dataset(
            dataframe=input_df, size= new_size, feature_cols=['pixel'+str(i) for i in range(784)])
        self.assertAlmostEqual(output_data_frame.count(), new_size, delta=original_size*0.05)

        # Test 2: enlargen to double size
        new_size = 2000
        output_data_frame = SemisupervisedMnist.enlarge_dataset(
            dataframe=input_df, size= new_size, feature_cols=['pixel'+str(i) for i in range(784)])
        self.assertAlmostEqual(output_data_frame.count(), new_size, delta=new_size*0.05)

    def test_subset_dataset_by_label(self):
        # Test 1:
        output_data_frame = SemisupervisedMnist.subset_dataset_by_label(
            self.sc, self.data_frame, 'label',0 ,1, 2)

        distinct_label = output_data_frame.select('label').distinct().collect()
        for val in map(lambda x: x['label'],distinct_label):
            self.assertIn(val, [0,1,2])

    def test_compute_fraction(self):

        # TEST 1: Check for constant fractions
        frac = 0.1
        broad_cast_frac = self.sc.broadcast(frac)
        computed_dict = SemisupervisedMnist._compute_fraction(
            sc=self.sc, dataframe=self.data_frame, fraction=frac,
            label_col='label')

        self.assertListEqual(list1=list(range(10)), list2=list(computed_dict.keys()))
        for key, val in computed_dict.items():
            self.assertEqual(val, 0.1)

        # TEST 2: Check for variable fractions
        actual_fractions = dict(zip(map(lambda x: str(x), range(10)), [0.1, 0.2, 0.3, 0.4, 0.4, 0.5, 0.8, 0.9, 0.01, 0.002]))
        try:
            variable_dict = SemisupervisedMnist._compute_fraction(
                sc=self.sc, dataframe=self.data_frame, label_col='label', **actual_fractions)
        except TypeError as te:
            print(te)
            print(actual_fractions)

        for key, val in variable_dict.items():
            self.assertEqual(first=val, second=variable_dict[key])