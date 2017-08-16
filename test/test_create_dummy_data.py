from unittest import TestCase
import shared.create_dummy_data as dd
from pyspark.sql import functions as F


class TestCreateOutliers(TestCase):

    sample_size = 20
    label_list = ['header_1', 'header_2', 'header_3']
    feature_list = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    factor = 10
    number_of_outliers = 5

    def test_size(self):
        '''test if the DF is the same size as we asked'''

        df_outliers = dd.create_dummy_data(number_of_samples=TestCreateOutliers.sample_size,
                                           label_names=TestCreateOutliers.label_list,
                                           feature_names=TestCreateOutliers.feature_list,
                                           outlier_factor=TestCreateOutliers.factor,
                                           outlier_number=TestCreateOutliers.number_of_outliers)

        res = df_outliers.count()

        self.assertEqual(res, TestCreateOutliers.sample_size)

    def test_outlier_size(self):
        '''test if the outliers in the DF is the same size as we asked - both for procentage and integers'''

        df_outliers = dd.create_dummy_data(number_of_samples=TestCreateOutliers.sample_size,
                                           label_names=TestCreateOutliers.label_list,
                                           feature_names=TestCreateOutliers.feature_list,
                                           outlier_factor=TestCreateOutliers.factor,
                                           outlier_number=TestCreateOutliers.number_of_outliers)

        res = df_outliers.where(' or '.join(map(lambda f: '(' + f + ' > 1)', TestCreateOutliers.feature_list))).count()

        if isinstance(TestCreateOutliers.number_of_outliers, float):
            self.assertEqual(res, int(TestCreateOutliers.sample_size*TestCreateOutliers.number_of_outliers))
        else:
            self.assertEqual(res, TestCreateOutliers.number_of_outliers)

    def test_size_wo_labels(self):
        '''test if the DF is the same size as we asked, when there's no labels'''

        df_outliers = dd.create_dummy_data(number_of_samples=TestCreateOutliers.sample_size,
                                           label_names=TestCreateOutliers.label_list,
                                           feature_names=TestCreateOutliers.feature_list,
                                           outlier_factor=TestCreateOutliers.factor,
                                           outlier_number=TestCreateOutliers.number_of_outliers)

        res = df_outliers.count()
        self.assertEqual(res, TestCreateOutliers.sample_size)

    def test_size_wo_features(self):
        '''test if the DF is the same size as we asked, when there's no features'''

        df_outliers = dd.create_dummy_data(number_of_samples=TestCreateOutliers.sample_size,
                                           label_names=TestCreateOutliers.label_list,
                                           feature_names=TestCreateOutliers.feature_list,
                                           outlier_factor=TestCreateOutliers.factor,
                                           outlier_number=TestCreateOutliers.number_of_outliers)

        res = df_outliers.count()
        self.assertEqual(res, TestCreateOutliers.sample_size)

    def test_type_wo_features(self):
        '''test if the DF only consists of StringTypes, if we remove features'''

        with self.assertRaises(TypeError) as cm:
            df_outliers = dd.create_dummy_data(number_of_samples=TestCreateOutliers.sample_size,
                                               label_names=['label'],
                                               #feature_names=TestCreateOutliers.feature_list,
                                               outlier_factor=TestCreateOutliers.factor,
                                               outlier_number=TestCreateOutliers.number_of_outliers)

        error_string = "create_dummy_data() missing 1 required positional argument: 'feature_names'"

        self.assertEqual(str(cm.exception), error_string, str(cm.exception)+" is not equal to: "+error_string)

    def test_type_wo_outlierfactor(self):
        '''test if the DF only consists of StringTypes, if we remove features'''

        df_outliers = dd.create_dummy_data(number_of_samples=TestCreateOutliers.sample_size,
                                           label_names=['label'],
                                           feature_names=TestCreateOutliers.feature_list,
                                           #outlier_factor=TestCreateOutliers.factor,
                                           outlier_number=TestCreateOutliers.number_of_outliers)

        res = df_outliers.count()
        self.assertEqual(res, TestCreateOutliers.sample_size)

    def test_type_wo_outliernumber(self):
        '''test if the DF only consists of StringTypes, if we remove features'''

        df_outliers = dd.create_dummy_data(number_of_samples=TestCreateOutliers.sample_size,
                                           label_names=['label'],
                                           feature_names=TestCreateOutliers.feature_list,
                                           outlier_factor=TestCreateOutliers.factor,
                                           #outlier_number=TestCreateOutliers.number_of_outliers)
                                           )

        res = df_outliers.count()
        self.assertEqual(res, TestCreateOutliers.sample_size)

    def test_size_wo_labels_features(self):
        '''test for an error when there's no features or labels'''

        with self.assertRaises(TypeError) as cm:
            dd.create_dummy_data(number_of_samples=TestCreateOutliers.sample_size,
                                 # label_names=estCreateOutliers.label_list,
                                 # feature_names=TestCreateOutliers.feature_list,
                                 # outlier_factor=TestCreateOutliers.factor,
                                 # outlier_number=TestCreateOutliers.number_of_outliers)
                                 )

        error_string = "create_dummy_data() missing 2 required positional arguments: 'feature_names' and 'label_names'"
        self.assertEqual(str(cm.exception), error_string, str(cm.exception)+" is not equal to: "+error_string)

    def test_size_more_outliers(self):
        '''test for an error when total size is less than outliers'''

        with self.assertRaises(SystemExit) as cm:
            dd.create_dummy_data(number_of_samples=10,
                                 label_names=TestCreateOutliers.label_list,
                                 feature_names=TestCreateOutliers.feature_list,
                                 outlier_factor=TestCreateOutliers.factor,
                                 outlier_number=20)

        self.assertEqual(cm.exception.code, "Your total size needs to be bigger then your outlier size")

    def test_make_outliers(self):
        '''test if number of outliers is as expected'''
        df = dd.create_dummy_data(number_of_samples=100,
                                  label_names=TestCreateOutliers.label_list,
                                  feature_names=TestCreateOutliers.feature_list,
                                  )
        outlier_number = 40
        out_df = dd.make_outliers(df, outlier_number, 100, features=["feature_4"])
        res = out_df.where(F.col("feature_4") > 1).count()
        self.assertTrue(outlier_number*0.90 <= res <= outlier_number*1.1, "Numer of outliers is: "+str(res))

    def test_introduce_string_features(self):
        """Test for string to list conversion in check_input_feature_label"""

        df = dd.create_dummy_data(number_of_samples=10,
                                  feature_names="x y z",
                                  label_names="label newLabel")

        columns = df.columns

        self.assertListEqual(columns, ["label", "newLabel", "x", "y", "z"])
