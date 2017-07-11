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
                                           labels=TestCreateOutliers.label_list,
                                           features=TestCreateOutliers.feature_list,
                                           outlier_factor=TestCreateOutliers.factor,
                                           outlier_number=TestCreateOutliers.number_of_outliers)

        res = df_outliers.count()

        self.assertEqual(res, TestCreateOutliers.sample_size)

    def test_outlier_size(self):
        '''test if the outliers in the DF is the same size as we asked - both for procentage and integers'''

        df_outliers = dd.create_dummy_data(number_of_samples=TestCreateOutliers.sample_size,
                                           labels=TestCreateOutliers.label_list,
                                           features=TestCreateOutliers.feature_list,
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
                                           #labels=estCreateOutliers.label_list,
                                           features=TestCreateOutliers.feature_list,
                                           outlier_factor=TestCreateOutliers.factor,
                                           outlier_number=TestCreateOutliers.number_of_outliers)

        res = df_outliers.count()
        self.assertEqual(res, TestCreateOutliers.sample_size)

    def test_size_wo_features(self):
        '''test if the DF is the same size as we asked, when there's no features'''

        df_outliers = dd.create_dummy_data(number_of_samples=TestCreateOutliers.sample_size,
                                           labels=TestCreateOutliers.label_list,
                                           #features=TestCreateOutliers.feature_list,
                                           outlier_factor=TestCreateOutliers.factor,
                                           outlier_number=TestCreateOutliers.number_of_outliers)

        res = df_outliers.count()
        self.assertEqual(res, TestCreateOutliers.sample_size)

    def test_type_wo_features(self):
        '''test if the DF only consists of StringTypes, if we remove features'''

        df_outliers = dd.create_dummy_data(number_of_samples=TestCreateOutliers.sample_size,
                                           labels=['label'],
                                           #features=TestCreateOutliers.feature_list,
                                           outlier_factor=TestCreateOutliers.factor,
                                           outlier_number=TestCreateOutliers.number_of_outliers)

        res = [f.dataType for f in df_outliers.schema.fields]

        self.assertEqual(str(res), '[StringType]')

    def test_size_wo_labels_features(self):
        '''test for an error when there's no features or labels'''

        with self.assertRaises(SystemExit) as cm:
            dd.create_dummy_data(number_of_samples=TestCreateOutliers.sample_size,
                                           #labels=estCreateOutliers.label_list,
                                           #features=TestCreateOutliers.feature_list,
                                           outlier_factor=TestCreateOutliers.factor,
                                           outlier_number=TestCreateOutliers.number_of_outliers)

        self.assertEqual(cm.exception.code, "You must provide at least labels or features as a dictionary")

    def test_size_more_outliers(self):
        '''test for an error when total size is less than outliers'''

        with self.assertRaises(SystemExit) as cm:
            dd.create_dummy_data(number_of_samples=10,
                                           labels=TestCreateOutliers.label_list,
                                           features=TestCreateOutliers.feature_list,
                                           outlier_factor=TestCreateOutliers.factor,
                                           outlier_number=20)

        self.assertEqual(cm.exception.code, "Your total size needs to be bigger then your outlier size")

