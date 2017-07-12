from unittest import TestCase
from shared.GeneralDataImport import GeneralDataImport

class TestGeneralDataImport(TestCase):

    def test_extract_type_1(self):
        self.assertEqual(GeneralDataImport.extract_type('/home/svanhmic/workspace/data/test.csv'), '.csv'
                         , 'output is: {}'.format(GeneralDataImport.extract_type('/home/svanhmic/workspace/data/test.csv')))

    def test_extract_type_2(self):
            self.assertEqual(GeneralDataImport.extract_type('/home/svanhmic/works.pace/data/test.csv'), '.csv'
                             , 'output is: {}'.format(GeneralDataImport.extract_type('/home/svanhmic/workspace/data/test.csv')))

    def test_import_data_1(self):
        data_import = GeneralDataImport('/home/svanhmic/workspace/data/DABAI/test.csv')
        test_imported_data = list(map(lambda x: (int(x[0]), int(x[1])), data_import.import_data().collect()))
        self.assertListEqual(test_imported_data, [(1, 2), (3, 4), (5, 6)])

    def test_import_data_2(self):
        data_import = GeneralDataImport('/home/svanhmic/workspace/data/DABAI/test.csv')
        test_imported_headers = data_import.import_data().columns
        self.assertListEqual(test_imported_headers, ['f_1', 'f_2'])



    def test_select_feature_columns(self):
        #self.fail()
        pass

    def test_select_label_columns(self):
        #self.fail()
        pass
