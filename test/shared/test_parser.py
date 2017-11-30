from unittest import TestCase
from shared.ParseLogFiles import convert, split_logfile, pairwise

class TestConvert(TestCase):

    def setUp(self):
        self.test_case_1 = [1,'h']
        self.test_case_2 = [1,'h', 23, 'min']
        self.test_case_3 = [32,'h', 33, 'min', 59, 's']

    def test_convert(self):

        result_1 = convert(self.test_case_1)
        result_2 = convert(self.test_case_2)
        result_3 = convert(self.test_case_3)

        self.assertEqual(60**2,
                         result_1,
                         'The calculated time is: {}, the real time is: {}'.format(result_1, 60**2))
        self.assertEqual(60**2+23*60,
                         result_2,
                         'Failed test {}. Calculated time is: {}, the real time is: {}'.format('result_2',result_2, 60**2+23*60))
        self.assertEqual(32*(60**2)+33*60+59,
                         result_3,
                         'The calculated time is: {}, the real time is: {}'.format(result_3, 32*(60**2)+33*60+59))
    def test_split_logfile(self):

        pass

    def test_pairwise(self):

        self.assertEqual([(1,'h')],
                         list(pairwise(self.test_case_1)),
                         'failed got {}'.format(list(pairwise(self.test_case_1))))
        self.assertEqual([(1, 'h'),(23, 'min')],
                         list(pairwise(self.test_case_2)),
                         'failed got {}'.format(list(pairwise(self.test_case_2))))
        self.assertEqual([(32, 'h'), (33, 'min'), (59, 's')],
                         list(pairwise(self.test_case_3)),
                         'failed got {}'.format(list(pairwise(self.test_case_3))))