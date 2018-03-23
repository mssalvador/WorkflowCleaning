import unittest
from unittest.mock import patch
import sys
from shared.OwnArguments import OwnArguments, LittleOwnArg

class TestOwnArguments(unittest.TestCase):
    def setUp(self):
        self.arguments =  [
            '--job', 'cleaning', '--job_args',
            'algorithm=Kmeans', 'k=10', '--features',
            'a', 'b', 'c', '--n', '10']

    def test_parse_argument(self):
        with patch.object(sys, 'argv', self.arguments) as inputs:
            # print(sys.argv)
            test_obj = LittleOwnArg('--job')
            test_obj.parse_argument()
            self.assertEqual(test_obj._name, '--job')
            self.assertEqual(test_obj._values, 'cleaning')
            test_obj = LittleOwnArg('--features', nargs='*')
            test_obj.parse_argument()
            self.assertEqual(test_obj._name, '--features')
            self.assertEqual(test_obj._values, ['a','b','c'])
            self.assertEqual(test_obj._dest, '--features')
            test_obj = LittleOwnArg('--features', nargs='*', dest='features')
            test_obj.parse_argument()
            self.assertEqual(test_obj._name, '--features')
            self.assertEqual(test_obj._values, ['a', 'b', 'c'])
            self.assertEqual(test_obj._dest, 'features')

    def test_extract_sublist(self):
        with patch.object(sys, 'argv', self.arguments) as inputs:
            # print(sys.argv)
            test_obj = LittleOwnArg('--job')
            self.assertListEqual(
                ['--job', 'cleaning'],
                test_obj.extract_sublist(0), msg='--job 1')
            # print(test_obj.extract_sublist(0))
            test_obj = LittleOwnArg('--job_args', nargs=2)
            self.assertListEqual(
                ['--job_args','algorithm=Kmeans','k=10'],
                test_obj.extract_sublist(2), msg='--job_args 2')
            test_obj = LittleOwnArg('--features', nargs='*')
            self.assertListEqual(
                ['--features', 'a', 'b', 'c'],
                test_obj.extract_sublist(5), msg='--features *')
            test_obj = LittleOwnArg('--job_args', nargs='*')
            self.assertListEqual(
                ['--job_args','algorithm=Kmeans','k=10'],
                test_obj.extract_sublist(2), msg='--job-args *')

    def test__set_all_main_args(self):
        with patch.object(sys, 'argv', self.arguments) as inputs:
            # print(sys.argv)
            desired_output = [True, False, True, False, False, True, False, False, False, True, False]
            computed_output = LittleOwnArg._set_all_main_args()
            # print(computed_output)
            self.assertListEqual(desired_output, computed_output)

    def test__correct_type(self):
        with patch.object(sys, 'argv', self.arguments) as inputs:
            # print(sys.argv)
            computed_obj = LittleOwnArg('--n', dest='n', types=int)
            computed_obj.parse_argument()
            self.assertTrue(
                isinstance(computed_obj._values, int),
                msg='wrong type! is not {} - but {}'.format(int, type(computed_obj._values))
            )
            print(computed_obj._values)

    def test_parses_arguments(self):
        with patch.object(sys, 'argv', self.arguments) as inputs:
            print(sys.argv)