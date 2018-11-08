from unittest import TestCase
from OwnArguments import OwnArgumentParser
from unittest.mock import patch
import sys
import pprint

class TestLittleOwnArg(TestCase):
    def setUp(self):
        self.arguments = ['path/to/my/job.py',
                          '--job cleaning',
                          '--job_args k=2 std=True',
                          '--input_data hdfs:///path/to/data/data.csv']

    def test__set_all_main_args(self):
        with patch.object(sys, 'argv', self.arguments):
            print("argv = {}".format(sys.argv))
            argObj = OwnArgumentParser()
            argObj.add_argument(name='--job', type=str)
            argObj.add_argument(name='--job_args', nargs='*', type=dict)
            argObj.add_argument(name='--input_data', type=str)
            argObj.parse_arguments()
            pprint.pprint(argObj.get_all())
            self.assertTrue(argObj.job == "cleaning")
            # self.assertIsInstance(argObj.job_args, list)

            print("args = {}".format(argObj.all_args))

    def test_add_argument(self):
        argObj = OwnArgumentParser()
        argObj.add_argument(name='--job', type=str)
        self.assertIn('job', argObj.all_args.keys())

    def test_parse_argument(self):
        self.fail()

    def test_cast_to(self):
        args = LittleOwnArg(name='did', nargs='*')
        vals_to_test = ['2', 'True', 'stri']
        self.assertIsInstance(args.cast_to(vals_to_test[0]), int)
        self.assertNotIsInstance(args.cast_to(vals_to_test[0]), float)
        self.assertIsInstance(args.cast_to(vals_to_test[1]), bool)
        self.assertIsInstance(args.cast_to(vals_to_test[2]), str)
