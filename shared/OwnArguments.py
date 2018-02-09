import sys

class LittleOwnArg(object):
    "container class"
    def __init__(self, name='dummy', nargs=1, types=str, dest=None, helper=None, required=False):
        print(sys.argv)
        self._name = name
        self._nargs = nargs
        self._types = types
        self._dest = dest
        self._helper = helper
        self._required = required
        self._values = None

    @staticmethod
    def _set_all_main_args(definition='--'):
        return [definition in arg for arg in sys.argv]

    @staticmethod
    def _correct_type(value, types=str):
        try:
            return types(value)
        except TypeError as te:
            print('something be wrong!')
            return value
        except Exception as e:
            print('something be really wrong!')

    def parse_argument(self):
        if self._name in sys.argv or self._required:
            idx = sys.argv.index(self._name)
            sub_list = self.extract_sublist(idx)
        else:
            return None

        if self._nargs != '*' and len(sub_list) == 2: #the short one
            self._values = LittleOwnArg._correct_type(sub_list[1], self._types)
        else:
            self._values = sub_list[1:]

        if self._dest == None:
            self._dest = self._name

    def extract_sublist(self, idx):
        if self._nargs == '*':
            main_args_after = LittleOwnArg._set_all_main_args()[idx + 1:]
            try:
                odx = main_args_after.index(True)
            except ValueError as Ve:
                odx = len(main_args_after)
        else:
            odx = self._nargs
        sub_list = sys.argv[idx:idx + odx + 1]
        return sub_list


class OwnArguments(object):
    def __init__(self):
        self._all_arguments = []

    def __str__(self):
        return 'OwnArguments({})'.format(self.__dict__)

    def __repr__(self):
        return 'Called'

    def add_argument(self, name='dummy', nargs=1, types=str, dest=None, helper=None, required=False):
        self._all_arguments.append(LittleOwnArg(
            name=name, nargs=nargs, types=types, dest=dest, helper=helper, required=required))

    def parse_arguments(self):
        for arg in self._all_arguments:
            arg.parse_argument()
            self.__dict__.update({arg._dest:arg._values})
        return self._all_arguments





