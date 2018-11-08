import sys
from string import digits

class OwnArgumentParser(object):

    all_args = {}

    def __init__(self, name=None, type=str, required=False, nargs=1):
        if nargs == '*' and type == str:
            type = list
        if name:
            dest = name.replace('--', '')
            self.all_args[dest] = {"type":type, "required":required, "nargs":nargs}

    def get_all(self):
        for name in OwnArgumentParser.all_args.keys():
            print("{}: {}".format(name, getattr(self, name)))

    def parse_arguments(self):
        stringed_args = str(" ".join(sys.argv)).split("--")[1:]
        params = list(filter(lambda x: x != "",map(lambda arg: arg.rstrip().split(' '), stringed_args)))
        for p in params:
            stored_value = self.all_args.get(p[0], None)
            if stored_value:
                if isinstance(stored_value["nargs"], str) and stored_value["type"] == dict:
                    splitted_sub_vals = map(lambda x: x.split("="), p[1:])
                    sub_vals = dict((key, OwnArgumentParser.cast_to(val)) for key, val in splitted_sub_vals)
                    setattr(self, p[0], sub_vals)
                elif isinstance(stored_value["nargs"], str) and stored_value["type"] == list:
                    setattr(self, p[0], [OwnArgumentParser.cast_to(val) for val in p[1:]])
                else:
                    setattr(self, p[0], stored_value["type"](p[1]))

    @classmethod
    def add_argument(cls, name=None, type=str, required=False, nargs=1):
        return cls(name=name, type=type, required=required, nargs=nargs)

    @staticmethod
    def cast_to( value: str):
        if value in digits:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        elif value.lower() in ("true", 'yes', 't'):
            return True
        elif value.lower() in ("false", 'no', 'f'):
            return False
        else:
            return value