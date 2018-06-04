from tabulate import tabulate
from collections import OrderedDict

#we create a jobContext Class


class JobContext(object):
    def __init__(self,sc):
        self.counters = OrderedDict()
        self.constants = OrderedDict()
        self._init_accumulators(sc) # accumulators are here!
        self._init_shared_data(sc) # broadcast values are here!

    def _init_accumulators(self, sc):
        pass

    def _init_shared_data(self, sc):
        pass

    def initilize_counter(self, sc, name):
        self.counters[name] = sc.accumulator(0)

    def set_constant(self, sc, name, value):
        self.constants[name] = sc.broadcast(value)

    def inc_counter(self, name, value=1):
        if name not in self.counters:
            raise ValueError(
                '{!s} counter was not initialized. ({!s})'.format(
                    name, self.counters.keys()
                )
            )
        self.counters[name] += value

    def print_accumulators(self):
        print(tabulate(
            self.counters.items(),
            self.counters.keys(), tablefmt="simple")
        )

    def print_broadcasts(self):
        print(tabulate(
            self.constants.items(),
            self.constants.keys(), tablefmt='simple')
        )
