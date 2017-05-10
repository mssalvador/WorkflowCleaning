import pyspark
from collections import OrderedDict

class JobContext(object):
    '''
    
    
    '''
    @staticmethod
    def __init__(self,sc):
        self.counters = OrderedDict()
        self._init_accumulators(sc)
        self._init_shared_data(sc)

    @staticmethod
    def _init_accumulators(self, sc):
        pass

    @staticmethod
    def _init_shared_data(self, sc):
        pass

    @staticmethod
    def initialize_counter(self,sc,name):
        self.counters[name] = sc.accumulator(0)

    @staticmethod
    def inc_counter(self,name,value=1):
        if name not in self.counters:
            raise ValueError("%s counter was not initialized. (%s)" % (name,self.counters.keys()))

        self.counters[name] += value

    @staticmethod
    def print_accumulators(self):
        print("aa\n" *2)





