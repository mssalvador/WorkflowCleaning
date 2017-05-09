#This class should execute the kmeans model on the recived data.
import pyspark.ml.clustering as models
import pyspark.ml.feature as features
from pydoc import locate
from pyspark.ml import Pipeline

###SPARK_HOME = "/usr/local/share/spark/python/"

class ExecuteWorkFlow(object):
    '''
    
    '''
    #print(models.__all__)

    def __init__(self):
        #, data=None, model="kmeans", params={}

        self._model = None
        self._params = None
        #self._data = None #Perhaps it is needed

    @property
    def model(self):
        #print("i'm the model property")
        return self._model

    @model.setter
    def model(self,name):

        assert name in models.__all__, 'The name is not an model!'
        self._model = name

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self,dic):
        assert isinstance(dic, dict)
        self._params = dic

    def construct_pipeLine(self):

        if self._params["standardize"]:
            scaling_model = getattr(features,"StandardScaler")
        else:
            scaling_model = None

        vectorized_features = features.VectorAssembler(inputCols=self._params["features"])

        cluster_model = getattr(models, self._model)

        #myclass = locate(str(models)+"."+self._model)
        return Pipeline([vectorized_features, scaling_model, cluster_model])

    @staticmethod
    def run(data):
        '''
        Executes the pipeline
        
        :return: A pyspark.ml.clustering model
        '''
        pass







