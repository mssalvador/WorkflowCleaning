# This class should execute the kmeans model on the recived data.
import pyspark.ml.clustering as models
import pyspark.ml.feature as features
from pydoc import locate
from pyspark.ml import Pipeline


### SPARK_HOME = "/usr/local/share/spark/python/"

class ExecuteWorkflow(object):
    '''
    
    '''

    # print(models.__all__)

    def __init__(self):
        # , data=None, model="kmeans", params={}

        self._params = None  # dict of parameters including model, mode and so on
        # self._data = None #Perhaps it is needed

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, dic):
        assert isinstance(dic, dict)
        self._params = dic

    def construct_pipeline(self):

        vectorized_features = features.VectorAssembler(inputCols=self._params["features"], outputCol="features")  # vectorization
        if self._params["standardize"]:
            scaling_model = features.StandardScaler(inputCol=vectorized_features.getOutputCol(),
                                                    outputCol="features",
                                                    withMean=True,
                                                    withStd=True
                                                    )
        else:
            scaling_model = None

        cluster_model = getattr(models, self._params["model"])  # Clustering method




        params = [{}]

        stages = list(filter(lambda stage: stage is not None, [vectorized_features, scaling_model, cluster_model]))
        return Pipeline(stages), params

    @staticmethod
    def run(self, data):
        '''
        A pyspark.ml.clustering model
        :param self: 
        :param data: A pyspark data frame with the relevant data
        :return: A fitted model in the pipeline
        '''

        pipeline, params = self.construct_pipeline()
        pipeline.fit(data, params)

