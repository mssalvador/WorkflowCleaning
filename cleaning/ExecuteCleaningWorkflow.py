# This class should execute the kmeans model on the recived data.

from pyspark.ml import clustering
import pyspark.ml.feature as features
from pyspark.ml import Pipeline
from shared.ConvertAllToVecToMl import ConvertAllToVecToMl
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import functions as F
import numpy as np
from pyspark.sql.functions import monotonically_increasing_id
import logging
import sys

sc = SparkContext.getOrCreate()

### SPARK_HOME = "/usr/local/share/spark/python/"


logger_execute = logging.getLogger(__name__)
logger_execute.setLevel(logging.DEBUG)
logger_file_handler_parameter = logging.FileHandler('/tmp/workflow_cleaning.log')
logger_formatter_parameter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logger_execute.addHandler(logger_file_handler_parameter)
logger_file_handler_parameter.setFormatter(logger_formatter_parameter)

class ExecuteWorkflow(object):
    """
    Object execute workflow. Builds a spark pipeline based on previous data from other class' and executes the pipeline
    """

    def __init__(self, dict_params=None, cols_features=None, cols_labels=None, standardize=False):
        """
        Constructor for Executeworkflow
        :param dict_params:
        :param cols_features:
        :param cols_labels:
        :param standardize:
        """
        # , data=None, model="kmeans", _params={}

        try:
            self._algorithm = dict_params.pop('algorithm', 'GaussianMixture')
        except AttributeError as ae:
            tb = sys.exc_info()[2]
            logger_execute.warning(ae.with_traceback(tb))
            self._algorithm = 'GaussianMixture'

        self._params = dict_params  # dict of _parameters including model, mode and so on
        self._features = cols_features
        self._labels = cols_labels
        self._standardize = standardize
        # self._data = None #Perhaps it is needed

        self._pipeline, self._params_labels = self.construct_pipeline()
        logger_execute.info('ExecuteWorkflow has been created')

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def params_labs_feats(self):
        return self._params_labels

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, dic):
        assert isinstance(dic, dict)
        self._params = dic

    def construct_pipeline(self):
        """
        Method that creates a spark pipeline.
        :return: pipeline,  labels_features_and_parameters
        """

        vectorized_features = features.VectorAssembler(inputCols=self._features, outputCol="features")  # vectorization

        caster = ConvertAllToVecToMl(inputCol=vectorized_features.getOutputCol(),
                                     outputCol="casted_features")  # does the double and ml.densevector cast

        if self._standardize:
            scaling_model = features.StandardScaler(
                inputCol="casted_features",
                outputCol="scaled_features",
                withMean=True,
                withStd=True
            )
        else:
            scaling_model = features.StandardScaler(
                inputCol="casted_features",
                outputCol="scaled_features",
                withMean=False,
                withStd=False
            )

        caster_after_scale = ConvertAllToVecToMl(inputCol=scaling_model.getOutputCol(),
                                                 outputCol="scaled_features")  # does the double and ml.densevector cast

        dict_params_labels = dict(filter(lambda x: not isinstance(x[1], tuple), self._params.items()))
        dict_params_labels['featuresCol'] = caster_after_scale.getOutputCol()

        # Model is set
        model = eval("clustering." + self._algorithm)(**dict_params_labels)

        # Set the prediction column
        dict_params_labels['prediction'] = model.getPredictionCol()

        # Adds also a probability column if the model is a gaussian mixture.
        try:
            dict_params_labels['probability'] = model.getProbabilityCol()
        except AttributeError:
            logger_execute.warning('Tried to add probability column. Model is not Gaussian Mixture')

        stages = [vectorized_features, caster, scaling_model, model]

        return Pipeline(stages=stages), dict_params_labels

    def execute_pipeline(self, data_frame):
        """
        Executes the pipeline with the dataframe
        :param data_frame: spark data frame that can be used for the algorithm
        :return: model and cluster centers with id
        """
        assert isinstance(data_frame, DataFrame), " data_frame is not of type dataframe but: "+type(data_frame)
        sql_ctx = SQLContext.getOrCreate(sc)
        model = self._pipeline.fit(data_frame)
        transformed_data = model.transform(data_frame)

        if self._algorithm == 'GaussianMixture':
            # convert gaussian mean/covariance dataframe to pandas dataframe
            pandas_cluster_centers = model.stages[-1].gaussiansDF.toPandas()
            centers = sql_ctx.createDataFrame(self.gen_gaussians_center(self._params_labels['k'], pandas_cluster_centers))

            merged_df = transformed_data.join(centers, self._params_labels['prediction'], 'inner')
        else:
            np_centers = model.stages[-1].clusterCenters()
            centers = self.gen_cluster_center(self._params_labels['k'], np_centers)
            broadcast_center = sc.broadcast(centers)

            # Create user defined function for added cluster centers to data frame
            udf_assign_cluster = F.udf(lambda x: Vectors.dense(broadcast_center.value[x]), VectorUDT())

            merged_df = transformed_data.withColumn("centers", udf_assign_cluster(self._params_labels['prediction']))

        # return the result
        return merged_df

    @staticmethod
    def gen_gaussians_center(k, gaussians, prediction_label='prediction'):
        """
        Create a pandas dataframe containing cluster centers (mean) and covariances and adds an id
        :param k: number of clusters
        :param gaussians: pandas dataframe containing mean and covariance
        :param prediction_label: optional variable only if we customize the prediction label
        :return: pandas data frame with id, mean and covariance
        """
        import pandas as pd

        # create dummy id pandas dataframe
        pandas_id = pd.DataFrame({prediction_label: (range(k))}, columns=[prediction_label])
        return pd.concat([gaussians, pandas_id], axis=1)

    @staticmethod
    def gen_cluster_center(k, centers):
        '''
        Create a
        :param k: number of clusters
        :param centers: center of k
        :return: dict with all clusters
        '''
        assert isinstance(k, int), str(k)+" is not integer"
        assert isinstance(centers, list), " center is type: "+str(type(centers))
        return dict(zip(np.array(range(0, k, 1)), centers))


