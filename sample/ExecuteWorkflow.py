# This class should execute the kmeans model on the recived data.
import pyspark.ml.clustering as models
import pyspark.ml.feature as features
from pyspark.ml import Pipeline
from shared.ConvertAllToVecToMl import ConvertAllToVecToMl
from shared.context import JobContext
from pyspark import SparkContext
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import functions as F
import numpy as np
from pyspark.sql.functions import monotonically_increasing_id


sc = SparkContext.getOrCreate()

### SPARK_HOME = "/usr/local/share/spark/python/"


class ExecuteWorkflow(object):
    '''
    
    '''

    # print(models.__all__)

    def __init__(self):
        # , data=None, model="kmeans", _params={}
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

        caster = ConvertAllToVecToMl(inputCol=vectorized_features.getOutputCol(),
                                     outputCol="casted_features")  # does the double and ml.densevector cast


        if self._params["standardize"]:
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

        cluster_model = getattr(models, self._params["model"])  # Clustering method
        if self._params["model"] == "KMeans":
            cluster_object = cluster_model(
                featuresCol=caster_after_scale.getOutputCol(),
                predictionCol="prediction",#  self._params["prediction"],
                k=self._params["clusters"],
                initMode=self._params["initialmode"],
                initSteps=self._params["initialstep"],
                tol=1e-4,
                maxIter=self._params["iterations"],
                seed=None
            )
        elif self._params["model"] == "BisectingKMeans":
            # cluster_object = cluster_model(
            #     featuresCol=caster_after_scale.getOutputCol(),
            #     predictionCol="prediction",#  self._params["prediction"],
            #     k=self._params["clusters"],
            #     maxIter=self._params["iterations"],
            #     minDivisibleClusterSize=self._params["mindivisbleClusterSize"],
            #     seed=None
            #)
            raise NotImplementedError(str(self._params["model"]) + " is not implemented yet due to an error in Spark 2.0")
        elif self._params["model"] == "GaussianMixture":
            cluster_object = cluster_model(
                featuresCol=caster_after_scale.getOutputCol(),
                predictionCol="prediction",#  self._params["prediction"],
                k=self._params["clusters"],
                #probabilityCol=["probability"],
                tol=1e-4,
                maxIter=self._params["iterations"],
                seed=None
            )
        else:
            raise NotImplementedError(str(self._params["model"])+" is not implemented")

        stages = [vectorized_features, caster, scaling_model, cluster_object]

        return Pipeline(stages=stages)

    def run(self, data):
        '''
        A pyspark.ml.clustering model the method inserts the cluster centers to each data point
        :param self: 
        :param data: A pyspark data frame with the relevant data
        :return: A fitted model in the pipeline
        '''

        pipeline = self.construct_pipeline()
        model = pipeline.fit(data)
        transformed = model.transform(data)

        if self._params["model"] == "KMeans" or self._params["model"] == "BisectingKMeans":
            centers = self.gen_cluster_center(self._params["clusters"], model.stages[-1].clusterCenters())
            broadcast_center = sc.broadcast(centers)
            udf_assign_cluster = F.udf(lambda x: Vectors.dense(broadcast_center.value[x]), VectorUDT())
            return transformed.withColumn("centers", udf_assign_cluster(pipeline.getStages()[-1].getPredictionCol()))
        elif self._params["model"] == "GaussianMixture":
            temp_model = model.stages[-1].gaussiansDF.withColumnRenamed('mean', 'centers')\
                .withColumn('prediction', monotonically_increasing_id())
            return transformed.join(temp_model, 'prediction', "left")
        else:
            raise NotImplementedError(str(self._params["model"]) + " is not implemented")

    def gen_cluster_center(self, k, centers):
        '''
        :param k: number of clusters
        :param centers: center of k
        :return: dict with all clusters
        '''
        assert isinstance(k, int), str(k)+" is not integer"
        assert isinstance(centers, list), " center is type: "+str(type(centers))
        return dict(zip(np.array(range(0, k, 1)), centers))


