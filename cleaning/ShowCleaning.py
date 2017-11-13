"""
Created on 22 August 2017

@author: svanhmic
"""

from pyspark.context import SparkContext
from ipywidgets import widgets
from pyspark.sql import functions as F
from pyspark.sql import types
import numpy as np
from scipy.stats import chi2
from IPython.display import display, clear_output, Javascript, HTML
import pyspark.ml.clustering as clusters
from pyspark.ml.linalg import VectorUDT

# TODO: Vi skal finde ud af strukturen i denne klasse. DVS. skal show_*** vise et cluster eller alle?
# TODO: Hvor lægges afstandsberegningen? I ExecuteWorkflow, eller i ShowResults?
# TODO: Hvad skal vi lægge ind i ShowResults klassen?


class ShowResults(object):
    """
    Object for displaying results from the clustering

    """

    def __init__(self,
                 sc,
                 dict_parameters,
                 list_features,
                 list_labels):


        assert dict_parameters['predictionCol'] != None, 'Prediction has not been made'
        assert dict_parameters['k'] != None, 'Number of cluster has not been set'
        self.sc = sc.getOrCreate()
        self._prediction_columns = dict_parameters['predictionCol']
        self._k_clusters = dict_parameters['k']

        self._data_dict = dict_parameters
        # self._dimensions = len(list_features)
        self._features = list_features
        self._labels = list_labels
        # self._boundary = chi2.ppf(0.99, self._dimensions)
        self._selected_cluster = 1
        # print(self._data_dict)


    def select_cluster(self):
        """
        Method to decide which cluster to pick!
        ### SKAL UNDERSØGES FOR BRUG ###
        :return:
        """

        from ipywidgets import widgets
        from IPython.display import display

        list_options = ['cluster ' + str(i+1) for i in range(self._data_dict['k'])]

        drop_down_clusters = widgets.Dropdown(
            options=list_options,
            value=1,
            description="Select a Cluster",
            disabled=False)

        display(drop_down_clusters)

    def show_cluster(self, df):
        """
        :param df: Spark data frame
        :return:
        """
        from shared.ComputeDistances import make_histogram

        list_distances = [i["distance"] for i in df.collect()]
        make_histogram(list_distances) # , self._dimensions)

    @staticmethod
    def compute_shift(dataframe, **kwargs):
        """
        Compute distance, percentage distance to cluster center, and if outlier.
        :param dataframe:
        :return: dataframe
        """
        prediction_col = kwargs.get('prediction_col','predictionCol')
        return dataframe.withColumn(colName= prediction_col, col= F.col(prediction_col) + 1)

    @staticmethod
    def add_distances(dataframe, **kwargs):

        from pyspark.sql import functions as F
        from pyspark.sql import types as T
        from shared import ComputeDistances

        centers_col = kwargs.get('center_col','centers')
        features_col = kwargs.get('feature_col', 'scaled_features')
        dist_udf = F.udf(lambda point, center: ComputeDistances.compute_distance(point, center), T.DoubleType())

        return dataframe.withColumn(
            colName= 'distance',
            col=dist_udf(F.col(features_col), F.col(centers_col))
        )

    @staticmethod
    def compute_summary(dataframe, **kwargs):
        prediction_col = kwargs.get('prediction_col', 'predictionCol')
        outlier_col = kwargs.get('outlier_col','is_outlier')

        count_outliers = F.udf(lambda col: int(np.sum(col)),types.IntegerType())
        # percentage_outliers = F.udf(lambda count, outliers: out ,types.DoubleType())

        return (dataframe
                .groupBy(prediction_col)
                .agg(F.count(prediction_col).alias('count'), F.collect_list(F.col(outlier_col)).alias('outliers'))
                .withColumn(colName= 'outlier_count',
                            col= count_outliers('outliers')
                            )
                .withColumn(colName= 'outlier percentage',
                            col= F.round(F.col('outlier_count') / F.col('count') * 100, scale= 0)
                            )
                .withColumnRenamed(existing= prediction_col,
                                   new= 'Prediction'
                                   )
                .drop('outliers')
                )

    @staticmethod
    def add_row_index(dataframe, **kwargs):

        row_id = kwargs.get('rowId','rowId')
        df_stats = dataframe.withColumn(
            colName=row_id, col=F.monotonically_increasing_id())
        return df_stats

    @staticmethod
    def add_outliers(dataframe, **kwargs):

        from pyspark.sql.window import Window
        prediction_col = kwargs.get('prediction_col', 'predictionCol')
        distance_col = kwargs.get('distance_col', 'distance')
        stddev = kwargs.get('stddev', 2.0)
        assert distance_col in dataframe.columns, 'Distances have not been computed!'

        window_outlier = Window().partitionBy(F.col(prediction_col))
        computed_boundary = F.mean(F.col(distance_col)).over(window_outlier) +\
                            stddev*F.stddev_pop(F.col(distance_col)).over(window_outlier)

        return (dataframe
            .withColumn(colName='computed_boundary', col= computed_boundary)
            .withColumn(colName= 'is_outlier',
                        col= F.when(F.col(distance_col) > computed_boundary, True).otherwise(False)))

    @staticmethod
    def prepare_table_data(dataframe, **kwargs):
        """
        This method should contain a widget that handles the selection of prototypes.
        The method call show_prototypes.
        :param:
        :return:
        """

        # Shift the prediction column with for, so it goes from 1 to n+1 we need to persist the dataframe in order to
        # ensure the consistency in the results.
        dataframe_updated = ShowResults.compute_shift(dataframe, **kwargs)
        dataframe_updated = ShowResults.add_row_index(dataframe_updated, **kwargs)
        dataframe_updated = ShowResults.add_distances(dataframe_updated, **kwargs)
        return ShowResults.add_outliers(dataframe_updated, **kwargs)




        # # create summary for the clusters along with number in each cluster and number of outliers
        # # find out how many unique data points we got, meaning that if the distance is equal then we won't display it
        # list_unique_values = self.compute_summary(dataframe_updated)
        #
        # list_clusters_with_outliers = sorted(map(
        #     lambda x: x[self._data_dict['predictionCol']], list_unique_values))
        # # print(list_clusters_with_outliers)
        #
        # dropdown_prototypes = widgets.Dropdown(
        #     options=list_clusters_with_outliers,
        #     description="Select Cluster",
        #     disabled=False
        # )
        #
        # def selected_cluster_number(b):
        #     clear_output()
        #     filter_expr = (F.col(self._data_dict['predictionCol']) == dropdown_prototypes.value)
        #     cluster_dataframe = dataframe_updated.filter(filter_expr)
        #
        #     self.show_cluster(cluster_dataframe)
        #     self._selected_cluster = dropdown_prototypes.value
        #
        #     # Show only a table containing outliers: This is bad but better than converting to pandas all the time
        #     output_cols = self._labels + list(self._features) + ['distance', 'Percentage distance', 'outliers']
        #     # print(output_cols)
        #     # cluster_dataframe.select(output_cols).show()
        #     pdf = (cluster_dataframe
        #            .select(output_cols)
        #            .filter(F.col('outliers') == 1)
        #            .orderBy(F.col('distance').desc())
        #            .toPandas()
        #            )
        #
        #     if len(pdf) != 0:
        #         display(pdf)
        #     else:
        #         print("There seems to be no outliers in this cluster")
        #
        # button_prototypes.on_click(selected_cluster_number)
        #
        # first_line = widgets.HBox((dropdown_prototypes, button_prototypes))
        # display(first_line)
        #
        # return dataframe_updated
