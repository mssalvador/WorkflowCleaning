"""
Created on 22 August 2017

@author: svanhmic
"""

from pyspark.context import SparkContext
from ipywidgets import widgets
from pyspark.sql import functions as F
from pyspark.sql import types
from scipy.stats import chi2
from IPython.display import display, clear_output, Javascript, HTML
import pyspark.ml.clustering as clusters
from pyspark.ml.linalg import VectorUDT

# TODO: Vi skal finde ud af strukturen i denne klasse. DVS. skal show_*** vise et cluster eller alle?
# TODO: Hvor lægges afstandsberegningen? I ExecuteWorkflow, eller i ShowResults?
# TODO: Hvad skal vi lægge ind i ShowResults klassen?

sc = SparkContext.getOrCreate()


class ShowResults(object):
    """
    Object for displaying results from the clustering

    """

    def __init__(self,
                 dict_parameters,
                 list_features,
                 list_labels):

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
        make_histogram(list_distances)#, self._dimensions)

    def compute_shift(self, dataframe):
        """
        Compute distance, percentage distance to cluster center, and if outlier.
        :param dataframe:
        :return: dataframe
        """
        from pyspark.sql import Window
        from pyspark.sql import functions as F
        import numpy as np
        import math

        # Window function
        win_percentage_dist = (Window
                               .orderBy(F.col('distance').desc())
                               .partitionBy(F.col(self._data_dict['predictionCol'])))

        # Udf's
        percentage_dist = 100-(F.max(F.col('distance')).over(win_percentage_dist)-F.col('distance'))/100
        udf_real_dist = F.udf(
            lambda c, p: float(math.sqrt(np.dot((c.toArray()-p.toArray()), (c.toArray()-p.toArray())))),
            types.DoubleType())

        new_dataframe = dataframe\
            .withColumn(self._data_dict['predictionCol'], F.col(self._data_dict['predictionCol']) + 1)\
            .withColumn('distance', udf_real_dist(dataframe.centers, dataframe.scaled_features))\
            .withColumn('Percentage distance', percentage_dist)

        # boundary = new_dataframe.select(F.mean(F.round(F.col('distance')))).collect()[0][0]
        # print('boundary: ', boundary)
        # new_dataframe = new_dataframe.withColumn('outliers', F.when(F.col('distance') > boundary, 1).otherwise(0))

        return new_dataframe

    @staticmethod
    def _check_outlier(distance_vector, mean):
        outlier_vector = []
        for ids, dist in distance_vector:
            if dist > mean * 2:
                outlier_vector.append((ids, True))
            else:
                outlier_vector.append((ids, False))
        return outlier_vector

    def compute_summary(self, dataframe):
        # df_stats = (dataframe.select(self._data_dict['predictionCol'], 'outliers', 'distance', 'centers')).persist()
        df_stats = (dataframe.select(F.monotonically_increasing_id().alias("rowId"), self._data_dict['predictionCol'],
                                     'distance', 'centers')).persist()

        expr_type = types.ArrayType(types.StructType([types.StructField('idRow', types.IntegerType(), False),
                                                      types.StructField('dist', types.FloatType(), False)]))

        udf_outliers = F.udf(lambda distVec, mean: ShowResults._check_outlier(distVec, mean), expr_type)

        df = df_stats \
            .groupBy(self._data_dict['predictionCol']) \
            .agg(F.count(self._data_dict['predictionCol']).alias("Count"), F.mean(F.col('distance')).alias('meanDist'),
                 F.collect_list(F.struct(F.col('rowId'), F.col('distance'))).alias('vecDist'))\
            .withColumn('outliers', udf_outliers(F.col('vecDist'), F.col('meanDist')))\
            .orderBy(self._data_dict['predictionCol']) \
            .filter(F.col("Count") >= 1) \
            .toPandas()

        display(df)
        print(df.printSchema())

        # .withColumn("Outlier Count", ),
        # F.round(F.sum(F.col("outliers")) / F.count(self._data_dict['predictionCol']) * 100, 1)
        # .alias("% Outlier")) \ \

        df_outliers = (df_stats
                       .select(self._data_dict['predictionCol'], "distance")
                       .distinct()
                       .groupBy(F.col(self._data_dict['predictionCol']))
                       .count()
                       .filter(F.col("count") >= 2)
                       .orderBy(self._data_dict['predictionCol'])
                       )

        # display(df_outliers.toPandas())
        list_clusters_with_outliers = (df_outliers
                                       .select(self._data_dict['predictionCol'])
                                       .collect())

        return list_clusters_with_outliers

    def select_prototypes(self, dataframe, **kwargs):
        """
        This method should contain a widget that handles the selection of prototypes.
        The method call show_prototypes.
        :param:
        :return:
        """

        button_prototypes = widgets.Button(description="Show prototypes")

        # Shift the prediction column with for, so it goes from 1 to n+1 we need to persist the dataframe in order to
        # ensure the consistency in the results.
        dataframe_updated = self.compute_shift(dataframe)

        # create summary for the clusters along with number in each cluster and number of outliers
        # find out how many unique data points we got, meaning that if the distance is equal then we won't display it
        list_unique_values = self.compute_summary(dataframe_updated)

        list_clusters_with_outliers = sorted(map(
            lambda x: x[self._data_dict['predictionCol']], list_unique_values))
        # print(list_clusters_with_outliers)

        dropdown_prototypes = widgets.Dropdown(
            options=list_clusters_with_outliers,
            description="Select Cluster",
            disabled=False
        )

        def selected_cluster_number(b):
            clear_output()
            filter_expr = (F.col(self._data_dict['predictionCol']) == dropdown_prototypes.value)
            cluster_dataframe = dataframe_updated.filter(filter_expr)

            self.show_cluster(cluster_dataframe)
            self._selected_cluster = dropdown_prototypes.value

            # Show only a table containing outliers: This is bad but better than converting to pandas all the time
            output_cols = self._labels + list(self._features) + ['distance', 'Percentage distance', 'outliers']
            # print(output_cols)
            # cluster_dataframe.select(output_cols).show()
            pdf = (cluster_dataframe
                   .select(output_cols)
                   .filter(F.col('outliers') == 1)
                   .orderBy(F.col('distance').desc())
                   .toPandas()
                   )

            if len(pdf) != 0:
                display(pdf)
            else:
                print("There seems to be no outliers in this cluster")

        button_prototypes.on_click(selected_cluster_number)

        first_line = widgets.HBox((dropdown_prototypes, button_prototypes))
        display(first_line)

        return dataframe_updated
