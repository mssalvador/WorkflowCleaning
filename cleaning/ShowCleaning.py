"""
Created on 22 August 2017
@author: svanhmic
"""

# import pandas as pd
# import json
# from shared import JSONEncoder

# Sparks imports
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from pyspark.ml.linalg import SparseVector

# Python imports
import functools
import math


# TODO: Vi skal finde ud af strukturen i denne klasse. DVS. skal show_*** vise et cluster eller alle?
# TODO: Hvor lægges afstandsberegningen? I ExecuteWorkflow, eller i ShowResults?
# TODO: Hvad skal vi lægge ind i ShowResults klassen?


class ShowResults(object):
    """
    Object for displaying results from the clustering

    """

    def __init__(self, id, list_features, list_labels, list_headers, **kwargs):
        assert kwargs['predictionCol'] is not None, 'Prediction has not been made'
        assert kwargs['k'] is not None, 'Number of cluster has not been set'
        self._prediction_columns = kwargs['predictionCol']
        self._k_clusters = kwargs['k']
        self._data_dict = kwargs
        self._id = id
        self._features = list_features
        self._labels = list_labels
        self._selected_cluster = 1
        self._headers = list_headers

    # def select_cluster(self, dataframe):
    #     """
    #     Method to decide which cluster to pick!
    #     ### SKAL UNDERSØGES FOR BRUG ###
    #
    #     :return:
    #     """
    #
    #     from ipywidgets import widgets
    #     from IPython.display import display
    #
    #     list_options = ['Cluster ' + str(i+1) for i in range(self._data_dict['k'])]
    #
    #     drop_down_clusters = widgets.Dropdown(
    #         options=list_options,
    #         value=list_options[0],
    #         description="Select a Cluster",
    #         disabled=False)
    #
    #     def observe_cluster_change(change):
    #         if change.new != change.old:
    #             filtered_df = dataframe.filter(
    #                 F.col(self._prediction_columns) == (int(change.new[-1])-1)).select('distance')
    #             ShowResults.show_cluster(filtered_df)
    #
    #     drop_down_clusters.observe(observe_cluster_change, names='value')
    #     display(drop_down_clusters)

    @staticmethod
    def show_cluster(df):
        """
        Visualization of data and outliers in histogram ... 
        TO BE EXPANDED
        
        :param df: Spark data frame
        :return:
        """
        from shared.ComputeDistances import make_histogram

        list_distances = [i["distance"] for i in df.collect()]
        make_histogram(list_distances)  # , self._dimensions)

    @staticmethod
    def _compute_shift(dataframe, **kwargs):
        """
        Adds 1 to the prediction column to have clusters named 1 to n+1, instead of 0 to n
        
        :param dataframe: 
        :param kwargs: prediction_col can be set in the function call, else it will search for 'prediction'
        :return: dataframe with shifted prediction_col
        """
        prediction_col = kwargs.get('prediction_col', 'prediction')
        return dataframe.withColumn(colName=prediction_col, col=F.col(prediction_col) + 1)

    @staticmethod
    def compute_distance(point, center):
        """
        Computes the euclidean  distance from a data point to the cluster center.

        :param point: coordinates for given point
        :param center: cluster center
        :return: distance between point and center
        """
        import numpy as np
        if isinstance(point, SparseVector) | isinstance(center, SparseVector):
            p_d = point.toArray()
            c_d = center.toArray()
            return float(np.linalg.norm(p_d - c_d, ord=2))
        else:
            return float(np.linalg.norm(point - center, ord=2))

    @staticmethod
    def _add_row_index(dataframe, **kwargs):
        """
        Uses pyspark's function monotonically_increasing_id() to add a column with indexes 
        
        :param dataframe: 
        :param kwargs: rowId can be set in the function call, else it will set the column name 'rowId'
        :return: dataframe with added index column
        """
        row_id = kwargs.get('rowId', 'rowId')
        df_stats = dataframe.withColumn(
            colName=row_id,
            col=F.monotonically_increasing_id()
        )
        return df_stats

    @staticmethod
    def _add_distances(dataframe, **kwargs):
        """
        Calculate the distances from points in each cluster to its center
        Uses ComputeDistances which uses the Euclidean distances
        
        :param dataframe: 
        :param kwargs: 
            center_col can be set in the function call, else it will search for 'centers'
            point_col can be set in the function call, else it will search for 'scaled_features'
        :return: dataframe with added distance column 
        """
        # from shared.ComputeDistances import compute_distance
        centers_col = kwargs.get('center_col', 'centers')
        points_col = kwargs.get('point_col', 'scaled_features')
        computed_dist = functools.partial(
            ShowResults.compute_distance
        )
        dist_udf = F.udf(
            f=lambda point, center: computed_dist(point, center),
            returnType=T.DoubleType()
        )
        return dataframe.withColumn(
            colName='distance',
            col=dist_udf(
                F.col(points_col),
                F.col(centers_col)
            )
        )

    @staticmethod
    def _add_outliers(dataframe, **kwargs):
        """
        Calculate a boundary for which a data point will be considered an outlier [bool]
        The boundary is the mean plus "stddev" (number of standard derivations) * the standard derivation
        Uses pyspark's Window function to partition over the special predictions and thereby count number of data 
        points in each cluster, their number of outliers and the outlier percentage 
        
        :param dataframe: 
        :param kwargs: 
            prediction_col can be set in the function call, else it will search for the column name 'predictionCol'
            distance_col can be set in the function call, else it will search for the column name 'distance'
            no_stddev (number of standard derivations) can be set in the function call, else default sat to 2
        :return: dataframe with added 'is_outlier' bool column
        """
        assert kwargs.get('distance_col', 'distance') in dataframe.columns, 'Distances have not been computed!'
        prediction_col = F.col(kwargs.get('prediction_col', 'prediction'))
        distance_col = F.col(kwargs.get('distance_col', 'distance'))
        no_stddev = kwargs.get('no_stddev', 2.0)
        window_outlier = Window().partitionBy(prediction_col)
        computed_boundary = (F.mean(distance_col).over(window_outlier)
                             + no_stddev * F.stddev_pop(distance_col).over(window_outlier)
                             )
        return (dataframe
                .withColumn(colName='computed_boundary',
                            col=computed_boundary)
                .withColumn(colName='is_outlier',
                            col=F.when(distance_col > computed_boundary, 1).otherwise(0))
                )

    # @staticmethod
    # def _add_median(dataframe, ):

    @staticmethod
    def prepare_table_data(dataframe, **kwargs):
        """
        This method should contain a widget that handles the selection of prototypes.
        The method call show_prototypes.
        :param:
        :return:
        """
        # Shift the prediction column, so it goes from n to n+1 we need to persist the dataframe in order to
        # ensure the consistency in the results.
        dataframe_updated = ShowResults._compute_shift(
            dataframe=dataframe, **kwargs
        )
        # Adds an index column (per default called rowId)
        dataframe_updated = ShowResults._add_row_index(
            dataframe=dataframe_updated, **kwargs
        )
        # Calculates the distances to the center point for all the data points in each cluster (default name 'distance')
        dataframe_updated = ShowResults._add_distances(
            dataframe=dataframe_updated, **kwargs
        )
        # Adds 'is_outlier' bool column
        return ShowResults._add_outliers(
            dataframe=dataframe_updated, **kwargs
        )

    @staticmethod
    def create_linspace(data, min, max, buckets, boundary):
        import numpy as np
        outlier_ratio = ShowResults._compute_outlier_ratio(
            data_points=data, boundary=boundary
        )
        outlier_buckets = math.ceil(buckets * outlier_ratio)
        prototypes = list(
            np.linspace(
                start=min, stop=boundary,
                num=buckets - outlier_buckets
            ).tolist()
        )
        outliers = list(
            np.linspace(
                start=boundary, stop=max,
                num=outlier_buckets
            ).tolist()
        )
        bucket_boundary = sorted(prototypes + outliers)
        output = len(bucket_boundary) * [0]
        bucket_outlier = len(prototypes) * [0] + len(outliers) * [1]
        tmp_list = data
        for bucket_idx, bucket_val in enumerate(bucket_boundary[1:]):
            for distance_val in tmp_list:
                if (bucket_idx == 0) and (distance_val < bucket_val):
                    output[bucket_idx] += 1
                elif ((distance_val < bucket_val)
                      and (distance_val >= bucket_boundary[bucket_idx - 1])):
                    output[bucket_idx] += 1
        return list(
            zip(range(len(output)), output, bucket_outlier)
        )

    @staticmethod
    def _compute_outlier_ratio(data_points, boundary):
        outliers = [0. if val < boundary else 1. for val in data_points]
        n_outliers = float(functools.reduce(lambda x, y: x + y, outliers))
        return float(n_outliers / len(data_points))

    @staticmethod
    def make_buckets(distances, ratio, boundary, n_buckets):
        num = n_buckets * [0]
        out = n_buckets * [0]

        for dist in distances:
            for i in range(n_buckets):
                if ratio * i < dist <= ratio * (i+1):
                    num[i] += 1
                    if dist > boundary:
                        out[i] = 1
        return list(zip(range(n_buckets), num, out))

    @staticmethod
    def frontend_result(sc, dataframe, buckets=20, prediction_col='prediction'):
        n_buckets = sc.broadcast(buckets)
        buckets_list_udf = F.udf(
            f=lambda dist, ratio, boundary: ShowResults.make_buckets(
                distances=dist, ratio=ratio, boundary=boundary, n_buckets=n_buckets.value)
            ,
            returnType=T.ArrayType(
                elementType=T.ArrayType(
                    elementType=T.IntegerType(),
                    containsNull=True),
                containsNull=True
            )
        )

        tmp = (dataframe
               .groupBy(prediction_col, F.col('computed_boundary'))
               .agg(F.min('distance').alias('min'), F.max('distance').alias('max'),
                    F.sum('is_outlier').alias('n_outliers'),
                    F.collect_list('distance').alias('distances'))
               .withColumn(colName='ratio', col=F.col('max')/n_buckets.value)
               .withColumn(colName='buckets', col=buckets_list_udf(
                            'distances', 'ratio', 'computed_boundary'))
               )

        return tmp.select(prediction_col, 'buckets')

    @staticmethod
    def create_buckets(sc, dataframe, buckets=20, prediction_col='prediction'):
        n_buckets = sc.broadcast(buckets)
        generate_list_udf = F.udf(
            f=lambda l, minimum, maximum, boundary: ShowResults.create_linspace(
                data=l, min=minimum, max=maximum,
                boundary=boundary, buckets=n_buckets.value
            ),
            returnType=T.ArrayType(
                elementType=T.ArrayType(
                    elementType=T.IntegerType(),
                    containsNull=True),
                containsNull=True
            )
        )
        tmp = (dataframe
               .groupBy(prediction_col, F.col('computed_boundary'))
               .agg(F.min('distance').alias('min'), F.max('distance').alias('max'),
                    F.sum('is_outlier').alias('n_outliers'),
                    F.collect_list('distance').alias('distances'))
               .withColumn(colName='buckets', col=generate_list_udf(
                            'distances', 'min', 'max',
                            'computed_boundary'))
               )
        return tmp.select(prediction_col, 'buckets')

    def arrange_output(self, sc, dataframe,
                       data_point_name='data_points', **kwargs):
        predict_col = kwargs.get('predictionCol', 'Prediction')
        show_buckets = kwargs.get('showBuckets', None)
        new_struct = F.struct(
            [*self._headers,
             'distance', 'is_outlier'
             ]
        ).alias(data_point_name)  # here we loose the rest of the columns... # self._id, *self._labels, *self._features,
        percentage_outlier = F.round(100 * F.col('percentage_outlier') / F.col('amount'), 3)

        bucket_df = ShowResults.frontend_result(
            sc=sc, dataframe=dataframe,
            buckets=20, prediction_col=predict_col)
        # bucket_df = ShowResults.create_buckets(
        #     sc=sc, dataframe=dataframe,
        #     buckets=20, prediction_col=predict_col
        # )
        re_arranged_df = (
            dataframe
            .select(F.col(predict_col), new_struct)
            .groupBy(F.col(predict_col))
            .agg(F.count(predict_col).alias('amount'),
                 F.sum(F.col(data_point_name
                             + ".is_outlier")).alias('percentage_outlier'),
                 F.collect_list(data_point_name).alias(data_point_name))
            .withColumn(colName='amount_outlier',
                        col=F.col('percentage_outlier')
                        )
            .withColumn(colName='percentage_outlier',
                        col=percentage_outlier
                        )
        )

        if show_buckets:
            return re_arranged_df.join(other=bucket_df, on=predict_col, how='inner')

        return re_arranged_df

    # @staticmethod
    # def compute_summary(dataframe, **kwargs):
    #     """
    #     This function creates the summary table for the K-clusters, with their data points, outliers and %-outlier
    #     :param dataframe:
    #     :param kwargs:
    #         prediction_col can be set in the function call, else it will search for 'predictionCol'
    #         outlier_col can be set in the function call, else it will search for 'is_outlier'
    #     :return: Dataframe with Prediction, count, outliers and outlier percentage
    #     """
    #     prediction_col = kwargs.get('prediction_col', 'prediction')
    #     outlier_col = kwargs.get('outlier_col', 'is_outlier')
    #     if prediction_col is None or outlier_col is None:
    #         return None
    #     count_outliers = F.udf(lambda col: int(np.sum(col)), types.IntegerType())
    #
    #     return (dataframe
    #             .groupBy(prediction_col)
    #             .agg(F.count(prediction_col).alias('count'),
    #                  F.collect_list(F.col(outlier_col)).alias('outliers'))
    #             .withColumn(colName='outlier_count',
    #                         col=count_outliers('outliers'))
    #             .withColumn(colName='outlier percentage',
    #                         col=F.round(F.col('outlier_count') / F.col('count') * 100, scale=0))
    #             .withColumnRenamed(existing=prediction_col,
    #                                new='Prediction')
    #             .withColumn(colName='Prediction', col=F.col('Prediction')-1)
    #             .drop('outliers')
    #             )
    #
    #
    # @staticmethod
    # def cluster_graph(dataframe, **kwargs):
    #     """
    #     This method creates a dataframe with table for a cluster histogram
    #     :param dataframe: containing distances and outliers for a specific cluster
    #     :param kwargs:
    #     :return: dataframe
    #     """
    #
    #     dist_out_df = dataframe[['distance', 'is_outlier']]
    #
    #     bins = np.linspace(0, np.max(dataframe.distance), 21)
    #
    #     height = []
    #     is_outlier = []
    #     bin = 0
    #     for i in bins[1:]:
    #         count = 0
    #         for j in range(dist_out_df.shape[0]):
    #             if bin < dist_out_df.iloc[j]['distance'] <= i:
    #                 count += 1
    #                 is_outlier.append(dist_out_df.iloc[j]['is_outlier'])
    #             else:
    #                 continue
    #         height.append(count)
    #         if len(is_outlier) < len(height):
    #             is_outlier.append(False)
    #         bin = i
    #
    #     graph_df = pd.DataFrame({'Bucket': range(1, 21), 'Height': height, 'Is_outlier': is_outlier})
    #
    #     return graph_df

    # @staticmethod
    # def json_histogram(dataframe, **kwargs):
    #     """
    #     Creates the json file with the information to draw the histograms for the different clusters.
    #     Uses cluster_graph
    #     :param dataframe: the prepared_table_data
    #     :param kwargs:
    #     :return: json
    #     """
    #     g = {}
    #     grouped = dataframe.groupby('prediction')
    #
    #     for i in range(1, len(dataframe.prediction.unique()) + 1):
    #         group = grouped.get_group(i)
    #         table = ShowResults.cluster_graph(group)
    #         g['group' + str(i)] = table
    #
    #     return json.dumps(g, sort_keys=True, indent=4, cls=JSONEncoder.JSONEncoder)

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
