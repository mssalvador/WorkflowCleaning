#Python related imports
from pyspark.context import SparkContext
from ipywidgets import widgets
from pyspark.sql import functions as F
from pyspark.sql import types
from IPython.display import display, clear_output, Javascript, HTML
import pyspark.ml.clustering as clusters
from shared.ComputeDistances import *
from pyspark.ml.linalg import VectorUDT

#TODO: Vi skal finde ud af strukturen i denne klasse. DVS. skal show_*** vise et cluster eller alle?
#TODO: Hvor lægges afstandsberegningen? I ExecuteWorkflow, eller i ShowResults?
#TODO: Hvad skal vi lægge ind i ShowResults klassen?

sc = SparkContext.getOrCreate()


class ShowResults(object):

    def __init__(self, dict):
        self.data_dict = dict
        self.dimensions = len(self.data_dict["features"])
        self.lables = [*self.data_dict["label"]]# TODO Should be part of data dict!!!
        self.boundary = chi2.ppf(0.99, self.dimensions)
        self.selected_cluster = 1

    def show_outliers(self, dataframe):
        '''
        This method should take all outliers from a specific cluster
        :param dataframe: Spark data frame containing data from a cluster or all clusters? 
        :return: 
        '''
        print("Nothing has been made, yet!")

    def show_prototypes(self):
        '''
        This method should take all prototypes from a specific cluster
        :param dataframe: Spark data frame containing data from a cluster or all clusters? 
        :return: 
        '''
        pass

    def show_cluster(self, df):
        '''

        :param df: Spark data frame
        :return:
        '''

        list_distances = [i["distances"] for i in df.collect()]

        make_histogram(list_distances, self.dimensions)

    def select_prototypes(self, dataframe, **kwargs):
        '''
                This method should contain a widget that handles the selection of prototypes.
                The method call show_prototypes. 
                :param: 
                :return: 
                '''

        button_prototypes = widgets.Button(description="Show prototypes")
        udf_distance = F.udf(lambda x, y: float(np.sqrt(np.sum((x - y) * (x - y)))), types.DoubleType())

        # Shift the prediction column with for, so it goes from 1 to n+1 we need to persist the dataframe in order to
        # ensure the consistency in the results.

        dataframe_updated = (dataframe
                             .withColumn(self.data_dict['prediction'], F.col(self.data_dict['prediction'])+1)
                             .withColumn('distances',
                                         udf_distance(dataframe.centers, dataframe.scaled_features))
                             .withColumn('outliers', F.when(F.col('distances') > self.boundary, 1).otherwise(0))
                             .persist()
                             )
        #dataframe_updated.show()

        # broadcast clusters and their center points to each node
        #b = sc.broadcast(dict(list(map(lambda x: (x[0], x[1]), updated_dataframe
        #                               .select(F.col('prediction'), F.col('centers')).distinct().collect()))))

        # create summary for the clusters along with number in each cluster and number of outliers
        list_stats_cols = [self.data_dict['prediction'], "outliers", "distances"]

        dataframe_for_stats = dataframe_updated.select(*list_stats_cols)

        dataframe_counter = (dataframe_for_stats
                             .groupBy(self.data_dict['prediction'])
                             .agg(F.count(self.data_dict['prediction']).alias("Count"), F.sum(F.col("outliers")).alias("Outlier Count"))
                             .orderBy(self.data_dict['prediction'])
                             .filter(F.col("Count") >= 1)
                             )
        dataframe_counter.show()

        # find out how many unique data points we got
        dataframe_unique_values = (dataframe_for_stats
                                   .select("prediction", "distances")
                                   .distinct()
                                   .groupBy(F.col("prediction"))
                                   .count()
                                   .filter(F.col("count") >= 2)
                                   )

        list_clusters_with_outliers = sorted(map(lambda x: x[self.data_dict['prediction']], dataframe_unique_values.collect()))
        #print(list_clusters_with_outliers)

        dropdown_prototypes = widgets.Dropdown(
            options=list_clusters_with_outliers,
            #options=list(map(lambda x: str(x), list([int(i.prediction) for i in counter.collect()]))),
            #value=1,
            description="Select Cluster",
            disabled=False
        )

        def selected_cluster_number(b):
            clear_output()
            cluster_dataframe = (dataframe_updated
                                 .filter(F.col(self.data_dict['prediction']) == dropdown_prototypes.value)
                                 )

            self.show_cluster(cluster_dataframe)
            self.selected_cluster = dropdown_prototypes.value

            # if updated_dataframe\
            #         .filter((F.col(self._parameters['prediction']) == self._selected_cluster) & (F.col('outliers') == 1))\
            #         .count() > 0:
            if cluster_dataframe.filter(F.col('outliers') == 1).count() > 0:

                output_cols = self.lables+list(self.data_dict['features'])+['distances', 'outliers']
                print(output_cols)
                cluster_dataframe.select(output_cols).show()
                display(cluster_dataframe.select(output_cols)
                        .filter(F.col('outliers') == 1)
                        .orderBy(F.col('distances').desc())
                        .toPandas()
                        )
            else:
                print("There seems to be no outliers in this cluster")

        button_prototypes.on_click(selected_cluster_number)

        first_line = widgets.HBox((dropdown_prototypes, button_prototypes))
        display(first_line)

    def select_outliers(self, dataframe):
        '''
        This method should contain a widget that handles the selection of outliers.
        The method call show_outliers. 
        :param: 
        :return: 
        '''
        button_outliers = widgets.Button(description="Show prototypes")
        button_outliers.on_click(self.show_prototypes(dataframe))

        dropdown_outliers = widgets.Dropdown(
            options = [1,2,3],
            value = 1,
            description = "Select Cluster",
            disabled = False
        )

        first_line = widgets.HBox((dropdown_outliers, button_outliers))
        display(first_line)



