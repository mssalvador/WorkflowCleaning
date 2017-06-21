#Python related imports
from pyspark.context import SparkContext
from ipywidgets import widgets
from pyspark.sql import functions as F, types, Window
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
        self.lables = [self.data_dict["label"]]# TODO Should be part of data dict!!!
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
        make_histogram(df.select(df.distances), self.dimensions)

    def select_prototypes(self, dataframe, **kwargs):
        '''
                This method should contain a widget that handles the selection of prototypes.
                The method call show_prototypes. 
                :param: 
                :return: 
                '''

        button_prototypes = widgets.Button(description="Show prototypes")

        updated_dataframe = dataframe. \
            select('*', (F.col(self.data_dict['prediction']) + 1).alias(self.data_dict['prediction'])) \
            .drop(dataframe.prediction)

        # broadcast clusters and their center points to each node
        b = sc.broadcast(dict(list(map(lambda x: (x[0], x[1]), updated_dataframe
                                       .select(F.col('prediction'), F.col('centers')).distinct().collect()))))

        distanceUdf = F.udf(lambda x, y: float(np.sqrt(np.sum((x - y) * (x - y)))), types.DoubleType())
        updated_dataframe = (updated_dataframe
                             .withColumn('distances', distanceUdf(updated_dataframe.centers, updated_dataframe.scaled_features))
                             .withColumn('outliers', F.when(F.col('distances') > self.boundary, 1).otherwise(0))
                             )

        counter = updated_dataframe.groupBy(F.col(self.data_dict['prediction'])) \
            .agg(F.count(F.lit(1)).alias("Count"), F.sum(F.col("outliers")).alias("Outlier Count"))\
            .orderBy(self.data_dict['prediction'])\
            .filter(F.col("Count") > 1)

        dropdown_prototypes = widgets.Dropdown(
            #options=list(map(lambda x: x+1, range(self.data_dict["clusters"]))),
            options=list(map(lambda x: str(x), list([int(i.prediction) for i in counter.collect()]))),
            #value=1,
            description="Select Cluster",
            disabled=False
        )

        def selected_cluster_number(b):
            clear_output()
            cluster_dataframe = updated_dataframe \
                .filter(F.col(self.data_dict['prediction']) == dropdown_prototypes.value)

            self.show_cluster(cluster_dataframe)
            self.selected_cluster = dropdown_prototypes.value

            # if updated_dataframe\
            #         .filter((F.col(self.data_dict['prediction']) == self.selected_cluster) & (F.col('outliers') == 1))\
            #         .count() > 0:
            if cluster_dataframe.filter(F.col('outliers') == 1).count() > 0:

                output_cols = list(self.lables)+list(self.data_dict['features'])+['distances', 'outliers']

                display(cluster_dataframe.select(*output_cols)\
                    .filter(F.col('outliers') == 1)\
                    .orderBy(F.col('distances').desc())
                    .toPandas())
            else:
                print("There seems to be no outliers in this cluster")

        button_prototypes.on_click(selected_cluster_number)

        counter.show()
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

        first_line = widgets.HBox((dropdown_outliers,button_outliers))
        display(first_line)
