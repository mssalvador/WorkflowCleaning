#Python related imports
from ipywidgets import widgets
from pyspark.sql import functions as F, Window
from IPython.display import display, Javascript, HTML
import pyspark.ml.clustering as clusters

#TODO: Vi skal finde ud af strukturen i denne klasse. DVS. skal show_*** vise et cluster eller alle?
#TODO: Hvor lægges afstandsberegningen? I ExecuteWorkflow, eller i ShowResults?
#TODO: Hvad skal vi lægge ind i ShowResults klassen?


class ShowResults(object):

    def __init__(self, dict):
        self.data_dict = dict

    def show_outliers(self, dataframe):
        '''
        This method should take all outliers from a specific cluster
        :param dataframe: Spark data frame containing data from a cluster or all clusters? 
        :return: 
        '''
        print("Nothing has been made, yet!")

    def show_prototypes(self, dataframe, cn):
        '''
        This method should take all prototypes from a specific cluster
        :param dataframe: Spark data frame containing data from a cluster or all clusters? 
        :return: 
        '''

        return dataframe.filter(F.col("Prediction")+F.lit(1) == cn-1).withColumn(F.col("Prediction"))

    def show_clusters(self):
        pass

    def select_prototypes(self, dataframe):
        '''
        This method should contain a widget that handles the selection of prototypes.
        The method call show_prototypes. 
        :param: 
        :return: 
        '''
        cluster_holder = widgets.IntText()
        button_prototypes = widgets.Button(description="Show prototypes")
        counter = dataframe.select((F.col("Prediction") + 1).alias("Prediction")).groupBy(F.col("Prediction")).count()\
            .orderBy("Prediction")

        dropdown_prototypes = widgets.Dropdown(
            options = list(map(lambda x: x+1, range(self.data_dict["clusters"]))),
            value = 1,
            description = "Select Cluster",
            disabled = False
        )

        def cluster_number(b):
            cluster_holder.value = dropdown_prototypes.value
            display(self.show_prototypes(dataframe, dropdown_prototypes.value).toPandas())

        button_prototypes.on_click(cluster_number)

        counter.show()
        first_line = widgets.HBox((dropdown_prototypes, button_prototypes))
        display(first_line, cluster_holder)

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