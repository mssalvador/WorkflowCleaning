#Python related imports
from ipywidgets import widgets
from pyspark.sql import functions as F, types, Window
from IPython.display import display, clear_output, Javascript, HTML
import pyspark.ml.clustering as clusters
from shared.ComputeDistances import *
from pyspark.ml.linalg import VectorUDT

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

    def show_prototypes(self):
        '''
        This method should take all prototypes from a specific cluster
        :param dataframe: Spark data frame containing data from a cluster or all clusters? 
        :return: 
        '''
        pass

    def show_cluster(self, df):
        distanceUdf = F.udf(lambda x, y: float(np.sqrt(np.sum((x - y) * (x - y)))), types.DoubleType())
        dist = df.select(distanceUdf(df[-2], df[-3]).alias('distance'))
        display(dist.show())
        make_histogram(dist)
        return dist

    def select_prototypes(self, dataframe):
        '''
        This method should contain a widget that handles the selection of prototypes.
        The method call show_prototypes. 
        :param: 
        :return: 
        '''

        button_prototypes = widgets.Button(description="Show prototypes")
        updated_dataframe = dataframe.\
            select('*', (F.col(self.data_dict['prediction']) + 1).alias(self.data_dict['prediction']))\
            .drop(dataframe.prediction)

        counter = updated_dataframe.groupBy(F.col(self.data_dict['prediction']))\
            .count().orderBy(self.data_dict['prediction'])

        dropdown_prototypes = widgets.Dropdown(
            options = list(map(lambda x: x+1, range(self.data_dict["clusters"]))),
            value = 1,
            description = "Select Cluster",
            disabled = False
        )

        def cluster_number(b):
            clear_output()
            cluster_dataframe = updated_dataframe\
                .filter(F.col(self.data_dict['prediction']) == dropdown_prototypes.value)
            self.show_cluster(cluster_dataframe)

        button_prototypes.on_click(cluster_number)

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