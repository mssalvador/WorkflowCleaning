#Python related imports
from ipywidgets import widgets
from IPython.display import display, Javascript, HTML
import pyspark.ml.clustering as clusters

#TODO: Vi skal finde ud af strukturen i denne klasse. DVS. skal show_*** vise et cluster eller alle?
#TODO: Hvor lægges afstandsberegningen? I ExecuteWorkflow, eller i ShowResults?
#TODO: Hvad skal vi lægge ind i ShowResults klassen?


class ShowResults(object):

    def __init__(self):
        pass

    def show_outliers(self, dataframe):
        '''
        This method should take all outliers from a specific cluster
        :param dataframe: Spark data frame containing data from a cluster or all clusters? 
        :return: 
        '''
        print("Nothing has been made, yet!")

    def show_prototypes(self, dataframe):
        '''
        This method should take all prototypes from a specific cluster
        :param dataframe: Spark data frame containing data from a cluster or all clusters? 
        :return: 
        '''
        print("Nothing has been made, yet!")

    def show_clusters(self):
        pass

    def select_prototypes(self, dataframe):
        '''
        This method should contain a widget that handles the selection of prototypes.
        The method call show_prototypes. 
        :param: 
        :return: 
        '''

        button_prototypes = widgets.Button(description="Show prototypes")
        button_prototypes.on_click(self.show_prototypes(dataframe))

        dropdown_prototypes = widgets.Dropdown(
            options = [1,2,3],
            value = 1,
            description = "Select Cluster",
            disabled = False
        )

        first_line = widgets.HBox((dropdown_prototypes,button_prototypes))
        display(first_line)

    def select_outliers(self,dataframe):
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