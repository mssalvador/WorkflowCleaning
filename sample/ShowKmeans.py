#Python related imports
from ipywidgets import widgets
from IPython.display import display, Javascript, HTML
import pyspark.ml.clustering as clusters


class ShowResults(object):

    def __init__(self):
        pass

    def show_outliers(self, dataframe):
        '''
        This method should take all outliers from a specific cluster or all clusters??? 
        :param dataframe: Spark data frame containing data from a cluster or all clusters? 
        :return: 
        '''



    def show_prototypes(self, dataframe):
        '''
        
        :param dataframe: Spark data frame containing data from a cluster or all clusters? 
        :return: 
        '''
        pass

    def show_clusters(self):
        pass