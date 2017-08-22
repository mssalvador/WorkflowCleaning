# Python related imports
from ipywidgets import widgets
from IPython.display import display, Javascript, HTML
import pyspark.ml.clustering as clusters
import traitlets

KMEANS_ALGORITHM = ["random", "k-means||"]
CLUSTER_METHODS = [i for i in clusters.__all__ if "Model" not in i if "BisectingKMeans" not in i]

class CreateParameters(object):
    '''
    Description: This class contains the user interface for selecting _parameters
    '''

    def __init__(self, feature_cols=[]):

        self._cols_feature = feature_cols

        self._cols_prediction = widgets.Text(value="prediction")

        self._algorithm = CLUSTER_METHODS[0] #"KMeans"

        self._initial_mode = widgets.Dropdown(options=KMEANS_ALGORITHM, value=KMEANS_ALGORITHM[0],
                                              description="Algorithms", disabled=False)  #KMeans specific
        self._initial_steps = widgets.IntSlider(value=10, min=2, max=100, step=1, description='Inital Steps: ')   #KMeans specific

        self._iterations = widgets.IntSlider(value=10, min=2, max=100, step=1, description='Iterations: ')

        self._number_clusters = widgets.IntSlider(value=10, min=2, max=100, step=1, description='Clusters: ')

        self._cols_feature_output = widgets.SelectMultiple(options=self._cols_feature, value=self._cols_feature[:1],
                                                           description="Features", disabled=False)

        self._cols_label_output = widgets.SelectMultiple(options=self._cols_feature, value=self._cols_feature[:1],
                                                           description="Label", disabled=False)

        self._standardize = widgets.Checkbox(value=False, description='Standardization', disabled=False)

        self._cols_probability = widgets.Select(options=self._cols_feature+[None], value=self._cols_feature[0], description="Probabilty",
                                                disabled=False)

        self._optimizer = None #LDA Specific
        self._minimum_divisible_cluster_size = widgets.FloatText(value=1.0, description="Minimum Divisble Cluster Size") #Bisecting Kmeans specific
        self._tolerance = widgets.FloatText(value=1e-4, description="Tolerance") # Specific to Gaussian Mixture and Kmeans
        self._seed = widgets.IntText(value=42, description="Seed")

    def select_params(self):

        algorithm = widgets.Select(
            options= CLUSTER_METHODS,
            value=self._algorithm,
            # rows=10,
            description='Clustering methods:',
            disabled=False
        )
        display(algorithm)

        box_column_first = widgets.VBox([self._number_clusters, self._seed, self._cols_feature_output]) # always the same
        box_column_second = widgets.VBox([self._iterations, self._initial_steps, self._tolerance])
        box_column_third = widgets.VBox([self._standardize, self._cols_label_output])

        display(widgets.HBox((box_column_first, box_column_second, box_column_third)))

        def changes_to_algorithm(change):

            self._algorithm = change.new

            if change.new == CLUSTER_METHODS[0]: #Bisecting Kmeans
                second_column = [self._iterations, self._minimum_divisible_cluster_size]
                third_column = [self._standardize]
            elif change.new == CLUSTER_METHODS[1]: #KMeans
                second_column = [self._iterations, self._initial_steps, self._tolerance]
                third_column = [self._standardize, self._initial_mode]
            elif change.new == CLUSTER_METHODS[2]: #Gaussian Mixture
                second_column = [self._iterations, self._cols_probability, self._tolerance]
                third_column = [self._standardize]
            elif change.new == CLUSTER_METHODS[3]: #LDA
                second_column = [self._optimizer]
                third_column = [self._standardize]
            else:
                raise NotImplementedError
            box_column_second.children = [i for i in second_column]
            box_column_third.children = [i for i in third_column]

        algorithm.observe(changes_to_algorithm, names="value")


    def export_values(self):
        return {"iterations": self._iterations.value,
                "initialstep": self._initial_steps.value,
                "clusters": self._number_clusters.value,
                "standardize": self._standardize.value,
                "features": self._cols_feature_output.value,
                "prediction": self._cols_prediction.value,
                "model": self._algorithm,
                "label" : self._cols_label_output.value,
                "probability": self._cols_probability.value,
                "initialmode": self._initial_mode.value,  # !!! HARDCODED FOR TESTING PURPOSES !!!
                "optimizer": self._optimizer,
                "mindivisbleClusterSize": self._minimum_divisible_cluster_size.value,
                "tolerance": self._tolerance.value,
                "seed": self._seed.value
                }
