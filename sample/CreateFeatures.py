#Python related imports
from ipywidgets import widgets
from IPython.display import display, Javascript, HTML
import pyspark.ml.clustering as clusters

class AssembleKmeans(object):

    def __init__(self, feature_cols=[]):
        self.numberClusters = widgets.IntText()
        self.featureCols = feature_cols
        self.predictionCols = widgets.Text()
        self.initialMode = widgets.Text()
        self.featureColsOutput = None
        self.initialSteps = widgets.IntText()
        self.iterations = widgets.IntText()
        self.standardize = False
        self.algorithm = None

    def select_parameters(self):

        initial_mode = widgets.Select(
            options = ["random","k-means||"],
            value='random',
            # rows=10,
            description='Methods:',
            disabled=False
        )

        algorithm = widgets.Select(
            options = [i for i in clusters.__all__ if "Model" not in i],
            value='KMeans',
            # rows=10,
            description='Clustering methods:',
            disabled=False
        )

        feature_select = widgets.SelectMultiple(
            options = self.featureCols,
            value = [self.featureCols[0]],
            description = "Feature columns",
            disabled = False

        )

        standardization_checkbox = widgets.Checkbox(
            value=False,
            description='Standardization',
            disabled=False
        )

        number_clusters = widgets.IntSlider(
            value=50,
            min=2,
            max=100,
            step=1,
            description='Number of clusters: ',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='i',
            slider_color='white'
        )

        number_init_steps = widgets.IntSlider(
            value=10,
            min=2,
            max=100,
            step=1,
            description='Number of initial steps: ',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='i',
            slider_color='white'
        )

        number_iterations = widgets.IntSlider(
            value=10,
            min=2,
            max=100,
            step=1,
            description='Iterations: ',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='i',
            slider_color='white'
        )

        def set_slider_values(cluster, initialsteps, iterations):
            self.iterations.value = iterations
            self.numberClusters.value = cluster
            self.initialSteps.value = initialsteps

        sliders = widgets.interactive(set_slider_values,
                                      cluster = number_clusters,
                                      initialsteps = number_init_steps,
                                      iterations = number_iterations,
                                      ) # the sliders are updated here!

        def set_feature_columns(features, method, standard):
            self.featureColsOutput = features
            self.algorithm = method
            self.standardize = standard

        multiple_feature_select = widgets.interactive(set_feature_columns,
                                                      features = feature_select,
                                                      method = algorithm,
                                                      standard = standardization_checkbox
                                                      ) #The feature columns are selected here!



        #cluster_number_button = widgets.Button(description="Show me the money!")

        firstline = widgets.HBox(multiple_feature_select.children)
        secondline = widgets.HBox(sliders.children)
        # thridline = widgets.HBox([cluster_number_button])

        display(widgets.VBox([firstline, secondline]))
        #  cluster_number_button.on_click(self.on_number_clusters_click)

    def on_number_clusters_click(self, b):

        print(self.numberClusters.value)
        print(self.initialSteps.value)
        print(self.iterations.value)
        print(self.featureCols)

    def export_values(self):
        return {"iterations":self.iterations.value,
                "initialstep":self.initialSteps.value,
                "clusters":self.numberClusters.value,
                "standardize":self.standardize,
                "features": self.featureColsOutput,
                "prediction":self.predictionCols.value,
                "model":self.algorithm,

                "initialmode": "random" #!!! HARDCODED FOR TESTING PURPOSES !!!
                }