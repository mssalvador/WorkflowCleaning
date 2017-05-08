#Python related imports
from ipywidgets import widgets
from IPython.display import display, Javascript, HTML

class AssembleKmeans(object):

    def __init__(self, feature_cols="features"):
        self.numberClusters = widgets.IntText()
        self.featureCols = feature_cols
        self.predictionCols = widgets.Text()
        self.initialMode = widgets.Text()
        self.initialSteps = widgets.IntText()
        self.iterations = widgets.IntText()
        self.standardize = widgets.IntText()

    def select_parameters(self):

        initial_mode = widgets.Select(
            options = ["random","k-means||"],
            value='random',
            # rows=10,
            description='Methods:',
            disabled=False
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
                                      cluster=number_clusters,
                                      initialsteps=number_init_steps,
                                      iterations=number_iterations)

        cluster_number_button = widgets.Button(description="Show me the money!")

        firstline = widgets.HBox([initial_mode,standardization_checkbox])
        secondline = widgets.HBox(sliders.children)
        thridline = widgets.HBox([cluster_number_button])

        display(widgets.VBox([firstline,secondline,thridline]))
        cluster_number_button.on_click(self.on_number_clusters_click)

    def on_number_clusters_click(self, b):

        print(self.numberClusters.value)
        print(self.initialSteps.value)
        print(self.iterations.value)

    def export_values(self):
        return {"iterations":self.iterations.value,
                "initialstep":self.initialSteps.value,
                "clusters":self.numberClusters.value,
                "standardize":self.standardize.value,
                "features": self.featureCols,
                "prediction":self.predictionCols.value
                }