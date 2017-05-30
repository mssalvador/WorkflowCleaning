#Python related imports
from ipywidgets import widgets
from IPython.display import display, Javascript, HTML
import pyspark.ml.clustering as clusters
import traitlets


KMEANS_ALGORITHM = ["random","k-means||"]


class CreateParameters(object):

    '''
    Description:
    
    '''

    def __init__(self, feature_cols=[]):
        self.numberClusters = widgets.IntText(value=50)
        self.featureCols = feature_cols
        self.predictionCols = "Prediction"
        self.initialMode = widgets.Text()
        self.featureColsOutput = widgets.SelectMultiple(options=self.featureCols, value=[])
        self.initialSteps = widgets.IntText(value=10)
        self.iterations = widgets.IntText(value=20)
        self.standardize = widgets.IntText(0)
        self.algorithm = "KMeans"

    def create_dropdown(self, opts, desc, disabled=False):
        '''
        This method contains a general way for handling drop menues.
        :param opts: options in the drop down menu
        :param initial_value: The value is not callable and is set to the first object in opts list.
        :param desc: A suitible title for the drop down
        :param disabled: A binary variable for if the parameter is fixed or not
        :return: a widgets.Select object
        '''
        assert isinstance(opts, list), "opts is of type "+str(type(opts))
        assert isinstance(desc, str), "desc is of type "+str(type(desc))

        return widgets.Select(
            options = opts,
            value = opts[0],
            description = desc,
            disabled = disabled
        )

    def create_multi_select(self, opts, desc, disabled=False):
        '''
        This method contains a general way for handling multiple selection fields aka. we want to choose more collumns
        :param opts: Options in the multiple selection window 
        :param desc: A suitable description of the multi selection window
        :param disabled: A binary variable for the widget if it is present or not 
        :return: 
        '''

        return widgets.SelectMultiple(
            options = opts,
            description = desc,
            disabled=disabled

        )

    def create_integer_slider(self,**kwargs):

        return widgets.IntSlider(description , value=kwargs.get("value", 50), min=kwargs.get("min", 2), max=kwargs.get("max", 100), step=1)


    def select_kmeans(self):

        dic = {}

        dic["initial_mode"] = self.create_dropdown(opts=["random","k-means||"], desc='Methods:', disabled=False)
        dic["select_features"] = self.create_multi_select(opts=self.featureCols,desc="Feature columns",disabled=False)
        dic["standardization_checkbox"] = widgets.Checkbox(value=False, description='Standardization', disabled=False)

        dic["number_clusters"] = widgets.IntSlider(value=self.numberClusters, min=2, max=100, step=1, description='Number of clusters: ',
            disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='i', slider_color='white'
        )
        dic["number_init_steps"] = widgets.IntSlider(value=self.initialSteps, min=2, max=100, step=1, description='Number of initial steps: ',
            disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='i', slider_color='white'
        )
        dic["number_iterations"] = widgets.IntSlider(value=self.iterations, min=2, max=100, step=1, description='Iterations: ',
            disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='i', slider_color='white'
        )

        return dic

    def set_kmeans_columns(self, cluster, initialsteps, iterations, features, method, standard):
        traitlets.link((iterations, 'value'), (self.iterations, 'value'))
        traitlets.link((cluster, 'value'), (self.numberClusters, 'value'))
        traitlets.link((initialsteps, 'value'), (self.initialSteps, 'value'))
        #traitlets.link((method, 'value'), (self.initialMode, 'value'))
        traitlets.link((standard, 'value'), (self.standardize, 'value'))
        traitlets.link((features, 'value'), (self.featureColsOutput, 'value'))

    def select_second_params(self):

        algorithm = widgets.Select(
            options=[i for i in clusters.__all__ if "Model" not in i],
            value=self.algorithm,
            # rows=10,
            description='Clustering methods:',
            disabled=False
        )

        km = widgets.Dropdown(options=KMEANS_ALGORITHM, value=KMEANS_ALGORITHM[0], description="Algorithms", disabled=False)
        slide_iteration = widgets.IntSlider(value=10, min=2, max=100, step=1, description='Iterations: ')
        slide_cluster = widgets.IntSlider(value=10, min=2, max=100, step=1, description='Clusters: ')
        probability = widgets.Select(options=["x", "y", "z"], value="x", description="Probabilty", disabled=False)
        slide_initial_steps = widgets.IntSlider(value=10, min=2, max=100, step=1, description='Inital Steps: ')
        multi_select_features = self.create_multi_select(self.featureCols, "Features", False)
        checkbox_standardize = widgets.Checkbox(value=False, description='Standardization', disabled=False)
        self.initialMode = km
        self.initialSteps = slide_initial_steps
        self.iterations = slide_iteration
        self.numberClusters = slide_cluster
        self.featureColsOutput = multi_select_features
        self.standardize = checkbox_standardize

        # new_box = widgets.HBox([widgets.VBox([multi_select_features, km, slide_iteration]),
        #                         widgets.VBox([checkbox_standardize, slide_cluster, slide_initial_steps])])
        new_box = widgets.HBox([widgets.VBox([multi_select_features, km, slide_iteration]),
                                widgets.VBox([checkbox_standardize, slide_cluster, slide_initial_steps])])
        # self.set_kmeans_columns(cluster=slide_cluster, initialsteps=slide_initial_steps, iterations=slide_iteration,
        #                         features=multi_select_features,
        #                         method=km,
        #                         standard=checkbox_standardize)

        def changes_to_algorithm(change):

            new_line = [slide_cluster]
            if change.new == "KMeans":
                new_line.append(self.initialMode)
                new_line.append(slide_iteration)

                #self.set_kmeans_columns(cluster_slider,slide,slide,slide,slide,slide)
            elif change.new == "LDA":

                new_line.append(probability)

            else:
                raise NotImplementedError

            new_box.children = [i for i in new_line]

        algorithm.observe(changes_to_algorithm, names="value")
        #self.set_kmeans_columns(cluster_slider, probability, slide, slide, slide, slide)

        display(algorithm)
        display(new_box)

    def select_parameters(self):

        algorithm = widgets.Select(
            options = [i for i in clusters.__all__ if "Model" not in i],
            value=self.algorithm,
            # rows=10,
            description='Clustering methods:',
            disabled=False
        )

        def set_slider_values(cluster, initialsteps, iterations):
            self.iterations = iterations
            self.numberClusters = cluster
            self.initialSteps = initialsteps

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

        print(self.numberClusters)
        print(self.initialSteps)
        print(self.iterations)
        print(self.featureCols)

    def export_values(self):
        return {"iterations": self.iterations.value,
                "initialstep": self.initialSteps.value,
                "clusters": self.numberClusters.value,
                "standardize": self.standardize.value,
                "features": self.featureColsOutput.value,
                "prediction": self.predictionCols,
                "model": self.algorithm,
                "initialmode": self.initialMode.value #!!! HARDCODED FOR TESTING PURPOSES !!!
                }




