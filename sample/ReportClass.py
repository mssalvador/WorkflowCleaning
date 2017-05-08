


#Python related imports
from ipywidgets import widgets
from IPython.display import display, Javascript, HTML
import re


#Spark related imports
from pyspark.sql import functions as F, SQLContext
from pyspark.ml import Pipeline, PipelineModel
from pyspark import SparkContext

# TODO: Need to remember includes


class ReportClass(object):
    def __init__(self, path, stages=None):

        # Import the data

        self.filePath = path
        self.buttonDf, self.featCols = dataImport(self.filePath)
        self.stages = stages
        self.pipeModel = PipelineModel(stages=self.stages)
        self.parameters = None
        self.outputSliderVariables = widgets.IntText()
        self.nOutliers = 5
        self.nPrototypes = 5
        self.clusterN = widgets.Text()
        # self.buttonDf.show(10)

    # TODO
    def test_import(self):
        importButton = widgets.Button(description="Import")

    def set_stages(self, stages):
        self.stages = stages

    def f(self, x):
        self.outputSliderVariables.value = x

    def test_operation(self):
        buttonTest = widgets.Button(description="Train model")
        fileText = widgets.Text(value=self.filePath, disabled=False)

        a_slider = widgets.IntSlider(min=2, max=100, step=1, value=2)
        z = widgets.interact(self.f, x=a_slider)
        display(widgets.HBox([fileText, buttonTest]))

        buttonTest.on_click(self.onClickTrain)

    def computeDists(self):
        centers = dict(list(map(lambda x: (x[0], x[1]), enumerate(self.pipeModel.stages[-1].clusterCenters()))))
        clusterCenters = sc.broadcast(centers)

        addCenterUDf = F.udf(lambda x: Vectors.dense(clusterCenters.value[x]), VectorUDT())
        distanceUdf = F.udf(lambda x, y: float(np.sqrt(np.sum((x - y) * (x - y)))), DoubleType())

        return (self.buttonDf
                .withColumn(col=addCenterUDf(F.col("prediction")), colName="center")
                .withColumn(col=distanceUdf(F.col("scaledFeatures"), F.col("center")), colName="distance")
                )

    def onClickPrototypes(self, b):

        print("The report is shown here!")

        # print(centers)

        cluster = int(re.sub(r'Cluster ', "", self.clusterN.value))

        self.getClosest2Center(cluster).show(truncate=False)

    def onClickOutliers(self, b):
        '''
            This method
        '''

        cluster = int(re.sub(r'Cluster ', "", self.clusterN.value))
        self.getOutliersFromCenter(cluster).show(truncate=False)

    def showOf(self):
        '''
            This method 

        '''
        secondOutlierButton = widgets.Button(description="Show Outliers")
        secondPrototypeButton = widgets.Button(description="Show Prototypes")
        z = self.browseClusters().widget.children[0]

        l = traitlets.link((z, "value"), (self.clusterN, "value"))

        display(widgets.HBox([secondPrototypeButton, secondOutlierButton]))

        secondOutlierButton.on_click(self.onClickOutliers)
        secondPrototypeButton.on_click(self.onClickPrototypes)

    def browseClusters(self):
        """
            Browse clusters one by one

        """

        # %matplotlib qt
        df = (self
              .getDistances()
              # .filter(F.col("prediction") == int(re.sub(r'Cluster ','',i)))
              .select("prediction", "distance"))

        availbleClusters = (df
                            .dropDuplicates()
                            .groupBy("prediction")
                            .count()
                            .filter(F.col("count") > 1)
                            .select("prediction")
                            .orderBy("prediction")
                            .collect()
                            )

        clusterList = ["Cluster " + str(i[0]) for i in availbleClusters]

        def viewImage(i):
            fig = plt.figure(figsize=plt.figaspect(0.75))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel("Distance")
            ax.set_title(i)
            pDf = df.filter(F.col("prediction") == int(re.sub(r'Cluster ', '', i))).select("distance").toPandas()
            sb.distplot(pDf, ax=ax, rug=True, norm_hist=False, kde=False)
            plt.show()

        return widgets.interact(viewImage, i=clusterList)

    def showDistances(self):

        '''
             Show all cluster histograms!

        '''

        buttonDistance = widgets.Button(description="Show distances")
        buttonDistance.on_click(self.onClickShowDistances)
        display(buttonDistance)

    def getClosest2Center(self, cluster):
        '''
            This method extracts the companies closest to its cluster center.
        '''

        rankingWindow = (Window
                         .partitionBy(F.col("prediction"))
                         .orderBy(F.col("distance").asc()))

        return (self.buttonDf
                .withColumn(col=F.rank().over(rankingWindow), colName="rank")
                .filter((F.col("rank") <= self.nPrototypes) & (F.col("prediction") == cluster))
                .select("cvrNummer", "distance", "prediction", "navn")
                .orderBy(F.col("prediction")))

    def getOutliersFromCenter(self, cluster):
        '''
            This method extracts the companies furtherst to its cluster center.
        '''

        rankingWindow = (Window
                         .partitionBy(F.col("prediction"))
                         .orderBy(F.col("distance").desc()))

        return (self.buttonDf
                .withColumn(col=F.rank().over(rankingWindow), colName="rank")
                .filter((F.col("rank") <= self.nOutliers) & (F.col("prediction") == cluster))
                .select("cvrNummer", "distance", "prediction", "navn")
                .orderBy(F.col("prediction")))

    def onClickTrain(self, b):

        self.parameters = ({kmeans.k: self.outputSliderVariables.value, kmeans.initMode: "random"})
        print("You have selected: " + str(self.outputSliderVariables.value) + " number of clusters.")
        pipeline = Pipeline(stages=self.stages)

        # take time start
        t0 = time.time()

        # fit the model
        self.pipeModel = pipeline.fit(self.buttonDf, params=self.parameters)
        self.buttonDf = self.pipeModel.transform(self.buttonDf)
        self.buttonDf = self.computeDists()
        # end time
        t1 = time.time()
        print("Done with training, it took: " + str(t1 - t0) + " seconds to do the training")

    def getDistances(self):
        '''
            Returns the distances for each group  with cvrNummer, distances and prediction

            Input:
                The stuff.

            Output:
                spark dataframe
        '''

        cols = ["cvrNummer", "prediction", "distance"]
        # self.buttonDf = self.computeDists()
        if ("prediction" in self.buttonDf.columns) and ("distance" in self.buttonDf.columns):
            return self.buttonDf.select(cols)
        else:
            return None

    def onClickShowDistances(self, b):

        nGroups = self.outputSliderVariables.value
        cols = 4  # fixed for the moment
        rows = int(self.outputSliderVariables.value / cols)  # likewis
        print(rows)

        df = self.getDistances().select("prediction", "distance").cache()

        nPlots = (df
                  .dropDuplicates()
                  .groupby("prediction")
                  .count()
                  .filter(F.col("count") > 1)
                  .select("prediction")
                  .orderBy("prediction")
                  .collect()
                  )

        nPlots = list(map(lambda x: x[0], nPlots))

        sb.set(color_codes=True)
        col = 4
        rows = int(len(nPlots) / col) + 1

        fig, axes = plt.subplots(ncols=col, nrows=rows, figsize=(20, 20))
        x = (df
             # .dropDuplicates(["cvrNummer","label","distance"])
             .toPandas())
        df.unpersist()

        a = axes.ravel()
        for idx, val in enumerate(nPlots):
            # print(v)

            pandasDf = x[x["prediction"] == val]
            a[idx].set_title("Cluster " + str(val))
            a[idx].set_xlabel("distance")
            sb.distplot(pandasDf[["distance"]], ax=a[idx], kde=True, rug=True)
        fig.subplots_adjust(hspace=1)
        plt.show()