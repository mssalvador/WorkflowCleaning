def semi_supervised_batch_single_classifier_generate_approach(data,
                                                              featureCols=None,
                                                              labelCol='used_label',
                                                              predictionCol='prediction',
                                                              *args,
                                                              **kwargs):
    """
    A first approach to a semi-supervised learning method. Uses a k-means combined with logistic regression to find
    the best classification of the data.
    @input: data: spark dataframe with missing lables, but all are missing!
    @input: featureCols:
    @input: labelCol:
    @input: predictionCol:
    returns spark dataframe with classified data, with help from the clustering method
    """
    import numpy as np
    import pandas as pd
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F
    from pyspark.ml import clustering
    from pyspark.ml import feature
    from pyspark.ml import Pipeline
    from pyspark.ml import classification
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

    assert labelCol in data.columns, 'Lables are missing please provide a label column!'
    assert isinstance(data, DataFrame), 'Data is not of type Spark.DataFrame, but {}'.format(type(data))
    assert featureCols is not None, 'Please give a list of features as string!'

    cluster_model = kwargs.get('clusterModel','KMeans') #TODO Future stuf that makes our semi supervised more dynamic
    classification_model = kwargs.get('classificationModel','LogisticRegression')

    k_clusters = (data
                  .filter((F.col(labelCol) != np.NaN))
                  .groupBy(labelCol)
                  .count()
                  .count()
                  )
    print(k_clusters)

    # Feature vectorizer and k-means model is initialized here!
    feature_vector = feature.VectorAssembler(
        inputCols=featureCols,
        outputCol='features')

    k_means = clustering.KMeans(
        featuresCol=feature_vector.getOutputCol(),
        predictionCol='Kmeans_prediction',
        k=k_clusters)

    # Classification begins here!
    log_reg = classification.LogisticRegression(
        featuresCol=feature_vector.getOutputCol(),
        labelCol=k_means.getPredictionCol(),
        predictionCol=predictionCol)

    # Pipeline get assembled here!
    pipeline = Pipeline(stages=[feature_vector, k_means, log_reg])

    # CrossValidation gets build here!
    param_grid = (ParamGridBuilder()
                  .addGrid(log_reg.regParam, [0.1, 0.01])
                  .build()
                  )
    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol=log_reg.getRawPredictionCol(),
        labelCol=k_means.getPredictionCol())

    folds = kwargs.get('folds', 3)

    cross_validator = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=folds)

    evaluated_pipeline = cross_validator.fit(data)
    cluster_fitted_data = evaluated_pipeline.transform(data)
    return cluster_fitted_data