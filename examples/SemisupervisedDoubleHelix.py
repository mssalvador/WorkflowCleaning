from shared import create_dummy_data as ccd
from semisupervised.depLabelPropagation import label_propagation
from pyspark.sql import functions as F
from shared.Plot2DGraphs import plot3D


def double_helix(sc, example, label):
    spark_double_helix = ccd.create_spark_data(
        sc, func=ccd.create_double_helix, points_pr_helix=example['n'],
        alpha=example['alpha'], beta=example['beta'], missing=example['missing'])

    # spark_double_helix.show()
    plot3D(spark_double_helix, label ,**example)
    spark_double_helix = spark_double_helix.withColumnRenamed(
        existing='label', new='original_label')
    weight_transition = label_propagation(
        sc=sc, data_frame=spark_double_helix,
        label_col=label, id_col='id', feature_cols='x y z'.split(),
        k=2, max_iters=25, sigma=0.43, )

    result = spark_double_helix.alias('a').join(
        weight_transition.alias('b'), F.col('a.id') == F.col('b.row'),
        how='inner').drop('b.row')
    # result.show()
    plot3D(result, 'label', **example)