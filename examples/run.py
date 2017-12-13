from shared import create_dummy_data as ccd
from semisupervised.LabelPropagation import label_propagation
from shared.parse_algorithm_variables import parse_algorithm_variables
from shared.Plot2DGraphs import plot3D
import pyspark.sql.functions as F

default_lp_param = {'method': 'double_helix', 'n': 300,
            'alpha': 1.0, 'beta': 1.0, 'missing': 1}

def run(sc, **kwargs):
    # select the data
    example = parse_algorithm_variables(kwargs.get(
        'algo_params', default_lp_param))
    for key in default_lp_param.keys():
        if key not in example:
            example[key] = default_lp_param[key]

    label =  kwargs.get('labels', 'unknown_label')
    if example['method'] == 'double_helix':
        spark_double_helix = ccd.create_spark_data(
            sc, func=ccd.create_double_helix, points_pr_helix=example['n'],
            alpha=example['alpha'], beta=example['beta'], missing=example['missing'])

        #spark_double_helix.show()
        #plot3D(spark_double_helix, label ,**example)
        spark_double_helix = spark_double_helix.withColumnRenamed(
            existing='label', new='original_label')
        weight_transition = label_propagation(
            sc=sc, data_frame=spark_double_helix,
            label_col=label, id_col='id', feature_cols='x y z'.split(),
            k=2 , max_iters= 20, sigma=0.43, )

        result = spark_double_helix.alias('a').join(
            weight_transition.alias('b'), F.col('a.id') == F.col('b.row'),
            how= 'inner').drop('b.row')
        #result.show()
        plot3D(result, 'label', **example)

    elif example['method'] == 'mnist':
        train_pdf, test_pdf = ccd.load_mnist()
        print(train_pdf)
