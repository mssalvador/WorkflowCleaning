import numpy as np
import pandas as pd
import math
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import VectorAssembler
from pyspark import SparkContext
from pyspark.ml.linalg import Vector, VectorUDT, DenseVector, DenseMatrix, MatrixUDT


# class LabelPropagation(object):
#
#     def __init__(self, max_iter):
#
#         self.max_iter = max_iter
#
#     def __str__(self):
#         pass
#
#     def __repr__(self):
#         print('LabelProgation( {!s} )'.format(self.max_iter))
#
#
#     def create_

def create_complete_graph(
        data_frame, points=None,
        sigma=0.7):
    """
    Does a cross-product on the dataframe. And makes sure that the "points" are kept as a vector
    points: column names that should be the feature vector
    """
    assert isinstance(points, list), 'this should be a list, not {}'.format(type(points))
    new_cols = list(set(data_frame.columns) - set(points))
    to_array_udf = F.udf(lambda x: x, T.DoubleType())

    feature_gen = VectorAssembler(inputCols=points, outputCol='features')

    df_cleaned = (feature_gen
                  .transform(data_frame)
                  .select(new_cols + [feature_gen.getOutputCol()])
                  )

    def _compute_eucleadian_dist(vec_x, vec_y):
        return float(math.sqrt(np.dot(vec_x-vec_y, vec_x-vec_y)))

    a_names = [F.col(name).alias('a_' + name) for name in df_cleaned.columns]
    b_names = [F.col(name).alias('b_' + name) for name in df_cleaned.columns]

    compute_distance_squared = F.udf(
        lambda x, y: math.exp(- _compute_eucleadian_dist(x, y) ** 2 / sigma ** 2),
        T.DoubleType()
    )
    df_crossed = (df_cleaned
        .select(*a_names)
        .join(df_cleaned.select(*b_names))
        .withColumn(
        'weights_ab',
        compute_distance_squared(
            F.col('a_features'),
            F.col('b_features')
        )
    )
    )

    return df_crossed


def compute_transition_values(
        data_frame, row=None,
        column=None, label=None,
        weight=None, sc=None):

    df_weights = (
        data_frame
            .select(row, column, weight, label)
            .cache()
    )
    df_weights.take(1)

    summed_edge_rows = (df_weights
                        .groupBy(column)
                        .sum(weight)
                        .rdd
                        .map(lambda x: (x[0], x[1]))
                        .collectAsMap()
                        )
    bcast_summed_edge_rows = sc.broadcast(summed_edge_rows)

    edge_normalization = F.udf(
        lambda index, weight: weight / bcast_summed_edge_rows.value[index],
        T.DoubleType()
    )

    df_joined_weights = (
        df_weights
            .withColumn(
            'transition_ab',
            edge_normalization(column, weight)
        )
            .withColumnRenamed(label, 'label')
            .withColumnRenamed(existing=column, new='column')
            .withColumnRenamed(existing=row, new='row')
    )
    bcast_summed_edge_rows
    return df_joined_weights

def generate_transition_matrix(
        data_frame, column_name = None,
        row_name = None, transition_name = None,
        label_name = None):

    temp_n = 4
    temp_k = 2
    sc = SparkContext.getOrCreate()
    bcast = sc.broadcast({'k': temp_k, 'n': temp_n})


    def _sort_by_key(lis):
        return list(map(lambda x: x[1], sorted(lis, key=lambda x: x[0])))

    def _label_by_row(label):
        output = [0.0]*bcast.value['k']
        try:
            output[int(label)] = 1.0
            return output
        except ValueError as ve:
            return output


    udf_sorts = F.udf(lambda x: DenseVector(_sort_by_key(x)), VectorUDT())
    udf_generate_intial_label = F.udf(lambda x: _label_by_row(x), T.ArrayType(T.DoubleType()))

    return (data_frame
            .groupBy(row_name, label_name)
            .agg(F.collect_list(F.struct(column_name, transition_name)).alias('row_trans'))
            .withColumn('row_trans', udf_sorts('row_trans'))
            .withColumn('initial_label', udf_generate_intial_label(label_name))
            .withColumn('is_clamped', ~F.isnan(F.col(label_name)))
            .cache()
            )


def generate_label(
        data_frame=None, label_weights='initial_label',
        bcast=None):
    expression = [F.struct(
        F.lit(i).alias('key'),
        F.col(label_weights)[i].alias('val')
    ) for i in range(bcast.value['k'])]

    return (data_frame
            .withColumn(
        colName='exploded',
        col=F.explode(F.array(expression))
    )
            .select(
        F.col('exploded.key'),
        F.col('exploded.val')
    )
            .groupBy('key')
            .agg(F.collect_list('val').alias('label_vector'))
            .rdd
            .map(lambda x: (x['key'], x['label_vector']))
            .collectAsMap()
            )


def label_propagation(
        data_frame, label_col,
        id_col, feature_cols,
        k=2, sigma=0.7,
        max_iters=5, tol=0.05):
    """
    The actual label propagation algorithm
    """

    sc = SparkContext.getOrCreate()
    n = data_frame.count()

    bcast = sc.broadcast({'k': k, 'n': n})

    df_with_weights = create_complete_graph(
        data_frame=data_frame,
        points=feature_cols,
        sigma=sigma
    )

    ### TODO: perhaps make a truncator, such that we can use sparse vectors instead.
    df_transition_values = compute_transition_values(
        data_frame= df_with_weights,
        row= 'a_'+id_col,
        column=' b_'+id_col,
        label= label_col,
        weight= 'transition_ab',
        sc= sc
    )

    df_transition_matrix = generate_transition_matrix(
        data_frame= df_transition_values,
        column_name= 'column',
        row_name= 'row',
        transition_name= 'transition_ab',
        label_name = 'label'
    )

    df_init_label = generate_label(
        df_transition_matrix.orderBy('row'),
        label_weights= 'initial_label', bcast= bcast
    )
    y_labels = sc.broadcast(df_init_label)

    #Enter the for-loop
    for iteration in range(max_iters):

        udf_dot = F.udf(
            lambda l: [float(l.dot(y_labels.value[i])) for i in range(bcast.value['k'])],
            T.ArrayType(T.DoubleType())
        )

        df_new_label = (df_transition_matrix
                 .withColumn('initial_label',
                             F.when(F.col('is_clamped'), F.col('initial_label')).otherwise(udf_dot('row_trans'))
                             )
                 .orderBy('row')
                 .cache()
                 )

        df_init_label = generate_label(
            df_new_label, label_weights='initial_label',
            bcast= bcast
        )
        y_labels.unpersist()
        print("iteration {} label values \n{}\n{}\n".format(iteration, *y_labels.value.items()))
        y_labels = sc.broadcast(df_init_label)


    return df_new_label