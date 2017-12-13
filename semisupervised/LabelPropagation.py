import numpy as np
from pyspark.sql import types as T
from pyspark.ml.linalg import VectorUDT, DenseVector
from pyspark.sql import functions as F
from shared import context
import math
from functools import partial, reduce
from shared.WorkflowLogger import logger_info_decorator, logger
from semisupervised.LP_Graph import create_complete_graph
from semisupervised.ClassMassNormalisation import class_mass_normalization

@logger_info_decorator
def generate_summed_weights(context, weights, **kwargs):

    """
    Creates a broadcast variable containing weights
    :param context:
    :param weights:
    :param kwargs:
    :return:
    """
    weight_col = kwargs.get('weight_col', 'weights_ab')
    columns = kwargs.get('column_col', 'column')
    broadcast_name = kwargs.get('broadcast', 'summed_row_weights')

    summed_weights = compute_distributed_weights(columns, weight_col, weights)
    context(broadcast_name, summed_weights)

@logger_info_decorator
def compute_distributed_weights(columns, weight_col, df_weights):
    summed_weights = (df_weights.groupBy(columns).sum(weight_col)
                      .rdd.map(lambda x: (x[0], x[1])).collectAsMap())
    return summed_weights

@logger_info_decorator
def compute_transition_values(context, weight, index):
    return weight / context.constants['summed_row_weights'].value[index]

def _sort_by_key(lis):
    return list(map(lambda x: x[1], sorted(lis, key=lambda x: x[0])))

@logger_info_decorator
def _label_by_row(k, label):
    output = [0.0]*k
    try:
        output[int(label)] = 1.0
        return output
    except TypeError as te:
        return [float(1.0/k)]*k
    except ValueError as ve:
        return [float(1.0/k)]*k

@logger_info_decorator
def _compute_entropy(context, **kwargs):
    labels = kwargs.get('labels','initial_label')
    return -reduce(lambda a,b: a+b, map(
        lambda vector: np.sum(np.multiply(vector, np.log(vector))),
        context.constants[labels].value)
                   )

@logger_info_decorator
def compute_sum_of_non_clamped_transitions(transition_row):
    # print(math.isnan(transition_row[0][2]))
    candidates = map(lambda x: float(x[1]), filter(lambda x: math.isnan(x[2]), transition_row))
    return reduce(lambda x,y:x+y, candidates)

@logger_info_decorator
def compute_convergence_iter(transition_row, tol, max_iter):
    for iters in range(max_iter-1,0,-1):
        if transition_row ** iters >= tol:
            #print(transition_row ** iters)
            return iters+1
    return 1

@logger_info_decorator
def generate_label(context, data_frame= None,
                   label_weights= 'initial_label'):

    expression = [F.struct(
        F.lit(i).alias('key'),
        F.col(label_weights)[i].alias('val')
    ) for i in range(context.constants['k'].value)]

    df_exploded = explode_dataframe(data_frame, expression)

    return (df_exploded.groupBy('key').agg(F.collect_list('val').alias('label_vector'))
            .rdd.map(lambda x: (x['key'], x['label_vector'])).collectAsMap()
            )

@logger_info_decorator
def explode_dataframe(data_frame, expression):
    df_exploded = (data_frame.withColumn(
        colName='exploded', col=F.explode(F.array(expression))
    )
                   .select(F.col('exploded.key'),
                           F.col('exploded.val'))
                   )
    return df_exploded

@logger_info_decorator
def _multiply_labels(label, broadcast_label, k):
    return [float(label.dot(broadcast_label[i] )) for i in range(k)]

@logger_info_decorator
def label_propagation(
        sc, data_frame, label_col= 'label',
        id_col= 'id', feature_cols = None,
        k= 2, sigma= 0.7, max_iters= 5,
        tol= 0.05, standardize= True,
        eval_type= 'max', priors= None):
    """
    The actual label propagation algorithm
    :param:eval_type: 'Type of after evaluation; two are supported
    max: maximum likelihood
    cmn: Class mass normalisation
    """
    # initial stuff sets up a jobcontext for shared values here.
    label_context = context.JobContext(sc)
    label_context_set = partial(label_context.set_constant, sc)
    label_context_set('k', k)
    label_context_set('n', data_frame.count())
    label_context_set('tol', tol)
    label_context_set('priors', priors)

    # Lets make a proper dataset
    df_with_weights = create_complete_graph(
        data_frame= data_frame, points= feature_cols,
        id_col= 'id', sigma= sigma, standardize= standardize)

    #renaming the columns
    df_transition_values = df_with_weights.select(
        F.col('a_'+id_col).alias('row'), F.col('b_'+id_col).alias('column'),
        F.col('weights_ab'), F.col('a_'+label_col).alias('row_label'),
        F.col('b_'+label_col).alias('column_label')
    ).cache()
    df_transition_values.take(1)

    generate_summed_weights(
        context= label_context_set, weights= df_transition_values)

    # udf's
    edge_normalization = F.udf(lambda column, weight: compute_transition_values(
        label_context, weight= weight, index= column), T.DoubleType()
    )
    udf_sorts = F.udf(lambda x: DenseVector(_sort_by_key(x)), VectorUDT())
    udf_generate_initial_label = F.udf(lambda x: _label_by_row(
        label_context.constants['k'].value, x), T.ArrayType(T.DoubleType()))
    udf_summation = F.udf(lambda col: float(np.sum(col)), T.DoubleType())
    udf_normalization = F.udf(lambda vector, norm: DenseVector(vector.toArray() / norm), VectorUDT())
    udf_convergence_sum = F.udf(lambda x: compute_sum_of_non_clamped_transitions(x),
                                T.DoubleType())
    udf_find_max_iter = F.udf(lambda x: compute_convergence_iter(
        x, label_context.constants['tol'].value, max_iters), T.IntegerType())

    df_normalized_transition_values = df_transition_values.withColumn(
        colName= 'transition_ab', col= edge_normalization('column', 'weights_ab')
    )
    convert_None_to_nan_expr = F.when(
        F.col('row_label') == None, value= np.NaN).otherwise(F.col('row_label'))

    df_aggregated_by_trans = (df_normalized_transition_values
                              .groupBy('row','row_label')
                              .agg(F.collect_list(F.struct('column', 'transition_ab', 'column_label')).alias('row_trans'))
                              )

    df_generate_normed_transitions = (
        df_aggregated_by_trans
            .withColumn(colName= 'converge_summed_transition',
                        col= udf_convergence_sum('row_trans'))
            .withColumn(colName= 'row_trans', col= udf_sorts('row_trans'))
            .withColumn(colName= 'label', col= convert_None_to_nan_expr)
            .withColumn(colName= 'initial_label',
                        col= udf_generate_initial_label('label'))
            .withColumn(colName= 'is_clamped', col= ~F.isnan('label'))
            .withColumn(colName= 'summed_transition',
                        col= udf_summation(F.col('row_trans')))
            .withColumn(colName= 'converge_summed_transition',
                        col= F.col('converge_summed_transition')/F.col('summed_transition')
                        )
            .withColumn(colName= 'max_iteration',
                        col= udf_find_max_iter(F.col('converge_summed_transition'))
                        )
            .withColumn(colName= 'row_trans',
                        col= udf_normalization('row_trans', 'summed_transition'))
    )

    df_transition_matrix = df_generate_normed_transitions.select(
        'row', 'label', 'initial_label', 'row_trans', 'is_clamped', 'max_iteration'
    ).orderBy('row').cache()
    # df_transition_matrix.printSchema()
    # df_transition_matrix.show(truncate=False)
    dict_initial_label = generate_label(
        context= label_context, data_frame= df_transition_matrix,
        label_weights= 'initial_label'
    )
    #
    convergence_iteration = (df_transition_matrix
                             .filter(~F.col('is_clamped'))
                             .agg(F.max('max_iteration').alias('max_iteration'))
                             .collect()[0]['max_iteration'])
    print('Number of iterations towards convergence: {}'.format(convergence_iteration))
    label_context_set('initial_label', dict_initial_label)
    # df_transition_matrix.show(truncate=False)
    # print("iteration -1 label values \n{}\n{}\n".format(*label_context.constants['initial_label'].value.items()))
    iters = 0
    while iters < convergence_iteration:
        iters+=1
        udf_dot = F.udf(lambda l: _multiply_labels(
            label= l,
            broadcast_label= label_context.constants['initial_label'].value,
            k= label_context.constants['k'].value),
                        T.ArrayType(T.DoubleType())
                        )

        clamping_expr = F.when(F.col('is_clamped'), F.col('initial_label')).otherwise(udf_dot('row_trans'))
        df_transition_matrix = (
            df_transition_matrix.withColumn(colName= 'new_label', col= clamping_expr)
                .orderBy('row').cache()
        )

        # convert the column with labels to a dictionary with vector label
        dict_initial_label = generate_label(
            context= label_context, data_frame= df_transition_matrix,
            label_weights= 'new_label'
        )

        # Check to see if some of the new labels can be clamped
        df_transition_matrix = (df_transition_matrix.withColumn(
            colName= 'initial_label', col= F.col('new_label'))
                                .drop('new_label')
                                )

        label_context_set('initial_label', dict_initial_label)
        # print("iteration {} label values \n{}\n{}\n".format(iters, *label_context.constants['initial_label'].value.items()))
        df_transition_matrix.unpersist()

    if eval_type == 'cmn':
        df_transition_matrix =  class_mass_normalization(label_context, df_transition_matrix)

    udf_descision_selection = F.udf(
            lambda x: float(np.argmax(np.array(x))), T.DoubleType())
    return df_transition_matrix.withColumn(
            colName= 'label', col= udf_descision_selection(F.col('initial_label')))
