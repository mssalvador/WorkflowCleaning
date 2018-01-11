from pyspark.mllib.linalg import distributed
import labelpropagation


def labelpropation(sc, data_frame=None, id_col='id', label_col='label', feature_col=None, **kwargs):
    """
    New Version of Labelpropagation with sparks matrix lib used
    :param sc:
    :param data_frame:
    :param id_col:
    :param label_col:
    :param feature_col:
    :param kwargs: iterations, tol, standardize, sigma, priors, evaluation_type
    :return:
    """
    n = data_frame.count()
    iterations = kwargs.get('iterations', 25)
    cartesian_demon_rdd = labelpropagation.do_cartesian(
        sc=sc, df=data_frame, id_col=id_col, feature_col=feature_col, **kwargs).cache()

    demon_matrix = distributed.CoordinateMatrix(entries=cartesian_demon_rdd, numRows=n, numCols=n)
    row_summed_matrix = demon_matrix.entries.flatMap(labelpropagation.lp_helper.triangle_mat_summation).reduceByKey(
        lambda x, y: x + y).collectAsMap()
    bc_row_summed = sc.broadcast(row_summed_matrix)
    # print(type(bc_row_summed.value))

    transition_rdd = demon_matrix.entries.map(
        lambda x: distributed.MatrixEntry(
            i=x.i, j=x.j, value=x.value / bc_row_summed.value.get(x.j))
    )
    col_summed_matrix = (transition_rdd.flatMap(labelpropagation.lp_helper.triangle_mat_summation)
        .reduceByKey(lambda x, y: x + y).collectAsMap())
    bc_col_summed = sc.broadcast(col_summed_matrix)

    hat_transition_rdd = transition_rdd.map(
        lambda x: distributed.MatrixEntry(
            i=x.i, j=x.j, value=x.value / bc_col_summed.value.get(x.i))
    )

    intial_y_matrix = labelpropagation.lp_helper.generate_label_matrix(df=data_frame)
    final_label_matrix = labelpropagation.propagation_step(
        transition_matrix=hat_transition_rdd, label_matrix=intial_y_matrix, max_iterations=iterations)
    output_data_frame = labelpropagation.lp_helper.merge_data_with_label(
        sc=sc, org_data_frame=data_frame, label_rdd=final_label_matrix, id_col=id_col)

    return output_data_frame
